#!/usr/bin/env python3
# ruff: noqa
"""
realtime_player.py

“Real-time-ish” end-to-end playback for experimentation:
- Plays the entire song (no looping) through your audio device
- Renders processed audio *ahead of playback* in a background thread
- Hot-reloads params when `realtime_params.json` changes, then crossfades

This is NOT a DAW-grade real-time plugin:
- uses chunked offline processing (master.py pipeline) to keep code reuse high
- if your settings are too heavy, it may underrun (you’ll hear dropouts)

Usage:
  pip install -r requirements-rt.txt
  python realtime_player.py input.wav

Optional:
  python realtime_player.py input.wav --params realtime_params.json
"""

from __future__ import annotations

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnusedCallResult=false

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import asdict

import numpy as np
import soundfile as sf

import master
from deshimmer_api import process_audio


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[:, None] if x.ndim == 1 else x


def _clip_audio(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)


def load_params_json(path: str) -> tuple[master.Params, master.MasterParams]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    p = master.Params(**obj.get("params", {}))
    mp = master.MasterParams(**obj.get("master_params", {}))
    return p, mp


def default_params() -> tuple[master.Params, master.MasterParams]:
    return master.Params(), master.MasterParams(enabled=False)


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a2 = _as_2d(a)
    b2 = _as_2d(b)
    n = int(max(0, min(n, a2.shape[0], b2.shape[0])))
    if n <= 0:
        return b2
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    b2 = b2.copy()
    b2[:n, :] = (1.0 - w) * a2[-n:, :] + w * b2[:n, :]
    return b2


class ParamWatcher(threading.Thread):
    def __init__(self, path: str, poll_s: float = 0.25):
        super().__init__(daemon=True)
        self.path: str = path
        self.poll_s: float = float(max(0.05, poll_s))
        self._stop: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._mtime: float | None = None
        self._latest: tuple[master.Params, master.MasterParams] | None = None

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[master.Params, master.MasterParams] | None:
        with self._lock:
            return self._latest

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                if os.path.exists(self.path):
                    mtime = os.path.getmtime(self.path)
                    if self._mtime is None or mtime > self._mtime:
                        p, mp = load_params_json(self.path)
                        with self._lock:
                            self._latest = (p, mp)
                        self._mtime = mtime
            except Exception:
                # ignore parse errors while file is being written
                pass
            time.sleep(self.poll_s)


class Renderer(threading.Thread):
    def __init__(
        self,
        x: np.ndarray,
        sr: int,
        params_path: str,
        out_q: "queue.Queue[np.ndarray]",
        *,
        chunk_s: float = 4.0,
        xfade_s: float = 0.05,
        ahead_chunks: int = 2,
    ):
        super().__init__(daemon=True)
        self.x: np.ndarray = _as_2d(x)
        self.sr: int = int(sr)
        self.params_path: str = params_path
        self.out_q: "queue.Queue[np.ndarray]" = out_q
        self.chunk: int = int(max(1024, round(float(chunk_s) * self.sr)))
        self.xfade: int = int(max(0, round(float(xfade_s) * self.sr)))
        self.ahead_chunks: int = int(max(1, ahead_chunks))
        self._stop: threading.Event = threading.Event()

        self._lock: threading.Lock = threading.Lock()
        self._playhead: int = 0  # in samples
        self._gen_id: int = 0

        self._watcher: ParamWatcher = ParamWatcher(params_path)

    def stop(self) -> None:
        self._stop.set()
        self._watcher.stop()

    def set_playhead(self, s: int) -> None:
        with self._lock:
            self._playhead = int(max(0, s))

    def bump_gen(self) -> None:
        with self._lock:
            self._gen_id += 1

    def run(self) -> None:
        self._watcher.start()
        p, mp = default_params()
        prev_tail: np.ndarray | None = None
        last_params_fingerprint: str | None = None

        while not self._stop.is_set():
            # keep queue from running too far ahead
            if self.out_q.qsize() >= self.ahead_chunks:
                time.sleep(0.01)
                continue

            with self._lock:
                s0 = int(self._playhead)

            latest = self._watcher.latest()
            if latest is not None:
                p_new, mp_new = latest
                fp = json.dumps({"p": asdict(p_new), "mp": asdict(mp_new)}, sort_keys=True)
                if last_params_fingerprint != fp:
                    # parameter change: restart rendering from current playhead
                    p, mp = p_new, mp_new
                    last_params_fingerprint = fp
                    prev_tail = None
                    print("!!! Param change detected - flushing queue & re-rendering from playhead ...")
                    # clear queue
                    try:
                        while True:
                            self.out_q.get_nowait()
                    except queue.Empty:
                        pass

            if s0 >= self.x.shape[0]:
                # end of file
                break

            s1 = min(self.x.shape[0], s0 + self.chunk)
            seg = self.x[s0:s1, :]

            # Process this chunk with the full pipeline (reuse master.py)
            y, _ = process_audio(seg, self.sr, params=p, master_params=mp, debug_params=master.DebugParams(enabled=False))
            y = _clip_audio(_as_2d(y))

            # crossfade against previous tail to reduce chunk seams (especially after param changes)
            if prev_tail is not None and self.xfade > 0:
                y = crossfade(prev_tail, y, self.xfade)
            prev_tail = y

            # If playhead moved due to callback while we processed, we may be behind; still enqueue.
            # (This is a “best effort” player.)
            self.out_q.put(y)


def main() -> int:
    ap = argparse.ArgumentParser(description="Play a file end-to-end while rendering master.py processing ahead of playback.")
    ap.add_argument("input", help="Input audio file")
    ap.add_argument("--params", default="realtime_params.json", help="JSON file written by UI to control params live")
    ap.add_argument("--chunk-s", type=float, default=0.5, help="Chunk size in seconds (bigger = safer, more latency)")
    ap.add_argument("--xfade-s", type=float, default=0.03, help="Crossfade seconds between chunks")
    ap.add_argument("--blocksize", type=int, default=256, help="Audio callback block size")
    ap.add_argument("--ahead-chunks", type=int, default=2, help="How many chunks to keep pre-rendered (lower = more responsive)")
    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=True)
    x = x.astype(np.float32, copy=False)

    out_q: "queue.Queue[np.ndarray]" = queue.Queue()
    renderer = Renderer(
        x,
        sr,
        args.params,
        out_q,
        chunk_s=float(args.chunk_s),
        xfade_s=float(args.xfade_s),
        ahead_chunks=int(args.ahead_chunks),
    )

    play_lock = threading.Lock()
    play_samp = 0
    cur_block: np.ndarray | None = None
    cur_off = 0

    def callback(outdata: np.ndarray, frames: int, _time: object, status: object) -> None:
        nonlocal play_samp, cur_block, cur_off
        if status:
            # Dropouts/underruns are expected if processing can’t keep up.
            pass

        out = np.zeros((frames, x.shape[1]), dtype=np.float32)
        filled = 0
        while filled < frames:
            if cur_block is None or cur_off >= cur_block.shape[0]:
                try:
                    cur_block = out_q.get_nowait()
                    cur_off = 0
                except queue.Empty:
                    break

            take = min(frames - filled, cur_block.shape[0] - cur_off)
            out[filled : filled + take, :] = cur_block[cur_off : cur_off + take, :]
            cur_off += take
            filled += take

        outdata[:] = out
        with play_lock:
            play_samp += frames
            renderer.set_playhead(play_samp)

    renderer.start()

    print(f"Playing: {args.input}")
    print(f"Watching params: {args.params} (edit/export it to retune live)")
    print("Tip: in the UI, export your current knobs to realtime_params.json")

    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        raise SystemExit(f"sounddevice not available: {e}")

    with sd.OutputStream(
        samplerate=sr,
        channels=int(x.shape[1]),
        dtype="float32",
        blocksize=int(args.blocksize),
        callback=callback,
    ):
        # keep alive until playback ends
        while True:
            with play_lock:
                done = play_samp >= x.shape[0]
            if done:
                break
            time.sleep(0.1)

    renderer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


