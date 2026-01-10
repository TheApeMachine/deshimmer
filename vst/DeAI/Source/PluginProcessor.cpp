/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DeAIAudioProcessor::DeAIAudioProcessor()
    :
#ifndef JucePlugin_PreferredChannelConfigurations
      AudioProcessor (BusesProperties()
                    #if ! JucePlugin_IsMidiEffect
                     #if ! JucePlugin_IsSynth
                      .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                     #endif
                      .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                    #endif
                      ),
#endif
      apvts (*this, nullptr, "PARAMS", createParameterLayout())
{
}

DeAIAudioProcessor::~DeAIAudioProcessor()
{
}

//==============================================================================
const juce::String DeAIAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool DeAIAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool DeAIAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool DeAIAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double DeAIAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int DeAIAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int DeAIAudioProcessor::getCurrentProgram()
{
    return 0;
}

void DeAIAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String DeAIAudioProcessor::getProgramName (int index)
{
    return {};
}

void DeAIAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
DeAIAudioProcessor::FFTProcessor::FFTProcessor() :
    fft(fftOrder),
    window(fftSize + 1, juce::dsp::WindowingFunction<float>::WindowingMethod::hann, false)
{
    // Window is length fftSize+1 to make it periodic; use first fftSize samples.
}

void DeAIAudioProcessor::FFTProcessor::reset()
{
    count = 0;
    pos = 0;
    inputFifo.fill (0.0f);
    outputFifo.fill (0.0f);
    persistDb.fill (0.0f);
    prevBandDb = -100.0f;
}

float DeAIAudioProcessor::FFTProcessor::processSample (float sample,
                                                       bool bypassed,
                                                       float sampleRate,
                                                       float startHz,
                                                       float endHz,
                                                       int freqMedBins,
                                                       float thrDb,
                                                       float slope,
                                                       float strength,
                                                       float maxAttDb,
                                                       float persistMs,
                                                       float persistThrDb,
                                                       float flatStart,
                                                       float flatEnd,
                                                       float densityLo,
                                                       float densityHi,
                                                       float fluxThrDb,
                                                       float fluxRangeDb)
{
    // Push input into FIFO
    inputFifo[(size_t) pos] = sample;

    // Read output from FIFO (delayed by fftSize)
    float outputSample = outputFifo[(size_t) pos];
    outputFifo[(size_t) pos] = 0.0f;

    // Advance circular position
    pos++;
    if (pos == fftSize)
        pos = 0;

    // Process frame every hopSize samples
    count++;
    if (count == hopSize)
    {
        count = 0;
        processFrame (bypassed, sampleRate, startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb);
    }

    return outputSample;
}

void DeAIAudioProcessor::FFTProcessor::processFrame (bool bypassed,
                                                    float sampleRate,
                                                    float startHz,
                                                    float endHz,
                                                    int freqMedBins,
                                                    float thrDb,
                                                    float slope,
                                                    float strength,
                                                    float maxAttDb,
                                                    float persistMs,
                                                    float persistThrDb,
                                                    float flatStart,
                                                    float flatEnd,
                                                    float densityLo,
                                                    float densityHi,
                                                    float fluxThrDb,
                                                    float fluxRangeDb)
{
    const float* inputPtr = inputFifo.data();
    float* fftPtr = fftData.data();

    // Copy circular FIFO into linear fft buffer in two parts
    std::memcpy (fftPtr, inputPtr + pos, (fftSize - pos) * sizeof (float));
    if (pos > 0)
        std::memcpy (fftPtr + (fftSize - pos), inputPtr, (size_t) pos * sizeof (float));

    // Analysis window
    window.multiplyWithWindowingTable (fftPtr, fftSize);

    // Forward FFT
    fft.performRealOnlyForwardTransform (fftPtr);

    if (! bypassed)
        processSpectrum (fftPtr, sampleRate, startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb);

    // Inverse FFT
    fft.performRealOnlyInverseTransform (fftPtr);

    // Synthesis window
    window.multiplyWithWindowingTable (fftPtr, fftSize);

    // Normalise for overlap-add using Hann @ 75% overlap
    for (int i = 0; i < fftSize; ++i)
        fftPtr[i] *= windowCorrection / (float) fftSize; // JUCE IFFT is unnormalised

    // Add into output FIFO at current pos (wrap)
    for (int i = 0; i < pos; ++i)
        outputFifo[(size_t) i] += fftPtr[i + (fftSize - pos)];
    for (int i = 0; i < (fftSize - pos); ++i)
        outputFifo[(size_t) (i + pos)] += fftPtr[i];
}

void DeAIAudioProcessor::FFTProcessor::processSpectrum (float* data,
                                                       float sampleRate,
                                                       float startHz,
                                                       float endHz,
                                                       int freqMedBins,
                                                       float thrDb,
                                                       float slope,
                                                       float strength,
                                                       float maxAttDb,
                                                       float persistMs,
                                                       float persistThrDb,
                                                       float flatStart,
                                                       float flatEnd,
                                                       float densityLo,
                                                       float densityHi,
                                                       float fluxThrDb,
                                                       float fluxRangeDb)
{
    auto* cdata = reinterpret_cast<std::complex<float>*> (data);

    const float sr = (sampleRate > 1.0f ? sampleRate : 48000.0f);
    const float thr = juce::jmax (0.0f, thrDb);
    const float sl = juce::jlimit (0.25f, 2.0f, slope);
    const float st = juce::jlimit (0.0f, 1.0f, strength);
    const float maxAttLin = std::pow (10.0f, -juce::jmax (0.0f, maxAttDb) / 20.0f);

    const float loHz = juce::jmin (startHz, endHz);
    const float hiHz = juce::jmax (startHz, endHz);

    const int k0 = juce::jlimit (1, numBins - 2, (int) std::floor (loHz * (float) fftSize / sr));
    const int k1 = juce::jlimit (1, numBins - 2, (int) std::floor (hiHz * (float) fftSize / sr));

    int medBinsN = juce::jlimit (3, 101, (freqMedBins | 1));
    const int half = medBinsN / 2;

    const float hopSeconds = (float) hopSize / sr;
    const float pMs = juce::jmax (0.0f, persistMs);
    const float pThr = juce::jmax (0.0f, persistThrDb);
    const float decay = (pMs > 1.0f ? std::exp (-hopSeconds / (pMs / 1000.0f)) : 0.0f);

    // Magnitudes
    for (int k = 0; k < numBins; ++k)
        magBins[(size_t) k] = std::abs (cdata[k]);

    const float eps = 1.0e-12f;

    // --- 1. Compute Band Properties (Flatness, Energy, Flux) ---
    float sumP = 0.0f;
    float sumLogP = 0.0f;
    int countBins = 0;

    for (int k = k0; k <= k1; ++k)
    {
        float mag = magBins[k];
        float p = mag * mag + eps;
        sumP += p;
        sumLogP += std::log (p);
        countBins++;
    }

    if (countBins < 1) countBins = 1;
    float meanP = sumP / (float) countBins;
    float meanLogP = sumLogP / (float) countBins;

    // Flatness (GeoMean / ArithMean)
    // GeoMean = exp(mean(log(x)))
    float flat = std::exp (meanLogP) / (meanP + eps);
    float wNoise = juce::jlimit (0.0f, 1.0f, (flat - flatStart) / juce::jmax (1.0e-6f, flatEnd - flatStart));

    // Band Energy dB
    float bandDb = 10.0f * std::log10 (meanP + eps);

    // Flux (Transient protection)
    float flux = 0.0f;
    if (prevBandDb > -99.0f)
        flux = juce::jmax (0.0f, bandDb - prevBandDb);
    prevBandDb = bandDb;

    float wTrans = juce::jlimit (0.0f, 1.0f, (flux - fluxThrDb) / juce::jmax (1.0e-6f, fluxRangeDb));
    float wNonTrans = 1.0f - wTrans;

    // --- 2. Calculate Density (first pass) ---
    int overCount = 0;
    for (int k = k0; k <= k1; ++k)
    {
        for (int j = 0; j < medBinsN; ++j)
        {
            const int kk = juce::jlimit (0, numBins - 1, k + (j - half));
            medScratch[(size_t) j] = magBins[(size_t) kk];
        }
        float* begin = medScratch.data();
        float* mid = begin + (medBinsN / 2);
        std::nth_element (begin, mid, begin + medBinsN);
        const float base = juce::jmax (eps, *mid);

        const float mag = magBins[(size_t) k];
        const float residDb = 20.0f * std::log10 ((mag + eps) / base);

        if (residDb > thr)
            overCount++;
    }

    float density = (float) overCount / (float) countBins;
    float wNarrow = 1.0f - juce::jlimit (0.0f, 1.0f, (density - densityLo) / juce::jmax (1.0e-6f, densityHi - densityLo));

    // Combined Depth
    float depth = wNoise * wNonTrans * wNarrow;

    // Edge taper length in Hz (default 200)
    float edgeHz = 200.0f;

    // --- 3. Apply Processing ---
    for (int k = k0; k <= k1; ++k)
    {
        // Re-calculate median (CPU tradeoff vs memory)
        for (int j = 0; j < medBinsN; ++j)
        {
            const int kk = juce::jlimit (0, numBins - 1, k + (j - half));
            medScratch[(size_t) j] = magBins[(size_t) kk];
        }
        float* begin = medScratch.data();
        float* mid = begin + (medBinsN / 2);
        std::nth_element (begin, mid, begin + medBinsN);
        const float base = juce::jmax (eps, *mid);

        const float mag = magBins[(size_t) k];
        const float residDb = 20.0f * std::log10 ((mag + eps) / base);
        const float overDb = residDb - thr;

        if (overDb <= 0.0f)
        {
            // decay persistence
            if (pMs > 1.0f)
                persistDb[(size_t) k] *= decay;
            else
                persistDb[(size_t) k] = 0.0f;
            continue;
        }

        // Persistence (instant rise, slow fall)
        if (pMs > 1.0f)
            persistDb[(size_t) k] = juce::jmax (overDb, persistDb[(size_t) k] * decay);
        else
            persistDb[(size_t) k] = overDb;

        float gate = 1.0f;
        if (pThr > 1.0e-6f && pMs > 1.0f)
            gate = juce::jlimit (0.0f, 1.0f, persistDb[(size_t) k] / pThr);

        // Slope is a multiplier in dB domain, not an exponent
        const float shaped = sl * overDb;
        const float attDb = st * gate * shaped;
        float g = std::pow (10.0f, -attDb / 20.0f);
        g = juce::jlimit (maxAttLin, 1.0f, g);

        // Edge taper weight
        float freq = (float) k * sr / (float) fftSize;
        float wEdge = 1.0f;
        if (freq < startHz + edgeHz)
            wEdge = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::pi * juce::jlimit(0.0f, 1.0f, (freq - startHz) / edgeHz));
        else if (freq > endHz - edgeHz)
            wEdge = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::pi * juce::jlimit(0.0f, 1.0f, (endHz - freq) / edgeHz));

        // Final effective gain
        float gEff = 1.0f - (depth * wEdge) * (1.0f - g);

        cdata[k] *= gEff;
    }
}

//==============================================================================
void DeAIAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    juce::ignoreUnused (samplesPerBlock);
    fftProc[0].reset();
    fftProc[1].reset();

    // Report algorithmic latency so hosts (e.g. Logic) can apply PDC.
    // With overlap-add STFT, a good practical latency is (fftSize - hopSize).
    latencySamps = FFTProcessor::fftSize;
    setLatencySamples (latencySamps);

    // Allocate dry delay line to match reported latency (keeps Mix/Delta sane).
    const int numCh = juce::jmax (1, getTotalNumOutputChannels());
    dryDelaySize = juce::jmax (1, latencySamps + juce::jmax (samplesPerBlock, 1) + 8);
    dryWritePos = 0;
    dryDelay.assign ((size_t) numCh, std::vector<float> ((size_t) dryDelaySize, 0.0f));
}

void DeAIAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
    setLatencySamples (0);
    latencySamps = 0;
    dryDelaySize = 0;
    dryWritePos = 0;
    dryDelay.clear();
}

juce::AudioProcessorValueTreeState::ParameterLayout DeAIAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // NOTE: If AAX is enabled in the project, JUCE requires non-zero version hints
    // for all params (to avoid automation breakage). Using versionHint=1 satisfies
    // the requirement and removes the runtime jassert.
    layout.add (std::make_unique<juce::AudioParameterBool> (juce::ParameterID { ParamIDs::bypass, 1 }, "Bypass", false));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::mix, 1 },
        "Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f),
        1.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::startHz, 1 },
        "Start Hz",
        juce::NormalisableRange<float> (1000.0f, 12000.0f, 1.0f, 0.5f),
        5100.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::endHz, 1 },
        "End Hz",
        juce::NormalisableRange<float> (1500.0f, 20000.0f, 1.0f, 0.5f),
        7200.0f));

    layout.add (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID { ParamIDs::freqMedBins, 1 },
        "Freq Median Bins",
        3,
        101,
        61));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::thrDb, 1 },
        "Residual Threshold dB",
        juce::NormalisableRange<float> (0.0f, 24.0f, 0.1f),
        8.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::slope, 1 },
        "Slope",
        juce::NormalisableRange<float> (0.25f, 2.0f, 0.01f),
        0.6f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::strength, 1 },
        "Strength",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f),
        0.8f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::maxAttDb, 1 },
        "Max Atten dB",
        juce::NormalisableRange<float> (0.0f, 36.0f, 0.1f),
        12.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::persistMs, 1 },
        "Persistence ms",
        juce::NormalisableRange<float> (0.0f, 2000.0f, 1.0f),
        800.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::persistThrDb, 1 },
        "Persist Thr dB",
        juce::NormalisableRange<float> (0.0f, 12.0f, 0.1f),
        2.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::flatStart, 1 },
        "Flatness Start",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f),
        0.25f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::flatEnd, 1 },
        "Flatness End",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f),
        0.70f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::densityLo, 1 },
        "Density Lo",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f),
        0.02f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::densityHi, 1 },
        "Density Hi",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f),
        0.15f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::fluxThrDb, 1 },
        "Flux Thr dB",
        juce::NormalisableRange<float> (0.0f, 24.0f, 0.1f),
        6.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { ParamIDs::fluxRangeDb, 1 },
        "Flux Range dB",
        juce::NormalisableRange<float> (0.1f, 24.0f, 0.1f),
        8.0f));

    layout.add (std::make_unique<juce::AudioParameterBool> (juce::ParameterID { ParamIDs::deltaListen, 1 }, "Delta Listen (Removed Only)", false));
    return layout;
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool DeAIAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    const auto in = layouts.getMainInputChannelSet();
    const auto out = layouts.getMainOutputChannelSet();

    // Allow:
    // - mono -> mono (mono tracks)
    // - mono -> stereo (mono tracks widened to stereo bus)
    // - stereo -> stereo (stereo tracks / master bus)
    if (in == juce::AudioChannelSet::mono())
        return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo());

    if (in == juce::AudioChannelSet::stereo())
        return (out == juce::AudioChannelSet::stereo());

    return false;
   #endif

    return true;
  #endif
}
#endif

void DeAIAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused (midiMessages);
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // If host instantiated mono->stereo, duplicate mono into R so the plugin behaves sensibly.
    if (totalNumInputChannels == 1 && totalNumOutputChannels >= 2)
        buffer.copyFrom (1, 0, buffer, 0, 0, buffer.getNumSamples());

    const bool bypass = apvts.getRawParameterValue (ParamIDs::bypass)->load() > 0.5f;
    const bool deltaListen = apvts.getRawParameterValue (ParamIDs::deltaListen)->load() > 0.5f;
    const float mix = juce::jlimit (0.0f, 1.0f, apvts.getRawParameterValue (ParamIDs::mix)->load());

    const float startHz = apvts.getRawParameterValue (ParamIDs::startHz)->load();
    const float endHz   = apvts.getRawParameterValue (ParamIDs::endHz)->load();
    const int freqMedBins = (int) apvts.getRawParameterValue (ParamIDs::freqMedBins)->load();
    const float thrDb   = apvts.getRawParameterValue (ParamIDs::thrDb)->load();
    const float slope   = apvts.getRawParameterValue (ParamIDs::slope)->load();
    const float strength = apvts.getRawParameterValue (ParamIDs::strength)->load();
    const float maxAttDb = apvts.getRawParameterValue (ParamIDs::maxAttDb)->load();
    const float persistMs = apvts.getRawParameterValue (ParamIDs::persistMs)->load();
    const float persistThrDb = apvts.getRawParameterValue (ParamIDs::persistThrDb)->load();

    const float flatStart = apvts.getRawParameterValue (ParamIDs::flatStart)->load();
    const float flatEnd = apvts.getRawParameterValue (ParamIDs::flatEnd)->load();
    const float densityLo = apvts.getRawParameterValue (ParamIDs::densityLo)->load();
    const float densityHi = apvts.getRawParameterValue (ParamIDs::densityHi)->load();
    const float fluxThrDb = apvts.getRawParameterValue (ParamIDs::fluxThrDb)->load();
    const float fluxRangeDb = apvts.getRawParameterValue (ParamIDs::fluxRangeDb)->load();

    const int numCh = buffer.getNumChannels();
    const int numSamps = buffer.getNumSamples();

    // Ensure dry alignment buffer exists
    if (dryAlignedBuf.getNumChannels() != numCh || dryAlignedBuf.getNumSamples() < numSamps)
        dryAlignedBuf.setSize (numCh, numSamps, false, false, true);

    const bool bypassed = (bypass || strength <= 1.0e-6f);

    // Host edge-cases: processBlock can be called before prepareToPlay/releaseResources juggling.
    // Never allow modulo by zero; fall back to "no alignment" if dry delay isn't ready.
    const bool dryDelayReady = (dryDelaySize > 0 && (int) dryDelay.size() >= juce::jmax (1, numCh));
    if (! dryDelayReady)
    {
        for (int ch = 0; ch < numCh; ++ch)
            buffer.clear (ch, 0, numSamps);

        // Pass-through (with STFT latency unaccounted) so audio keeps flowing.
        for (int i = 0; i < numSamps; ++i)
        {
            const float inL = buffer.getReadPointer (0)[i];
            const float inR = (numCh > 1 ? buffer.getReadPointer (1)[i] : inL);

            const float wetL = fftProc[0].processSample (inL, bypassed, (float) getSampleRate(), startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb);
            const float wetR = (numCh > 1)
                ? fftProc[1].processSample (inR, bypassed, (float) getSampleRate(), startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb)
                : wetL;

            float outL = wetL;
            float outR = wetR;

            if (deltaListen)
            {
                outL = inL - wetL;
                outR = inR - wetR;
            }
            else if (mix < 0.999f)
            {
                outL = mix * wetL + (1.0f - mix) * inL;
                outR = mix * wetR + (1.0f - mix) * inR;
            }

            buffer.getWritePointer (0)[i] = outL;
            if (numCh > 1)
                buffer.getWritePointer (1)[i] = outR;
        }

        return;
    }

    // Sample-by-sample processing (stable STFT output FIFO; no "outHop + zeros" gaps)
    for (int i = 0; i < numSamps; ++i)
    {
        // Dry alignment ring write/read
        const int w = dryWritePos;
        const int r = (w - latencySamps + dryDelaySize) % dryDelaySize;

        float inL = buffer.getReadPointer (0)[i];
        float inR = (numCh > 1 ? buffer.getReadPointer (1)[i] : inL);

        if ((int) dryDelay.size() >= 1 && dryDelaySize > 0)
        {
            dryDelay[0][(size_t) w] = inL;
            dryAlignedBuf.getWritePointer (0)[i] = dryDelay[0][(size_t) r];
            if (numCh > 1)
            {
                dryDelay[1][(size_t) w] = inR;
                dryAlignedBuf.getWritePointer (1)[i] = dryDelay[1][(size_t) r];
            }
        }
        else
        {
            dryAlignedBuf.getWritePointer (0)[i] = inL;
            if (numCh > 1) dryAlignedBuf.getWritePointer (1)[i] = inR;
        }

        // Wet STFT (per-channel)
        const float wetL = fftProc[0].processSample (inL, bypassed, (float) getSampleRate(), startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb);
        const float wetR = (numCh > 1)
            ? fftProc[1].processSample (inR, bypassed, (float) getSampleRate(), startHz, endHz, freqMedBins, thrDb, slope, strength, maxAttDb, persistMs, persistThrDb, flatStart, flatEnd, densityLo, densityHi, fluxThrDb, fluxRangeDb)
            : wetL;

        float outL = wetL;
        float outR = wetR;

        const float dryL = dryAlignedBuf.getReadPointer (0)[i];
        const float dryR = (numCh > 1 ? dryAlignedBuf.getReadPointer (1)[i] : dryL);

        if (deltaListen)
        {
            outL = dryL - wetL;
            outR = dryR - wetR;
        }
        else if (mix < 0.999f)
        {
            outL = mix * wetL + (1.0f - mix) * dryL;
            outR = mix * wetR + (1.0f - mix) * dryR;
        }

        buffer.getWritePointer (0)[i] = outL;
        if (numCh > 1)
            buffer.getWritePointer (1)[i] = outR;

        dryWritePos = (dryWritePos + 1) % dryDelaySize;
    }
}

//==============================================================================
bool DeAIAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* DeAIAudioProcessor::createEditor()
{
    return new DeAIAudioProcessorEditor (*this);
}

//==============================================================================
void DeAIAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void DeAIAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    if (auto xmlState = getXmlFromBinary (data, sizeInBytes))
        apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeAIAudioProcessor();
}
