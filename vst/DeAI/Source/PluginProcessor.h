/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>

//==============================================================================
/**
*/
class DeAIAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    DeAIAudioProcessor();
    ~DeAIAudioProcessor() override;

    struct ParamIDs
    {
        static constexpr const char* bypass      = "bypass";
        static constexpr const char* mix         = "mix";
        static constexpr const char* startHz     = "startHz";
        static constexpr const char* endHz       = "endHz";
        static constexpr const char* freqMedBins = "freqMedBins";
        static constexpr const char* thrDb       = "thrDb";
        static constexpr const char* slope       = "slope";
        static constexpr const char* strength    = "strength";
        static constexpr const char* maxAttDb    = "maxAttDb";
        static constexpr const char* persistMs   = "persistMs";
        static constexpr const char* persistThrDb = "persistThrDb";
        static constexpr const char* deltaListen = "deltaListen";

        static constexpr const char* flatStart   = "flatStart";
        static constexpr const char* flatEnd     = "flatEnd";
        static constexpr const char* densityLo   = "densityLo";
        static constexpr const char* densityHi   = "densityHi";
        static constexpr const char* fluxThrDb   = "fluxThrDb";
        static constexpr const char* fluxRangeDb = "fluxRangeDb";
    };

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    //==============================================================================
    struct FFTProcessor
    {
        // The FFT has 2^order points and fftSize/2 + 1 bins.
        static constexpr int fftOrder = 11;
        static constexpr int fftSize = 1 << fftOrder;      // 2048
        static constexpr int numBins = fftSize / 2 + 1;
        static constexpr int overlap = 4;                  // 75% overlap
        static constexpr int hopSize = fftSize / overlap;  // 512

        // Gain correction for using Hann window with 75% overlap (COLA).
        static constexpr float windowCorrection = 2.0f / 3.0f;

        FFTProcessor();

        int getLatencyInSamples() const { return fftSize; }

        void reset();
        float processSample (float sample,
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
                             float fluxRangeDb);

    private:
        void processFrame (bool bypassed,
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
                           float fluxRangeDb);

        void processSpectrum (float* data,
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
                              float fluxRangeDb);

        juce::dsp::FFT fft;
        juce::dsp::WindowingFunction<float> window;

        // Counts up until the next hop.
        int count = 0;
        // Write position in input FIFO and read position in output FIFO.
        int pos = 0;

        // Circular buffers for incoming and outgoing audio data.
        std::array<float, fftSize> inputFifo{};
        std::array<float, fftSize> outputFifo{};

        // Per-bin persistence (dB above threshold) for gating stationary whines.
        std::array<float, numBins> persistDb{};

        float prevBandDb = -100.0f;

        // FFT working space. Contains interleaved complex numbers.
        std::array<float, fftSize * 2> fftData{};

        // Scratch buffers for median baseline
        std::array<float, numBins> magBins{};
        std::array<float, 101> medScratch{}; // max median window
    };

    FFTProcessor fftProc[2];
    bool lastBypass = false;

    // Dry-path delay to time-align with the STFT latency (so Mix/Delta behave as expected).
    int latencySamps = 0;
    int dryDelaySize = 0;
    int dryWritePos = 0;
    std::vector<std::vector<float>> dryDelay; // [channel][sample]

    juce::AudioBuffer<float> dryAlignedBuf;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeAIAudioProcessor)
};
