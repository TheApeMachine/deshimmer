/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DeAIAudioProcessorEditor::DeAIAudioProcessorEditor (DeAIAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    auto initSlider = [] (juce::Slider& s, const juce::String& name)
    {
        s.setName (name);
        s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 18);
    };

    initSlider (mixSlider, "Mix");
    initSlider (startHzSlider, "Start Hz");
    initSlider (endHzSlider, "End Hz");
    initSlider (freqMedBinsSlider, "Freq Med");
    initSlider (thrDbSlider, "Residual Thr dB");
    initSlider (slopeSlider, "Slope");
    initSlider (strengthSlider, "Strength");
    initSlider (maxAttDbSlider, "Max Att dB");
    initSlider (persistMsSlider, "Persist ms");
    initSlider (persistThrDbSlider, "Persist Thr");

    addAndMakeVisible (bypassButton);
    addAndMakeVisible (deltaButton);
    addAndMakeVisible (mixSlider);
    addAndMakeVisible (startHzSlider);
    addAndMakeVisible (endHzSlider);
    addAndMakeVisible (freqMedBinsSlider);
    addAndMakeVisible (thrDbSlider);
    addAndMakeVisible (slopeSlider);
    addAndMakeVisible (strengthSlider);
    addAndMakeVisible (maxAttDbSlider);
    addAndMakeVisible (persistMsSlider);
    addAndMakeVisible (persistThrDbSlider);

    // Attachments
    bypassAttachment = std::make_unique<ButtonAttachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::bypass, bypassButton);
    deltaAttachment  = std::make_unique<ButtonAttachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::deltaListen, deltaButton);

    mixAttachment      = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::mix, mixSlider);
    startHzAttachment  = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::startHz, startHzSlider);
    endHzAttachment    = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::endHz, endHzSlider);
    freqMedBinsAttachment = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::freqMedBins, freqMedBinsSlider);
    thrDbAttachment    = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::thrDb, thrDbSlider);
    slopeAttachment    = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::slope, slopeSlider);
    strengthAttachment = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::strength, strengthSlider);
    maxAttDbAttachment = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::maxAttDb, maxAttDbSlider);
    persistMsAttachment = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::persistMs, persistMsSlider);
    persistThrDbAttachment = std::make_unique<Attachment> (audioProcessor.apvts, DeAIAudioProcessor::ParamIDs::persistThrDb, persistThrDbSlider);

    setSize (980, 360);
}

DeAIAudioProcessorEditor::~DeAIAudioProcessorEditor()
{
}

//==============================================================================
void DeAIAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions (16.0f));
    g.drawFittedText ("DeAI (prototype)", 10, 8, getWidth() - 20, 24, juce::Justification::centredLeft, 1);
}

void DeAIAudioProcessorEditor::resized()
{
    auto r = getLocalBounds().reduced (10);
    r.removeFromTop (30);

    auto top = r.removeFromTop (28);
    bypassButton.setBounds (top.removeFromLeft (120));
    deltaButton.setBounds (top.removeFromLeft (200));

    r.removeFromTop (10);
    auto row1 = r.removeFromTop (150);
    auto row2 = r.removeFromTop (150);

    auto placeRow = [] (juce::Rectangle<int> rr, std::initializer_list<juce::Component*> comps)
    {
        const int n = (int) comps.size();
        if (n <= 0) return;
        const int w = rr.getWidth() / n;
        int i = 0;
        for (auto* c : comps)
        {
            if (c != nullptr)
                c->setBounds (rr.removeFromLeft (w).reduced (6));
            ++i;
        }
    };

    placeRow (row1, { &mixSlider, &startHzSlider, &endHzSlider, &freqMedBinsSlider, &persistMsSlider });
    placeRow (row2, { &thrDbSlider, &slopeSlider, &strengthSlider, &maxAttDbSlider, &persistThrDbSlider });
}
