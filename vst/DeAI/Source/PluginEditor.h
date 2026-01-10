/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class DeAIAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    DeAIAudioProcessorEditor (DeAIAudioProcessor&);
    ~DeAIAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    DeAIAudioProcessor& audioProcessor;

    juce::ToggleButton bypassButton { "Bypass" };
    juce::ToggleButton deltaButton { "Delta (Removed Only)" };

    juce::Slider mixSlider;
    juce::Slider startHzSlider;
    juce::Slider endHzSlider;
    juce::Slider freqMedBinsSlider;
    juce::Slider thrDbSlider;
    juce::Slider slopeSlider;
    juce::Slider strengthSlider;
    juce::Slider maxAttDbSlider;
    juce::Slider persistMsSlider;
    juce::Slider persistThrDbSlider;

    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    std::unique_ptr<Attachment> mixAttachment;
    std::unique_ptr<Attachment> startHzAttachment;
    std::unique_ptr<Attachment> endHzAttachment;
    std::unique_ptr<Attachment> freqMedBinsAttachment;
    std::unique_ptr<Attachment> thrDbAttachment;
    std::unique_ptr<Attachment> slopeAttachment;
    std::unique_ptr<Attachment> strengthAttachment;
    std::unique_ptr<Attachment> maxAttDbAttachment;
    std::unique_ptr<Attachment> persistMsAttachment;
    std::unique_ptr<Attachment> persistThrDbAttachment;
    std::unique_ptr<ButtonAttachment> bypassAttachment;
    std::unique_ptr<ButtonAttachment> deltaAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeAIAudioProcessorEditor)
};
