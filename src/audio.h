#pragma once

#include <vector>
#include <atomic>
#include <thread>

// Audio capture and FFT analysis for reactive wave simulation
class AudioCapture {
public:
    static constexpr int FFT_SIZE = 1024;
    static constexpr int NUM_BANDS = 8;  // Frequency bands (matches max sources)

    AudioCapture();
    ~AudioCapture();

    // Start/stop audio capture thread
    bool start();
    void stop();
    bool isRunning() const { return m_running; }

    // Get current band levels (0-1 range, thread-safe)
    // Band mapping: 0=bass, 7=treble
    void getBandLevels(float* levels, int numBands) const;

    // Get overall volume level (0-1)
    float getVolume() const;

    // Get dominant frequency in Hz
    float getDominantFrequency() const;

private:
    void captureThread();
    void processFFT(const float* samples, int numSamples);

    std::thread m_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};

    // FFT output bands (updated by capture thread)
    mutable std::atomic<float> m_bandLevels[NUM_BANDS];
    mutable std::atomic<float> m_volume{0.0f};
    mutable std::atomic<float> m_dominantFreq{100.0f};

    // Sample rate from WASAPI
    int m_sampleRate = 48000;
};
