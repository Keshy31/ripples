#include "audio.h"

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <comdef.h>
#include <cmath>
#include <iostream>
#include <algorithm>

// Simple DFT for frequency analysis (no external FFT library needed)
// For real-time audio, DFT on small buffers is fast enough
void computeDFT(const float* input, float* magnitudes, int n, int sampleRate) {
    const float PI = 3.14159265359f;

    // Only compute first half (positive frequencies)
    int halfN = n / 2;
    for (int k = 0; k < halfN; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int t = 0; t < n; t++) {
            float angle = 2.0f * PI * k * t / n;
            real += input[t] * cosf(angle);
            imag -= input[t] * sinf(angle);
        }
        magnitudes[k] = sqrtf(real * real + imag * imag) / n;
    }
}

AudioCapture::AudioCapture() {
    // Initialize band levels to zero
    for (int i = 0; i < NUM_BANDS; i++) {
        m_bandLevels[i].store(0.0f);
    }
}

AudioCapture::~AudioCapture() {
    stop();
}

bool AudioCapture::start() {
    if (m_running) return true;

    m_shouldStop = false;
    m_running = true;
    m_thread = std::thread(&AudioCapture::captureThread, this);
    return true;
}

void AudioCapture::stop() {
    if (!m_running) return;

    m_shouldStop = true;
    if (m_thread.joinable()) {
        m_thread.join();
    }
    m_running = false;
}

void AudioCapture::getBandLevels(float* levels, int numBands) const {
    int count = (numBands < NUM_BANDS) ? numBands : NUM_BANDS;
    for (int i = 0; i < count; i++) {
        levels[i] = m_bandLevels[i].load();
    }
}

float AudioCapture::getVolume() const {
    return m_volume.load();
}

float AudioCapture::getDominantFrequency() const {
    return m_dominantFreq.load();
}

void AudioCapture::processFFT(const float* samples, int numSamples) {
    // Use FFT_SIZE samples for analysis
    int n = (numSamples < FFT_SIZE) ? numSamples : FFT_SIZE;
    if (n < 64) return;  // Need enough samples

    // Compute magnitudes via DFT
    std::vector<float> magnitudes(n / 2);
    computeDFT(samples, magnitudes.data(), n, m_sampleRate);

    // Calculate overall volume (RMS)
    float sum = 0.0f;
    for (int i = 0; i < numSamples; i++) {
        sum += samples[i] * samples[i];
    }
    float rms = sqrtf(sum / numSamples);
    m_volume.store(std::min(1.0f, rms * 5.0f));  // Scale for visibility

    // Find dominant frequency
    int halfN = n / 2;
    float maxMag = 0.0f;
    int maxBin = 1;
    for (int k = 1; k < halfN; k++) {
        if (magnitudes[k] > maxMag) {
            maxMag = magnitudes[k];
            maxBin = k;
        }
    }
    float dominantFreq = (float)maxBin * m_sampleRate / n;
    m_dominantFreq.store(dominantFreq);

    // Divide spectrum into bands (logarithmic spacing)
    // Band edges (Hz): 20, 60, 180, 540, 1600, 4800, 14400, 20000
    float bandEdges[NUM_BANDS + 1] = {20.0f, 60.0f, 180.0f, 540.0f, 1600.0f, 4800.0f, 14400.0f, 20000.0f, 22050.0f};

    for (int band = 0; band < NUM_BANDS; band++) {
        float lowFreq = bandEdges[band];
        float highFreq = bandEdges[band + 1];

        // Convert to bin indices
        int lowBin = (int)(lowFreq * n / m_sampleRate);
        int highBin = (int)(highFreq * n / m_sampleRate);
        lowBin = std::max(0, std::min(lowBin, halfN - 1));
        highBin = std::max(lowBin + 1, std::min(highBin, halfN));

        // Average magnitude in this band
        float bandSum = 0.0f;
        for (int k = lowBin; k < highBin; k++) {
            bandSum += magnitudes[k];
        }
        float avg = bandSum / (highBin - lowBin);

        // Smooth transition (exponential smoothing)
        float current = m_bandLevels[band].load();
        float smoothed = current * 0.7f + avg * 30.0f * 0.3f;  // Scale up for visibility
        m_bandLevels[band].store(std::min(1.0f, smoothed));
    }
}

void AudioCapture::captureThread() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    IMMDeviceEnumerator* pEnumerator = nullptr;
    IMMDevice* pDevice = nullptr;
    IAudioClient* pAudioClient = nullptr;
    IAudioCaptureClient* pCaptureClient = nullptr;
    WAVEFORMATEX* pwfx = nullptr;

    HRESULT hr;

    // Create device enumerator
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), (void**)&pEnumerator);
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Get default render device (for loopback capture)
    hr = pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Activate audio client
    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Get mix format
    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    m_sampleRate = pwfx->nSamplesPerSec;
    std::cout << "Audio capture: " << m_sampleRate << " Hz, " << pwfx->nChannels << " channels" << std::endl;

    // Initialize audio client in loopback mode
    hr = pAudioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK,
        10000000,  // 1 second buffer
        0,
        pwfx,
        nullptr
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Get capture client
    hr = pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to get capture client: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Start capture
    hr = pAudioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "Failed to start audio capture: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    std::cout << "Audio loopback capture started" << std::endl;

    // Capture loop
    {
        std::vector<float> monoSamples;
        monoSamples.reserve(FFT_SIZE * 2);

        while (!m_shouldStop) {
            Sleep(10);  // ~100 Hz update rate

            UINT32 packetLength = 0;
            hr = pCaptureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) continue;

            while (packetLength > 0) {
                BYTE* pData = nullptr;
                UINT32 numFrames = 0;
                DWORD flags = 0;

                hr = pCaptureClient->GetBuffer(&pData, &numFrames, &flags, nullptr, nullptr);
                if (FAILED(hr)) break;

                // Convert to mono float samples
                if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT) && pData != nullptr) {
                    // Assume float format (most common for WASAPI)
                    float* floatData = (float*)pData;
                    int channels = pwfx->nChannels;

                    for (UINT32 i = 0; i < numFrames; i++) {
                        float mono = 0.0f;
                        for (int ch = 0; ch < channels; ch++) {
                            mono += floatData[i * channels + ch];
                        }
                        mono /= channels;
                        monoSamples.push_back(mono);
                    }
                }

                pCaptureClient->ReleaseBuffer(numFrames);

                // Process FFT when we have enough samples
                if (monoSamples.size() >= FFT_SIZE) {
                    processFFT(monoSamples.data(), (int)monoSamples.size());
                    monoSamples.clear();
                }

                hr = pCaptureClient->GetNextPacketSize(&packetLength);
                if (FAILED(hr)) break;
            }
        }
    }

    pAudioClient->Stop();
    std::cout << "Audio loopback capture stopped" << std::endl;

cleanup:
    if (pwfx) CoTaskMemFree(pwfx);
    if (pCaptureClient) pCaptureClient->Release();
    if (pAudioClient) pAudioClient->Release();
    if (pDevice) pDevice->Release();
    if (pEnumerator) pEnumerator->Release();

    CoUninitialize();
}

#else
// Non-Windows stub implementation
AudioCapture::AudioCapture() {
    for (int i = 0; i < NUM_BANDS; i++) {
        m_bandLevels[i].store(0.0f);
    }
}

AudioCapture::~AudioCapture() {
    stop();
}

bool AudioCapture::start() {
    std::cerr << "Audio capture not supported on this platform" << std::endl;
    return false;
}

void AudioCapture::stop() {
    m_running = false;
}

void AudioCapture::getBandLevels(float* levels, int numBands) const {
    for (int i = 0; i < numBands; i++) {
        levels[i] = 0.0f;
    }
}

float AudioCapture::getVolume() const { return 0.0f; }
float AudioCapture::getDominantFrequency() const { return 100.0f; }
void AudioCapture::processFFT(const float*, int) {}
void AudioCapture::captureThread() {}
#endif
