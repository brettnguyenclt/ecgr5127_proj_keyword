#ifndef AUDIO_PROVIDER_H
#define AUDIO_PROVIDER_H

#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"

// --- Audio Configuration ---
// These must perfectly match the Python training scripts
#define SAMPLE_RATE         16000
#define SAMPLE_BITS         16
#define CHANNELS            1

// 1 second of audio = 16,000 samples. 
// Since each sample is 16 bits (2 bytes), the total size is 32,000 bytes.
#define AUDIO_WINDOW_SAMPLES 16000 
#define AUDIO_WINDOW_BYTES   (AUDIO_WINDOW_SAMPLES * 2)

// --- Function Prototypes ---
// Initializes the I2S microphone and starts the background sampling task
bool init_audio_provider();

// Fills the provided array with the *most recent* 1 second of audio from the stream
// Returns true if enough data was available
bool get_latest_audio_window(int16_t* audio_buffer);

#endif // AUDIO_PROVIDER_H