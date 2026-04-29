#ifndef DSP_PIPELINE_H
#define DSP_PIPELINE_H

#include <stdint.h>

// Spectrogram parameters matching Python training
#define SAMPLE_RATE 16000
#define FRAME_LENGTH 480   // 30ms window
#define FRAME_STEP 320     // 20ms stride
#define NUM_FRAMES 49      // (16000 - 480) / 320 + 1
#define FFT_BINS 241       // (480 / 2) + 1

// Initializes pre-computed windowing arrays to save CPU cycles
void init_dsp_pipeline();

// Converts 1 second of raw audio into the exact INT8 2D tensor the model expects
void generate_spectrogram_int8(const int16_t* audio_buffer, int8_t* output_tensor, float model_scale, int model_zero_point);

#endif // DSP_PIPELINE_H