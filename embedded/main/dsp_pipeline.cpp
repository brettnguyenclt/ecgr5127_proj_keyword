#include "dsp_pipeline.h"
#include "dsps_fft2r.h" // Espressif DSP Library
#include <math.h>
#include <string.h>

static float hann_window[FRAME_LENGTH];
static float fft_input[512 * 2]; // Complex buffer: [real0, imag0, real1, imag1...]

void init_dsp_pipeline() {
    // 1. Initialize the FFT lookup tables
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, 512);
    if (ret != ESP_OK) {
        printf("DSP: Failed to initialize FFT\n");
    }

    // 2. Pre-compute Hann window
    for (int i = 0; i < FRAME_LENGTH; i++) {
        hann_window[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / FRAME_LENGTH);
    }
}

void generate_spectrogram_int8(const int16_t* audio_buffer, int8_t* output_tensor, float model_scale, int model_zero_point) {
    int tensor_index = 0;

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int start_idx = frame * FRAME_STEP;

        // 1. Prepare FFT Input (Windowing + Padding + Complex conversion)
        for (int i = 0; i < 512; i++) {
            if (i < FRAME_LENGTH) {
                float sample = (float)audio_buffer[start_idx + i] / 32768.0f;
                fft_input[i * 2] = sample * hann_window[i]; // Real part
            } else {
                fft_input[i * 2] = 0; // Zero padding to reach 512
            }
            fft_input[i * 2 + 1] = 0; // Imaginary part is 0
        }

        // 2. Execute Hardware-Accelerated FFT
        dsps_fft2r_fc32(fft_input, 512);
        // Bit-reversal (required for radix-2)
        dsps_bit_rev_fc32(fft_input, 512);

        // 3. Compute Magnitudes for the first 241 bins
        for (int k = 0; k < FFT_BINS; k++) {
            float real = fft_input[k * 2];
            float imag = fft_input[k * 2 + 1];
            float magnitude = sqrtf(real * real + imag * imag);

            // 4. Quantize to INT8 for the TFLite Model
            int32_t quantized = (int32_t)(magnitude / model_scale) + model_zero_point;
            
            // Clip to INT8 range
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            
            output_tensor[tensor_index++] = (int8_t)quantized;
        }
    }
}