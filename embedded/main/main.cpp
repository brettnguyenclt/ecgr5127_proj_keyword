#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

// Custom Audio and DSP headers
#include "audio_provider.h"
#include "dsp_pipeline.h"

// TFLite Micro Headers
#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "KWS_MAIN";

// Class names matching the alphabetical order of tf.keras.utils.audio_dataset_from_directory
const char* CLASS_NAMES[] = {"_silence", "_unknown", "positive", "yes"};
const int NUM_CLASSES = 4;

// Tensor Arena (Memory pool for TFLM). 100KB is usually plenty for a lightweight CNN.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Command Smoother State
int previous_prediction = -1;
int consecutive_hits = 0;

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting Keyword Spotting System...");

    // 1. Initialize TFLite Micro
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        return;
    }

    // The '6' is the number of operations we are registering
    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddResizeBilinear();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu(); // Most KWS models use ReLU or ReLU6
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddPad();
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    ESP_LOGI(TAG, "Model Loaded. Input shape: [%d, %d, %d, %d]", 
             input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);

    // 2. Initialize Audio & DSP
    if (!init_audio_provider()) {
        ESP_LOGE(TAG, "Audio provider failed to initialize.");
        return;
    }
    init_dsp_pipeline();

    static int16_t audio_buffer[AUDIO_WINDOW_SAMPLES];

    // 3. Continuous Inference Loop
    while (true) {
        // Fetch the sliding 1-second window
        if (get_latest_audio_window(audio_buffer)) {
            // ESP_LOGI(TAG, "Audio window captured!"); // Temporarily uncomment to verify

            // Generate INT8 Spectrogram directly into the TFLM input tensor
            generate_spectrogram_int8(audio_buffer, 
                                      input->data.int8, 
                                      input->params.scale, 
                                      input->params.zero_point);

            // Time the inference for the Report Metric
            int64_t start_time = esp_timer_get_time();
            /*
            printf("Spectrogram Sample:\n");
            for (int f = 0; f < 10; f++) {
                for (int b = 0; b < 10; b++) {
                    printf("%4d ", input->data.int8[f * FFT_BINS + b]);
                }
                printf("\n");
            } */

            if (interpreter.Invoke() != kTfLiteOk) {
                ESP_LOGE(TAG, "Invoke failed!");
                continue;
            }
            
            int64_t end_time = esp_timer_get_time();
            int execution_time_ms = (end_time - start_time) / 1000;

            // Process Softmax Output
            int current_prediction = -1;
            float max_confidence = 0.0f;

            for (int i = 0; i < NUM_CLASSES; i++) {
                // De-quantize back to float for easier thresholding
                float confidence = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
                if (confidence > max_confidence) {
                    max_confidence = confidence;
                    current_prediction = i;
                }
            }

            // --- COMMAND SMOOTHER LOGIC ---
            // Only trigger if confidence is > 80% AND it matches the last prediction
            if (max_confidence > 0.80f) {
                if (current_prediction == previous_prediction) {
                    consecutive_hits++;
                } else {
                    consecutive_hits = 1;
                    previous_prediction = current_prediction;
                }

                if (consecutive_hits == 2) {
                    // We have a solid hit!
                    ESP_LOGI(TAG, "\n==============================");
                    ESP_LOGI(TAG, "  DETECTED: %s (%.1f%%)", CLASS_NAMES[current_prediction], max_confidence * 100);
                    ESP_LOGI(TAG, "  [REPORT METRIC] Inference Time: %d ms (~%d FPS)", execution_time_ms, 1000 / execution_time_ms);
                    ESP_LOGI(TAG, "==============================\n");
                    
                    // Reset to avoid spamming the same command
                    consecutive_hits = 0; 
                    previous_prediction = -1;
                    
                    // Optional: Sleep a bit longer after a valid detection to clear the buffer
                    vTaskDelay(pdMS_TO_TICKS(500)); 
                }
            } else {
                // Confidence too low, reset smoother
                consecutive_hits = 0;
                previous_prediction = -1;
            }

        } else {
            // Wait briefly for the audio buffer to fill
            vTaskDelay(pdMS_TO_TICKS(20));
            static int fail_count = 0;
            if(fail_count++ % 100 == 0){
                ESP_LOGW(TAG, "Waiting for audio data");
            }
        }
    }
}