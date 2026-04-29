#include "audio_provider.h"
#include "driver/i2s.h"
#include "esp_log.h"

static const char* TAG = "AUDIO_FRONTEND";

// ESP32-S3-EYE Onboard Microphone Pins (Adjust if using external mic)
#define I2S_PORT            I2S_NUM_0
#define I2S_SCK_PIN         41
#define I2S_WS_PIN          42
#define I2S_SDIN_PIN        2

#define STRIDE_SAMPLES 3200 // 200ms of new data per "window"
static int16_t audio_history[AUDIO_WINDOW_SAMPLES] = {0};

// FreeRTOS Ring Buffer handle
RingbufHandle_t audio_ring_buffer = NULL;

// A background task that constantly pulls audio from I2S and pushes it to our ring buffer
static void audio_capture_task(void *arg) {
    size_t bytes_read = 0;
    const size_t chunk_size = 1024; // Read in small chunks
    int16_t i2s_read_buff[chunk_size / 2];

    while (1) {
        // Read raw data from I2S DMA
        i2s_read(I2S_PORT, (void*)i2s_read_buff, chunk_size, &bytes_read, portMAX_DELAY);
        
        if (bytes_read > 0) {
            // Push to ring buffer. If buffer is full, the oldest data is naturally pushed out
            // to maintain a sliding continuous window of the latest audio.
            xRingbufferSend(audio_ring_buffer, i2s_read_buff, bytes_read, pdMS_TO_TICKS(10));
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

bool init_audio_provider() {
    ESP_LOGI(TAG, "Initializing I2S Driver for continuous streaming...");

    // Create a ring buffer large enough to hold ~1.5 seconds of audio
    audio_ring_buffer = xRingbufferCreate(AUDIO_WINDOW_BYTES * 2, RINGBUF_TYPE_BYTEBUF);
    if (audio_ring_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to create ring buffer");
        return false;
    }

    // Configure I2S for 16kHz, 16-bit Mono
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SDIN_PIN
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install I2S driver");
        return false;
    }

    if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set I2S pins");
        return false;
    }

    // Start the background capture task pinned to Core 0 (leaving Core 1 for Neural Net inference)
    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture", 8192, NULL, 5, NULL, 0);
    
    ESP_LOGI(TAG, "Audio Frontend initialized successfully.");
    return true;
}

// Define our sliding window stride (e.g., 3200 samples = 200ms stride)
// This means the model will run 5 times per second.

bool get_latest_audio_window(int16_t* audio_buffer) {
    size_t item_size;
    
    // 1. Try to get just the "new" chunk of audio (e.g., 200ms)
    void* data = xRingbufferReceiveUpTo(audio_ring_buffer, &item_size, pdMS_TO_TICKS(10), STRIDE_SAMPLES * sizeof(int16_t));
    
    if (data != NULL) {
        if (item_size > 0) {
            int samples_received = item_size / sizeof(int16_t);
            
            // 2. Shift old data left
            memmove(audio_history, 
                    &audio_history[samples_received], 
                    (AUDIO_WINDOW_SAMPLES - samples_received) * sizeof(int16_t));
            
            // 3. Append new data at the end
            memcpy(&audio_history[AUDIO_WINDOW_SAMPLES - samples_received], 
                   data, 
                   item_size);
            
            vRingbufferReturnItem(audio_ring_buffer, data);
            
            // 4. Copy the whole 1-second history to the main loop's buffer
            memcpy(audio_buffer, audio_history, AUDIO_WINDOW_BYTES);
            return true;
        }
        vRingbufferReturnItem(audio_ring_buffer, data);
    }
    return false;
}

/*
bool get_latest_audio_window(int16_t* audio_buffer) {
    size_t item_size;
    // Attempt to pull exactly 1 second (16000 samples) of audio from the ring buffer
    void* data = xRingbufferReceiveUpTo(audio_ring_buffer, &item_size, pdMS_TO_TICKS(10), AUDIO_WINDOW_BYTES);
    
    if (data != NULL) {
        if (item_size == AUDIO_WINDOW_BYTES) {
            // We got a full 1-second window
            memcpy(audio_buffer, data, AUDIO_WINDOW_BYTES);
            vRingbufferReturnItem(audio_ring_buffer, data);
            return true;
        }
        // If we didn't get enough data, just return it so it can build up
        vRingbufferReturnItem(audio_ring_buffer, data);
    }
    return false;
} */