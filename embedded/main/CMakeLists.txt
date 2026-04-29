idf_component_register(SRCS "dsp_pipeline.cpp" "audio_provider.cpp" "main.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES espressif__esp-tflite-micro driver esp_timer)