#include <Wire.h>
#include <Adafruit_VL53L0X.h>
#include <driver/i2s.h>

Adafruit_VL53L0X sensor = Adafruit_VL53L0X();
const i2s_port_t I2S_PORT = I2S_NUM_0;

void setup() {
  Serial.begin(921600);
  Wire.begin(8, 9); 
  
  
  if (!sensor.begin()) {
    Serial.println("Error VL53L0X");
    while (1);
  }

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .dma_buf_count = 4,
    .dma_buf_len = 1024
  };
  
  i2s_pin_config_t pin_config = { 
    .bck_io_num = 37, 
    .ws_io_num = 36, 
    .data_out_num = -1, 
    .data_in_num = 35 
  };
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;
  sensor.rangingTest(&measure, false);
  
  if (measure.RangeStatus != 4 && measure.RangeMilliMeter < 500) {
    // Enviamos Distancia y la marca de tiempo actual del sistema
    Serial.printf("OBJ:%d\nTIME_MS:%lu\nSTART_AUDIO\n", measure.RangeMilliMeter, millis());
    
    uint32_t start = millis();
    int32_t buf[512];
    size_t bytes_read;

    while (millis() - start < 4000) {
      esp_err_t err = i2s_read(I2S_PORT, (void*)buf, sizeof(buf), &bytes_read, portMAX_DELAY);
      if (err == ESP_OK && bytes_read > 0) {
        int samples_count = bytes_read / 4;
        int16_t out_buf[samples_count]; 
        for (int i = 0; i < samples_count; i++) {
          out_buf[i] = (int16_t)(buf[i] >> 12); 
        }
        Serial.write((uint8_t*)out_buf, samples_count * sizeof(int16_t));
      }
    }
    Serial.println("\nEND_AUDIO"); 
  }
  delay(100);
}

