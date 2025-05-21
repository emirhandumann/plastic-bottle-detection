import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
from hx711 import HX711

hx = HX711(dout_pin=5, pd_sck_pin=6)
hx.reset()
hx.zero()
print("Tartı sıfırlandı.")

# Kalibrasyon katsayısı (örnek)
referenceUnit = 1.15  # Hesapladığın değeri buraya yaz

try:
    while True:
        raw_val = hx.get_weight(5)
        # Eğer ham değer çok büyük/küçük çıkarsa, referans katsayını değiştir
        val = raw_val / referenceUnit
        print(f"Ağırlık: {val:.2f} gram")
        hx.power_down()
        hx.power_up()
        time.sleep(1)
except KeyboardInterrupt:
    print("Çıkılıyor...")
finally:
    pass
