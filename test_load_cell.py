import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
from hx711 import HX711

hx = HX711(dout_pin=5, pd_sck_pin=6)
hx.reset()
hx.tare()
print("Tartı sıfırlandı.")

referenceUnit = 1.15  # Hesapladığın değeri buraya yaz
hx.set_reference_unit(referenceUnit)

try:
    while True:
        val = hx.get_weight(5)
        print(f"Ağırlık: {val:.2f} gram")
        hx.power_down()
        hx.power_up()
        time.sleep(1)
except KeyboardInterrupt:
    print("Çıkılıyor...")
finally:
    pass
4
