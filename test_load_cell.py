import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
from hx711 import HX711

hx = HX711(dout=5, pd_sck=6)
hx.reset()
hx.zero()
print("Tartı sıfırlandı.")

referenceUnit = 1.15  # Kendi kalibrasyon katsayını yaz
hx.set_reference_unit(referenceUnit)


def get_stable_weight(hx, sample_count=10):
    vals = [hx.get_weight(5) for _ in range(sample_count)]
    return sum(vals) / len(vals)


try:
    while True:
        val = get_stable_weight(hx, 10)
        print(f"Stabil Ağırlık: {val:.2f} gram")
        hx.power_down()
        hx.power_up()
        time.sleep(1)
except KeyboardInterrupt:
    print("Çıkılıyor...")
finally:
    pass
