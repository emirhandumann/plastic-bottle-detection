from hx711 import HX711
import time

# HX711 pinlerini tanımla
hx = HX711(dout_pin=5, pd_sck_pin=6)

# Kalibrasyon ve sıfırlama
hx.zero()
print("Tartı sıfırlandı. Lütfen bekleyin...")

# 5 kez ağırlık oku ve ekrana yazdır
for i in range(5):
    weight = hx.get_weight(5)
    print(f"Ağırlık: {weight:.2f} gram")
    time.sleep(1)

print("Test tamamlandı.")
