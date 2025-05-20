from HX711 import SimpleHX711, Mass
import time

# Pinler: DT=5, SCK=6
# İlk başta referans birim ve offset bilinmiyor, kalibrasyon yapacağız
REFERENCE_UNIT = -370  # Örnek değer, kendi kalibrasyonuna göre değiştir
OFFSET = 0  # Kalibrasyon sonrası güncellenecek

hx = SimpleHX711(5, 6, REFERENCE_UNIT, OFFSET)
hx.setUnit(Mass.Unit.G)

print("Tartı sıfırlanıyor...")
hx.zero()  # Sıfırlama (tare)
print("Tartı sıfırlandı. Şimdi ölçüm başlıyor...")

try:
    for i in range(10):
        weight = hx.weight(10)  # 10 örnekle ortalama gram cinsinden ağırlık
        print(f"Ağırlık: {float(weight):.2f} gram")
        time.sleep(1)
finally:
    hx.cleanup()
    print("Test tamamlandı.")
