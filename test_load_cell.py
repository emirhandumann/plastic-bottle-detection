import RPi.GPIO as GPIO
from hx711 import HX711

GPIO.setmode(GPIO.BCM)

hx = HX711(dout_pin=6, pd_sck_pin=5)
hx.zero()

while True:
    reading = hx.get_data_mean()
    print(reading)
