import RPi.GPIO as GPIO
import time

# GPIO pin numarası (servo motorun bağlı olduğu pin)
SERVO_PIN = 14

# GPIO modunu ayarla
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM nesnesi oluştur (50Hz frekans)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)


def set_angle(angle):
    # Açıyı duty cycle'a çevir (0-180 derece arası)
    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)


try:
    print("Servo motor testi başlıyor...")

    # 0 derece pozisyonuna git
    print("0 derece pozisyonuna gidiliyor...")
    set_angle(0)
    time.sleep(1)

    # 90 derece pozisyonuna git
    print("90 derece pozisyonuna gidiliyor...")
    set_angle(90)
    time.sleep(1)

    # 180 derece pozisyonuna git
    print("180 derece pozisyonuna gidiliyor...")
    set_angle(180)
    time.sleep(1)

    # Başlangıç pozisyonuna dön
    print("Başlangıç pozisyonuna dönülüyor...")
    set_angle(0)

except KeyboardInterrupt:
    print("\nProgram kullanıcı tarafından durduruldu")
finally:
    pwm.stop()
    GPIO.cleanup()
    print("Program sonlandırıldı")
