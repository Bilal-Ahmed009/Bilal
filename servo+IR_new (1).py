import RPi.GPIO as GPIO
from time import sleep


sensor = 23 #IR sensor
servo  = 17
flag   = 5
#led    = 40

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor, GPIO.IN)
GPIO.setup(flag,   GPIO.IN)
GPIO.setup(servo,  GPIO.OUT)

p = GPIO.PWM(servo, 50)

p.start(4)
p.ChangeDutyCycle(2.5)
try:
    while True:
        #print("IR: ", GPIO.input(sensor))
        #print("Flag: ",GPIO.input(flag))
        if (GPIO.input(flag)==True):
            #GPIO.output(led, False)
            if (GPIO.input(sensor)==1):
                print("place hand to unlock Door") 
            if (GPIO.input(sensor)==True):
                p.ChangeDutyCycle(7.5)
                sleep(4)
                print("Door unlocked")
            if (GPIO.input(sensor)==False):
                p.ChangeDutyCycle(2.5)
                print("Door locked")
        else:
            print("Wait...")
            
            sleep(1)
        
finally:
    GPIO.cleanup()     
