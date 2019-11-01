import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD) #higher level mode

mode = GPIO.getmode() #get the mode

GPIO.setwarnings(False) #turn off mode

GPIO.setup(8, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
    if GPIO.input(8) == GPIO.HIGH: #if the button is pushed
        print("Button was pushed", time.ctime(time.time()))
        time.sleep(5) #sleep for 5 seconds between print
        
