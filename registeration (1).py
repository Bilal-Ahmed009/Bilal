import time
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import mysql.connector


db = mysql.connector.connect(
  host="localhost",
  user="attendanceadmin",
  passwd="pimylifeup",
  database="attendancesystem"
)

cursor = db.cursor()
reader = SimpleMFRC522()
flag = "1"
try:
  while flag != "0":
    print('Place Card to\nregister')
    id, text = reader.read()
    cursor.execute("SELECT id FROM users WHERE rfid_uid="+str(id))
    cursor.fetchone()

    if cursor.rowcount >= 1:
      print("Overwrite\nexisting user?")
      overwrite = input("Overwite (Y/N)? ")
      flag = input('press 0 to end registeration '+' press 1 to register more :\n')
      print('registeration end')
      
      if overwrite[0] == 'Y' or overwrite[0] == 'y':
        print("Overwriting user.")
        time.sleep(1)
        sql_insert = "UPDATE users SET name = %s, RollNumber = %s WHERE rfid_uid=%s"
      else:
        continue;
    else:
      sql_insert = "INSERT INTO users (name, RollNumber, rfid_uid) VALUES (%s, %s, %s)"

    print('Enter new name')
    new_name = input("Name: ")

    print('Enter New Roll Number')
    new_Roll_Number = input("Roll Number: ")

    cursor.execute(sql_insert, (new_name, new_Roll_Number, id))

    db.commit()

    print("User " + new_name + "\nSaved")
    print("User " + new_Roll_Number + "\nSaved")
    time.sleep(2)
    flag = input('press 0 to end registeration '+' press 1 to register more :\n')
    print('registeration end')
finally:
  GPIO.cleanup()


