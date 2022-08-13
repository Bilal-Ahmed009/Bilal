from matplotlib import pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
#
import time
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614
from mfrc522 import SimpleMFRC522
from time import sleep
import mysql.connector
from picamera import PiCamera
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# led1 = 1
# led2 = 2
# led3 = 3
# led4 = 4
# GPIO.setup(led1, GPIO.OUT)
# GPIO.setup(led2, GPIO.OUT)
# GPIO.setup(led3, GPIO.OUT)
# GPIO.setup(led4, GPIO.OUT)

db = mysql.connector.connect(
   host="localhost",
   user="attendanceadmin",
   passwd="pimylifeup",
   database="attendancesystem"
)

cursor = db.cursor()

GPIO.setup(17, GPIO.OUT)
p = GPIO.PWM(17, 50)
p.start(2.5)
img_counter=0

try:
  while True:
      reader = SimpleMFRC522()
      print('Place Card to Capture image')
      sleep(2)
      print('Stand Still to get Clear Image')

      id, text = reader.read()
      print(id)
      if id == 659875232099 or id == 1045376603853 or id == 583140890749 or id == 41249012171:
        # define a video capture object
            vid = cv2.VideoCapture(0)   
        # Capture the video frame
        # by frame
            ret, frame = vid.read()
        #cv2.imshow('frame', frame)
            cv2.imwrite("/home/pi/Documents/test1.jpg", frame)
#             GPIO.output(led1, GPIO.HIGH)
#             time.sleep(2)
#             GPIO.output(led1, GPIO.LOW)
            print('image captured wait for Mask Detection')
            cv2.destroyAllWindows()
            vid.release()

      def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
          (h, w) = frame.shape[:2]
          blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
              (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
          faceNet.setInput(blob)
          detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
          faces = []
          locs = []
          preds = []

        # loop over the detections
          for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
              confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
              if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                  box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                  (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                  (startX, startY) = (max(0, startX), max(0, startY))
                  (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                  face = frame[startY:endY, startX:endX]
                  face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                  face = cv2.resize(face, (224, 224))
                  face = img_to_array(face)
                  face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                  faces.append(face)
                  locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
          if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
              faces = np.array(faces, dtype="float32")
              preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
          return (locs, preds)
    #------------------------------------------------------------------
    # load our serialized face detector model from disk
      prototxtPath = r"/home/pi/Documents/deploy.prototxt"
      weightsPath = r"/home/pi/Documents/res10_300x300_ssd_iter_140000.caffemodel"
      faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load our serialized face detector model from disk
      print("[INFO] loading face detector model...")
    #-----------------------------------------------------------------

    # load the face mask detector model from disk
      print("[INFO] loading face mask detector model...")
      maskNet = load_model("/home/pi/Documents/mask_detector.model")

    # initialize the video stream and allow the camera sensor to warm up
    #print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    # vs = VideoStream(src=0).start() ----------------------------------------------- Commented 29/5/2022 ------------------------
      time.sleep(2.0)

    # loop over the frames from the video stream

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
      frame = cv2.imread('/home/pi/Documents/test1.jpg')
      frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
      (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
      for (box, pred) in zip(locs, preds):
          
    # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

    # determine the class label and color we'll use to draw
    # the bounding box and text
        if mask > withoutMask:
#           GPIO.output(led1, GPIO.HIGH)
#           time.sleep(2)
#           GPIO.output(led1, GPIO.LOW)
          label = "Thank You. Mask On."
          color = (0, 255, 0)
        else:
          label = "No Face Mask Detected"
#           GPIO.output(led4, GPIO.HIGH)
#           time.sleep(2)
#           GPIO.output(led4, GPIO.LOW)
          color = (0, 0, 255)
        
    
        cv2.putText(frame, label, (startX-50, startY - 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    # show the output frame
      imgplot = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      plt.show()
      
      #if label == "No Face Mask Detected":
          #print("No Face Mask Detected!")
          #continue
      print('Place your hand for Temperature')
      sleep(2)
      bus = SMBus(1)
      sensor = MLX90614(bus, address=0x5A)
      temp = sensor.get_obj_temp()
      bus.close()
      print("Temperature: ", temp)
      if temp > 38:
          temp_bol = False
#           GPIO.output(led4, GPIO.HIGH)
#           time.sleep(2)
#           GPIO.output(led4, GPIO.LOW)
      else:
          temp_bol = True
#           GPIO.output(led2, GPIO.HIGH)
#           time.sleep(2)
#           GPIO.output(led2, GPIO.LOW)
      print('Place Card to Record Attendance')
      sleep(2)
      id, text = reader.read()
#       GPIO.output(led3, GPIO.HIGH)
#       time.sleep(2)
#       GPIO.output(led3, GPIO.LOW)    
#       cursor.execute("Select id, name, RollNumber FROM users WHERE rfid_uid="+str(id))
#       result = cursor.fetchone()
      print('Look towards camera')
      print('wait, just Finishing...')
      #if mask_bol and temp_bol:
          #print('Conditions Satisfied')
          #p.ChangeDutyCycle(7.5)
          #sleep(5)
          #p.ChangeDutyCycle(2.5)
          #sleep(1)
      #result == [1, 'Visitor/Guest', 1]
      if id == 1045376603853:
          #vid = cv2.VideoCapture(0)   
        # Capture the video frame by frame
          #ret, frame = vid.read()
        #cv2.imshow('frame', frame)
          #cv2.imwrite("/home/pi/attendancesystem/Visitors/visitor.jpg", frame)
          #time.sleep(2)
          img_name = "//home/pi/Documents" +"/visitor{}.jpg".format(img_counter)
          camera = PiCamera()
          time.sleep(2)
          camera.capture(img_name)
          camera.close()
          print("{} guest entry".format(img_name))
          img_counter += 1
          time.sleep(2)

      if cursor.rowcount >= 1:
          print("attendance recorded " + result[1])
          cursor.execute("INSERT INTO attendance (user_id, name, RollNumber, temp) VALUES (%s, %s, %s, %s)", (result[0], result[1], result[2], temp,) )
          db.commit()
      else:
          print("User does not exist.")
#           GPIO.output(led4, GPIO.HIGH)
#          time.sleep(2)
#          GPIO.output(led4, GPIO.LOW)
#          time.sleep(2)
      if temp_bol = True:
          print('Conditions Satisfied')
          p.ChangeDutyCycle(7.5)
          sleep(5)
          p.ChangeDutyCycle(2.5)
          sleep(1)
          GPIO.cleanup()
finally:
  GPIO.cleanup()
