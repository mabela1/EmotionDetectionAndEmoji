import os
import cv2
import numpy as np
#from keras.preprocessing import image
import keras.utils as image
import warnings
warnings.filterwarnings("ignore")
#from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array

from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cvlearn import FaceMesh
from cvlearn.Utils import *

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

def draw_arc(test_img, x, y, rotation, radius):
    center = (x, y)
    axes = (radius, radius)
    angle = rotation
    startAngle = 100
    endAngle = 0

    cv2.ellipse(test_img, center, axes, angle, startAngle, endAngle, (255, 255, 255), int(radius/2))

detector = FaceMesh.FaceMeshDetector()

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    test_img, faces = detector.findFaceMesh(test_img, draw=False)


    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    if faces:
        for j in range(len(faces)):
            for i in range(len(faces[j])):
                faceLength, face_pos = findDistance(faces[j][face['faceUp']], faces[j][face['faceDown']], test_img)

                cv2.ellipse(test_img, (face_pos[0], face_pos[1]-30), (int(faceLength/1.5), int(faceLength/1.5)+10),
                            find_rotation(faces[j][face['faceUp']], faces[j][face['faceDown']]), 0, 360, (52, 255, 255), -1)
                #draw eyes
                Right_eye_length, R_eye_pos = findDistance(faces[j][rightEye['eyeUp']], faces[j][rightEye['eyeDown']], test_img)
                cv2.circle(test_img, (R_eye_pos[0], R_eye_pos[1]-30), int(Right_eye_length/1.3), (65, 71, 100), -1)
                draw_arc(test_img, R_eye_pos[0]-3, R_eye_pos[1]-33, find_rotation(faces[j][rightEye['eyeRight']], faces[j][rightEye['eyeLeft']]),
                         int(Right_eye_length/3))

                Left_eye_length, L_eye_pos = findDistance(faces[j][leftEye['eyeUp']], faces[j][leftEye['eyeDown']], test_img)
                cv2.circle(test_img, (L_eye_pos[0], L_eye_pos[1] - 30), int(Left_eye_length / 1.3), (65, 71, 100), -1)
                draw_arc(test_img, L_eye_pos[0] - 3, L_eye_pos[1] - 33,
                         find_rotation(faces[j][leftEye['eyeRight']], faces[j][leftEye['eyeLeft']]),
                         int(Left_eye_length / 3))

                #draw mouth
                MouthLength1, Mouth_pos = findDistance(faces[j][mouth['mouthRight']], faces[j][mouth['mouthLeft']], test_img)
                MouthLength2, _ = findDistance(faces[j][mouth['mouthUp']], faces[j][mouth['mouthDown']], test_img)
                cv2.ellipse(test_img, (Mouth_pos[0], Mouth_pos[1]-30), (int(MouthLength1 / 1.6), int(MouthLength2)),
                            find_rotation(faces[j][mouth['mouthRight']], faces[j][mouth['mouthLeft']]), 360, 180, (0, 60, 255), -1)

                #draw teeth
                if MouthLength2 >= 30:
                    cv2.ellipse(test_img, (Mouth_pos[0], Mouth_pos[1] - 30), (int(MouthLength1 / 1.6)-10, 20),
                                find_rotation(faces[j][mouth['mouthRight']], faces[j][mouth['mouthLeft']]), 360, 180,
                                (255, 255, 255), -1)


                #draw eyebrow
                right_eyebrow = [
                    [faces[j][336][0], faces[j][336][1]-40],
                    [faces[j][296][0], faces[j][296][1]-40],
                    [faces[j][334][0], faces[j][334][1]-40],
                    [faces[j][293][0], faces[j][293][1]-40],
                    [faces[j][300][0], faces[j][300][1]-40],
                    [faces[j][283][0], faces[j][283][1]-40],
                    [faces[j][282][0], faces[j][282][1]-40],
                    [faces[j][295][0], faces[j][295][1]-40],
                    [faces[j][285][0], faces[j][285][1]-40]
                ]

                right_eyebrow_polygen = np.array([right_eyebrow], np.int32)
                cv2.fillPoly(test_img, pts=[right_eyebrow_polygen], color=(0, 0, 0))

                left_eyebrow = [
                    [faces[j][70][0], faces[j][70][1]-40],
                    [faces[j][63][0], faces[j][63][1]-40],
                    [faces[j][105][0], faces[j][105][1]-40],
                    [faces[j][66][0], faces[j][66][1]-40],
                    [faces[j][107][0], faces[j][107][1]-40],
                    [faces[j][55][0], faces[j][55][1]-40],
                    [faces[j][65][0], faces[j][65][1]-40],
                    [faces[j][52][0], faces[j][52][1]-40],
                    [faces[j][53][0], faces[j][53][1]-40]
                ]

                left_eyebrow_polygen = np.array([left_eyebrow], np.int32)
                cv2.fillPoly(test_img, pts=[left_eyebrow_polygen], color=(0, 0, 0))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows