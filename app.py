#Based on Adrian Rosebrock's Tutorial: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
import numpy as np
import pickle
import time
import cv2
import os

avengers = ["Captain_America","Iron_Man","Thor","Black_Widow","Hulk","Hawkeye","Other"]
colors = [(255,0,0),(0,0,255),(0,255,255),(0,0,0),(0,255,0),(255,0,255),(255,255,0)]

protoPath = "models/face_detector.prototxt"
modelPath = "models/face_detector.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("models/embedding.t7")
recognizer = pickle.loads(open("models/avengers.pickle", "rb").read())
le = pickle.loads(open("models/le.pickle", "rb").read())

#Specify video clip to process
video = cv2.VideoCapture("videos/Avengers_Clip_1.mp4")
time.sleep(1.0)

#Define output video size
outputVideoWidth = 1920
outputVideoHeight = 1080

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#Save processed video										
out = cv2.VideoWriter("output/Avengers_Detected.mp4", fourcc, 30, (outputVideoWidth,outputVideoHeight))

ret,frame = video.read()
while ret:
	frame = cv2.resize(frame,(outputVideoWidth,outputVideoHeight))
	(h, w) = frame.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.70:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			avengerIndex = avengers.index(name)
			name = name.replace("_"," ")
			color = colors[avengerIndex]

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),color, 4)
			if(name == "Black Widow"):
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 1.00, (155,155,155), 5)	
			else:
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 1.00, (0,0,0), 5)
			cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 1.00, color, 3)


	cv2.imshow("Recognition", frame)
	out.write(frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	ret,frame = video.read()
	
cv2.destroyAllWindows()
video.release()
out.release()
