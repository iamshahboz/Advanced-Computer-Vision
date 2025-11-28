import cv2 
import mediapipe as mp 
import time 

input_path = 'FaceVideos/podcast_hd.mp4'

cap = cv2.VideoCapture(input_path)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            print(detection.score)





    if not success:
        print('The vide0 finished, bye')
        break

    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    cv2.imshow('Image', img)


    cv2.waitKey(1)
