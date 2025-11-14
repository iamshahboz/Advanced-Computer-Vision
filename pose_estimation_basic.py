import cv2 
import mediapipe as mp
import time 


mpDraw = mp.solutions.drawing_utils
pose_landmark_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
pose_connection_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

mpPose = mp.solutions.pose

pose = mpPose.Pose()


cap = cv2.VideoCapture('PoseVideos/2.mp4')

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 960, 540)   # adjust as needed

pTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mpPose.POSE_CONNECTIONS,
            pose_landmark_style,
            pose_connection_style)


    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow('Image', img)


    cv2.waitKey(15)

