import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
hand_landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hand_connection_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

pTime = 0
cTime = 0



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm) #prin the landmarks
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                
                print(id, cx, cy)

                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)


            mpDraw.draw_landmarks(
                img, 
                handLms, 
                mpHands.HAND_CONNECTIONS,
                hand_landmark_style,
                hand_connection_style)
            
    cTime = time.time()
    fps = 1/(cTime-pTime)

    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

