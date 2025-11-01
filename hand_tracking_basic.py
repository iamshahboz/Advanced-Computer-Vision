import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
hand_landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hand_connection_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img, 
                handLms, 
                mpHands.HAND_CONNECTIONS,
                hand_landmark_style,
                hand_connection_style)



    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

