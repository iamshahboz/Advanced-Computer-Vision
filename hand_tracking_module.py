import cv2 
import mediapipe as mp 
import time 


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.hand_landmark_style = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.hand_connection_style = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, 
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.hand_landmark_style,
                        self.hand_connection_style)

        return img

            
# for id, lm in enumerate(handLms.landmark):
#                     print(id, lm) #prin the landmarks
#                     height, width, channel = img.shape
#                     cx, cy = int(lm.x * width), int(lm.y * height)
                    
#                     print(id, cx, cy)

#                     if id == 4:
#                         cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)





def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)

        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ =='__main__':
    main()


