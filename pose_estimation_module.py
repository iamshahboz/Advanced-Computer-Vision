import cv2 
import mediapipe as mp
import time 



class poseDetector():
    def __init__(self, 
                 mode=False,
                 model_complexity=1,
                 smooth=True, 
                 detectionCon=0.5, 
                 trackCon=0.5):

        self.mode = mode 
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.pose_landmark_style = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        self.pose_connection_style = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, 
                    self.results.pose_landmarks, 
                    self.mpPose.POSE_CONNECTIONS,
                    self.pose_landmark_style,
                    self.pose_connection_style)
                
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x* width), int(lm.y*height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList




    



def main():
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        if not success:
            print('End of video, bye')
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        #lmList = detector.findPosition(img, False)
        print(lmList[14])
        #cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        cv2.imshow('Image', img)


        cv2.waitKey(15)

        

'''
now lets say we want the landmark number 14

'''    


if __name__ == "__main__":
    main()



