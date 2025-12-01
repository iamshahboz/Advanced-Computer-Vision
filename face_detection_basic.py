import cv2 
import mediapipe as mp 
import time 

input_path = 'FaceVideos\podcast.mp4'



cap = cv2.VideoCapture(input_path)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.4)



while True:
    success, img = cap.read()
    if not success:
        print('The video finished, bye')
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for detection in results.detections:

            # Get bounding box (relative values)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape

            # Convert to absolute pixel values
            bbox = (
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h)
            )

            # Draw GREEN bounding box
            cv2.rectangle(img, bbox, (0, 255, 0), 2)

            # Optionally draw detection score
            score = int(detection.score[0] * 100)
            cv2.putText(img, f'{score}%', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
