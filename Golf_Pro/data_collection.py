import mediapipe as mp
import cv2
import numpy as np
import csv

#Used to illustrate Landmark detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Specify Current Videos target variable before Data Collection
class_name = "Pro"
#Flag to stop accidentle data collection
flag = False
#Initilize number of frame, DO NOT alter
num = 1

cap = cv2.VideoCapture("C:\\Users\\prsullins\\OneDrive - Creative Manufacturing, LLC\\ALM.SWE\\DGMD-14\\tiger-2.mp4")


if flag:
    # Initiate Pose model
    with mp_pose.Pose(min_detection_confidence=0, min_tracking_confidence=0, model_complexity=2) as pose_model:


        frame_width = int(720)
        frame_height = int(1280)

        while cap.isOpened():
            ret, frame = cap.read()


            try:
            # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            except:

                break
            # Make Detections
            results = pose_model.process(image)


            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            # Recolor image back to BGR for rendering

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                   # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )

            try:
                # Extract Pose landmarks Data

                pose = results.pose_landmarks.landmark
                current_class = list(np.array([class_name]).flatten())
                frame_num = list(np.array([num]).flatten())
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                           landmark.visibility] for landmark in pose]).flatten())
                pose_row = (current_class + pose_row + frame_num)
                print(pose_row)

                num += 1
                print(num)

                with open("C:\\Users\\prsullins\\OneDrive - Creative Manufacturing, LLC\\ALM.SWE\\DGMD-14\\golf.csv",
                          mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(pose_row)

            except:
                pass


            cv2.imshow('Frame', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
else:
    print("Flag Is Set To False")


print("over")
cap.release()
cv2.destroyAllWindows()



