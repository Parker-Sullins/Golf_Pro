import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from calculate_angle import get_landmark, calc_angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# 3. Apply Styling
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

#Used to establish first frame of video do NOT alter.
num = 1
#Iniital value of probability DO NOT alter
pro_prob_counter = 0
am_prob_counter = 0

#Specify the file name to be run
file_name = 'John'
#Specify pro or amatuer video folder
Video_Folder = "Pro_Video"

#Choose model - look in \\models for more models
with open(".\\Models\\golf_model_rf.pkl", 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(".\\Raw_Video\\" + Video_Folder + "\\" + file_name + ".mp4")
# Initiate Pose model
with mp_pose.Pose(min_detection_confidence=0, min_tracking_confidence=0, model_complexity=2) as pose_model:

    #Unused, can be ignored
    frame_width = int(720)
    frame_height = int(1280)

    #Loop through video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()


        try:
        # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            break

        # Make Detections and store in results
        results = pose_model.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections styling
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            frame_num = list(np.array([num]).flatten())
            land_marks = list(np.array([[landmark.x, landmark.y, landmark.z,
                                       landmark.visibility] for landmark in pose]).flatten())
            pose_row = (land_marks + frame_num)

            num += 1
            #Display probability frame by frame
            X = pd.DataFrame([pose_row])
            pose_class = model.predict(X)[0]
            pose_prob = model.predict_proba(X)[0]
            print(pose_prob)
            pro_prob_counter += pose_prob[1]
            am_prob_counter += pose_prob[0]

        except:
            pass
        #Display image
        cv2.imshow('Frame', image)
        #Press Q to break from running model
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


pro_percentage = ((pro_prob_counter*100)/num)
am_percentage = ((am_prob_counter*100)/num)

print("pro:", pro_percentage, am_percentage)
print(num)
print("over")
cap.release()
cv2.destroyAllWindows()
