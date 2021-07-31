import mediapipe as mp
import numpy as np
'''
joint1 = [pose[mp_pose.PoseLandmark.a.value].x, pose[mp_pose.PoseLandmark.a.value].y]
joint2 = [pose[mp_pose.PoseLandmark.b.value].x, pose[mp_pose.PoseLandmark.b.value].y]
joint3 = [pose[mp_pose.PoseLandmark.c.value].x, pose[mp_pose.PoseLandmark.c.value].y]
'''

def get_landmark(a, b, c, pose, mp_pose):
    left_shoulder = [pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [pose[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, pose[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [pose[mp_pose.PoseLandmark.LEFT_WRIST.value].x, pose[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    right_shoulder = [pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [pose[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, pose[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    return a, b, c


def calc_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle