import mediapipe as mp
import cv2
import numpy as np

import csv


def create_csv(file_name: str, num_coords=33):

    landmarks = ['class']
    for val in range(1, num_coords + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    landmarks += ["Frame"]

    with open("C:\\Users\\prsullins\\OneDrive - Creative Manufacturing, LLC\\ALM.SWE\\DGMD-14\\" + file_name + ".csv",
              mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)


create_csv(file_name='golf')

