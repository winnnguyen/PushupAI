import cv2
import mediapipe as mp 
import numpy as np
import pandas as pd
    
def make_row(results, direction):
    dir = np.array([direction])
    arr = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
    return_array = np.append(dir, arr).flatten().reshape(1, -1)
    return return_array
    

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
data = np.empty(133)

cap = cv2.VideoCapture('pushupdata/pushups2.mov')
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    
    while cap.isOpened():
        ret, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('u'):
            curr = make_row(results, 'up')
            data = np.vstack((data, curr))
        
        if key & 0xFF == ord('d'):
            curr = make_row(results, 'down')
            data = np.vstack((data, curr))

        cv2.imshow('Raw Webcam Feed', image)

        if key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

cleaned_data = np.delete(data, 0, 0)
df = pd.DataFrame(cleaned_data)
df.to_csv('landmarks2.csv')

