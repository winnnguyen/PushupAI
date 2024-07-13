import cv2
import mediapipe as mp 
import numpy as np 
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='PushupAI',
    page_icon='💪'
)
count = 0
with st.sidebar:
    selected = option_menu(
        menu_title= 'Main Menu',
        options= ['Home', 'Tracker', 'Leaderboard'],
        menu_icon='cast'
    )
if selected == 'Home':
    st.title('Home')
    st.write('Welcome to PushupAI!')
    st.write('Pushups are a fundamental bodyweight exercise that primarily targets the muscles of the chest, shoulders, and triceps, while also engaging the core and lower body to a lesser extent. PushupAI uses machine learning and computer vision to process the amount of pushups you do in one sitting. Track your high scores and weekly goals!')
if selected == 'Leaderboard':
    st.title('Leaderboard')
    st.write('Track your highest count!')
if selected == 'Tracker':
    st.title('Pushup Tracker')
    st.write('Before clicking the Start button, maintain Push Up position.')
    start_button = st.button('Start')
    frame_placeholder = st.empty()
    down = False
    if start_button:
        start_button = st.empty()
        stop_button = st.button('Stop')

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with open('model_pickle', 'rb') as f:
            model = pickle.load(f)

        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
            
            while cap.isOpened() and not stop_button:
                ret, image = cap.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if (results.pose_landmarks):
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().reshape(1, -1)
                    body_language_class = model.predict(row)[0]
                    body_language_prob = model.predict_proba(row)[0]

                    if body_language_class.split(' ')[0] == 'up' and down:
                        count += 1
                        down = False

                    if body_language_class.split(' ')[0] == 'down':
                        down = True

                    cv2.rectangle(image, (0,0), (600, 150), (245, 117, 16), -1)
                    cv2.putText(image, 'CLASS', (200,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0], (200,102), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'PROB', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (20,102), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'COUNT', (400,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, str(count), (435,102), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                frame_placeholder.image(image, channels='BGR')

                if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
                    break

        cap.release()
        cv2.destroyAllWindows()

