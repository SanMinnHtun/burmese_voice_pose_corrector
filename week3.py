import cv2
import mediapipe as mp
import numpy as np
import math
import time
import edge_tts
import asyncio
import pygame
import os

# =========================
# TEXT TO SPEECH SETUP
# =========================
pygame.mixer.init()
async def speak_async(text):

    filename = "voice.mp3"

    communicate = edge_tts.Communicate(
        text=text,
        voice="my-MM-NilarNeural"
    )

    await communicate.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # wait until audio finishes
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

    # stop and unload music
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    # small delay so Windows releases file
    await asyncio.sleep(0.2)

    # now remove file safely
    if os.path.exists(filename):
        os.remove(filename)


def speak_burmese(text):
    asyncio.run(speak_async(text))

# =========================
# MEDIAPIPE SETUP
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# =========================
# GLOBAL VARIABLES
# =========================
counter = 0
stage = "up"
last_spoken_time = 0

# =========================
# ANGLE CALCULATION
# =========================
def calculateAngle(a, b, c):

    x1, y1 = a[:2]
    x2, y2 = b[:2]
    x3, y3 = c[:2]

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )

    if angle < 0:
        angle += 360

    return angle

# =========================
# POSE DETECTION
# =========================
def detectPose(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    landmarks = []

    if results.pose_landmarks:

        h, w, _ = frame.shape

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        for lm in results.pose_landmarks.landmark:
            landmarks.append((
                int(lm.x * w),
                int(lm.y * h),
                lm.z
            ))

    return frame, landmarks

# =========================
# PUSH-UP DETECTION
# =========================
def detect_pushup(landmarks):

    global counter, stage, last_spoken_time

    feedback = []

    # LEFT SIDE LANDMARKS
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    # ELBOW ANGLE
    angle = calculateAngle(shoulder, elbow, wrist)

    # =========================
    # PUSH-UP COUNTER
    # =========================

    # DOWN POSITION
    if angle < 100:
        stage = "down"

    # UP POSITION
    if angle > 160 and stage == "down":

        stage = "up"
        counter += 1

        feedback.append("Good Rep!")

        if time.time() - last_spoken_time > 2:
            speak_burmese("လုပ်တာကောင်းပါတယ်")
            last_spoken_time = time.time()

    # =========================
    # FORM CHECKS
    # =========================

    # BODY STRAIGHT CHECK
    shoulder_y = shoulder[1]
    hip_y = hip[1]
    ankle_y = ankle[1]

    body_alignment = abs((shoulder_y + ankle_y) / 2 - hip_y)

    if body_alignment > 40:

        feedback.append("Keep your body straight")
        speak_burmese("ကိုယ်ကိုတည့်တည့်ထားပါ")

        if time.time() - last_spoken_time > 3:
            speak_burmese("ကိုယ်ကိုတည့်တည့်ထားပါ")
            last_spoken_time = time.time()

    # GO LOWER
    if angle > 170:
        feedback.append("Lower your body")
        speak_burmese("ပိုနိမ့်ပါ")

    # TOO LOW
    elif angle < 60:
        speak_burmese("အရမ်းမနိမ့်ပါနဲ့")

    return feedback, angle

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("AI Pushup Trainer", cv2.WINDOW_NORMAL)

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        continue

    # SELFIE VIEW
    frame = cv2.flip(frame, 1)

    # RESIZE
    frame = cv2.resize(frame, (1280, 720))

    # DETECT POSE
    frame, landmarks = detectPose(frame)

    # IF LANDMARKS FOUND
    if landmarks:

        feedback, angle = detect_pushup(landmarks)

        # =========================
        # DISPLAY COUNTER
        # =========================
        cv2.putText(
            frame,
            f'Push-ups: {counter}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        # =========================
        # DISPLAY STAGE
        # =========================
        cv2.putText(
            frame,
            f'Stage: {stage}',
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            3
        )

        # =========================
        # DISPLAY ANGLE
        # =========================
        cv2.putText(
            frame,
            f'Angle: {int(angle)}',
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            3
        )

        # =========================
        # DISPLAY FEEDBACK
        # =========================
        y = 220

        for msg in feedback:

            cv2.putText(
                frame,
                msg,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            y += 50

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("AI Pushup Trainer", frame)

    # ESC TO EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()