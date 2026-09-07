import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
# ============================================================
# CAMERA ACTION AI
# ============================================================
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")
# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)
# ============================================================
# CAMERA SETUP
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()
# ============================================================
# RECORDING VARIABLES
# ============================================================
recording = False
video_writer = None
start_time = 0
VIDEO_DURATION = 5
ACTION_COOLDOWN = 0.8
last_action_time = 0
# ============================================================
# ACTION MESSAGE
# ============================================================
display_action = "Waiting..."
action_display_until = 0
# ============================================================
# WAVE + GESTURE STABILITY
# ============================================================

x_history = deque(maxlen=10)
gesture_history = deque(maxlen=5)


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(landmarks):

    if landmarks is None:
        return None

    # --------------------------------------------------------
    # Landmark points
    # --------------------------------------------------------

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    index_tip = landmarks[8]
    index_pip = landmarks[6]

    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    wrist = landmarks[0]


    # --------------------------------------------------------
    # Finger positions
    # --------------------------------------------------------

    index_up = (
        index_tip[1] < index_pip[1] - 0.02
    )

    middle_up = (
        middle_tip[1] < middle_pip[1] - 0.02
    )

    ring_up = (
        ring_tip[1] < ring_pip[1] - 0.02
    )

    pinky_up = (
        pinky_tip[1] < pinky_pip[1] - 0.02
    )


    # ========================================================
    # PEACE ✌️
    # ========================================================

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):

        return "peace"


    # ========================================================
    # THUMBS UP 👍
    # ========================================================

    # Thumb must be above its joint
    thumb_up = (
        thumb_tip[1] < thumb_ip[1] - 0.03
    )

    # Other four fingers must be folded
    fingers_folded = (
        not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    )

    # Thumb should be clearly separated from wrist
    thumb_extended = (
        thumb_tip[1] < wrist[1] - 0.10
    )

    if (
        thumb_up
        and fingers_folded
        and thumb_extended
    ):

        return "thumbs_up"


    # ========================================================
    # OPEN PALM / WAVE ✋👋
    # ========================================================

    if (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    ):

        current_x = index_tip[0]

        x_history.append(current_x)


        # Need enough movement history
        if len(x_history) >= 8:

            movement_range = (
                max(x_history)
                - min(x_history)
            )


            # Moving hand = WAVE
            if movement_range > 0.15:

                return "wave"


        # Stationary hand = PALM
        return "palm"


    return None


# ============================================================
# STABLE GESTURE
# ============================================================

def get_stable_gesture(new_gesture):

    gesture_history.append(new_gesture)

    if len(gesture_history) < 3:
        return None

    recent = list(gesture_history)

    if (
        recent[-1] is not None
        and recent[-1] == recent[-2]
        and recent[-2] == recent[-3]
    ):

        return recent[-1]

    return None


# ============================================================
# ACTION MESSAGE FUNCTION
# ============================================================

def set_action(message, seconds=2):

    global display_action
    global action_display_until

    display_action = message

    action_display_until = (
        time.time() + seconds
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Unable to read camera.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe processing
    results = hands.process(rgb_frame)


    gesture = None


    # ========================================================
    # HAND DETECTION
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # Convert landmarks
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ])


        # Detect gesture
        detected_gesture = recognize_gesture(
            landmarks
        )


        # Make gesture stable
        gesture = get_stable_gesture(
            detected_gesture
        )


    else:

        gesture_history.clear()
        x_history.clear()


    # ========================================================
    # CURRENT TIME
    # ========================================================

    current_time = time.time()


    # ========================================================
    # GESTURE DISPLAY
    # ========================================================

    if gesture is not None:

        gesture_text = (
            "Gesture: "
            + gesture.upper()
        )

    else:

        gesture_text = "Gesture: None"


    # ========================================================
    # PEACE → PHOTO
    # ========================================================

    if gesture == "peace":

        if (
            current_time - last_action_time
            > ACTION_COOLDOWN
        ):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                "captured_images/"
                "photo_"
                + timestamp
                + ".jpg"
            )


            # Save photo
            cv2.imwrite(
                filename,
                frame
            )


            print(
                "Photo saved:",
                filename
            )


            set_action(
                "PHOTO CAPTURED!",
                2
            )


            last_action_time = current_time


    # ========================================================
    # THUMBS UP → CONFIRM
    # ========================================================

    elif gesture == "thumbs_up":

        set_action(
            "CONFIRMED!",
            2
        )


    # ========================================================
    # PALM → STOP / CANCEL
    # ========================================================

    elif gesture == "palm":

        set_action(
            "STOP / CANCEL",
            1
        )


    # ========================================================
    # WAVE → RECORD
    # ========================================================

    elif gesture == "wave":

        if (
            not recording
            and current_time - last_action_time
            > ACTION_COOLDOWN
        ):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            video_filename = (
                "captured_images/"
                "video_"
                + timestamp
                + ".mp4"
            )


            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            fps = 20.0


            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )


            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )


            video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                fps,
                (width, height)
            )


            recording = True

            start_time = current_time

            last_action_time = current_time


            print(
                "Recording started:",
                video_filename
            )


            set_action(
                "RECORDING...",
                5
            )


    # ========================================================
    # RECORD VIDEO
    # ========================================================

    if recording:

        elapsed = (
            current_time - start_time
        )


        if video_writer is not None:

            video_writer.write(frame)


        remaining = max(
            0,
            int(VIDEO_DURATION - elapsed)
        )


        display_action = (
            "RECORDING... "
            + str(remaining)
            + "s"
        )


        action_display_until = (
            current_time + 0.2
        )


        # Finish recording
        if elapsed >= VIDEO_DURATION:

            recording = False


            if video_writer is not None:

                video_writer.release()

                video_writer = None


            print(
                "Video recording completed."
            )


            set_action(
                "VIDEO SAVED!",
                2
            )


    # ========================================================
    # WAITING STATUS
    # ========================================================

    if current_time > action_display_until:

        display_action = "Waiting..."


    action_text = (
        "Action: "
        + display_action
    )


    # ========================================================
    # TOP UI
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (800, 125),
        (20, 20, 20),
        -1
    )


    # Title
    cv2.putText(
        frame,
        "CAMERA ACTION AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    # Gesture
    cv2.putText(
        frame,
        gesture_text,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # Action
    cv2.putText(
        frame,
        action_text,
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # GESTURE GUIDE
    # ========================================================

    guide = (
        "PEACE = PHOTO | "
        "THUMBS UP = CONFIRM | "
        "PALM = STOP | "
        "WAVE = RECORD"
    )


    cv2.putText(
        frame,
        guide,
        (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Camera Action AI",
        frame
    )


    # ========================================================
    # QUIT WITH Q
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

if video_writer is not None:

    video_writer.release()

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Camera Action AI stopped.")
