import cv2
import mediapipe as mp
from draw_on_image import draw_landmarks_on_image
# -----------------------------
# MediaPipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -----------------------------
# OpenCV webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect face
    result = landmarker.detect(mp_image)

    expression = "Neutral"

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # Example: mouth opening
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        mouth_open = abs(lower_lip.y - upper_lip.y)

        # Very simple rule
        if mouth_open > 0.04:
            expression = "Surprised"

        # Draw a few landmarks on the web image
        h, w, _ = frame.shape

        for landmark in landmarks[::10]:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0), # green color
                -1 # fill the circle
            )

    # -----------------------------
    # Display expression
    # -----------------------------
    cv2.putText(
        frame,
        f"Expression: {expression}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # visualize some marks
    annotated_frame = draw_landmarks_on_image(frame, result)

    cv2.imshow("Facial Expression Classifier", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()