# Surprise Face Recognition

A simple computer vision project that uses OpenCV and MediaPipe Face Landmarker to detect facial features associated with a surprised facial expression in real time.

This project is designed as an introductory Computer Vision + Human-Computer Interaction (HCI) project. It demonstrates how facial landmarks and facial blendshape coefficients can be used to build a simple expression classifier.

## How It Works

The program follows this pipeline:

Webcam
   ↓
OpenCV
   ↓
MediaPipe Face Landmarker
   ↓
Facial Landmarks (+ Blendshape Coefficients)
   ↓
Selected Facial Features
   ↓
Rule-Based Surprise Detector
   ↓
"Surprised" / "Neutral"

## Surprise Detected Using Mouth Opening 
(To be improved with facial blendshape coefficients)

This start package uses a simple approach calculating:

mouth_open = abs(lower_lip.y - upper_lip.y)

However, an open mouth can occur when someone is:

talking
yawning
laughing
surprised
singing

Therefore, mouth opening alone is not sufficient to identify surprise.

## Facial Features

This program can be improved by using several MediaPipe face blendshapes related to the facial actions commonly associated with surprise:

browInnerUp
browOuterUpLeft
browOuterUpRight
eyeWideLeft
eyeWideRight
jawOpen

Conceptually, surprise involves:

raised eyebrows
widened eyes
an open jaw/mouth

Using several features together is more reliable than using mouth opening alone.

## Important Note About the Thresholds

The improved classifier can use simple threshold values such as:

if (
    brow_inner > 0.20
    and brow_outer > 0.20
    and eye_wide > 0.25
    and jaw_open > 0.20
):
    expression = "Surprised"

These values are starting points for experimentation, not universal scientific thresholds.

Different people have different neutral facial expressions. A more advanced system should calibrate each person's neutral face and/or train a machine-learning model using labeled examples.

## Requirements
Python 3.11+
OpenCV
MediaPipe
NumPy

Install the required packages with:

pip install -r requirements.txt

## MediaPipe Model

This project requires the MediaPipe Face Landmarker model:

face_landmarker.task

Place the model in the models folder:

project/
│
│── face_landmarker.task
│
├── draw_on_image.py
├── face_basic.py
├── requirements.txt
└── README.md

The Face Landmarker model provides facial landmarks and optional facial blendshape coefficients (set `output_face_blendshapes=True` in `options`).

## Running the Program

After installing the requirements and downloading the model:

`python face_basic.py`

The program will open the webcam and display the detected facial information.

Press:

q

to quit.

## Understanding Blendshape Scores

MediaPipe provides a score for each blendshape.

For example:

browInnerUp = 0.42
eyeWideLeft = 0.55
eyeWideRight = 0.51
jawOpen = 0.38

The scores generally range from approximately 0 to 1, representing the strength of the detected facial movement.

A single score should not be interpreted as an emotion by itself. The project combines multiple facial features to estimate an apparent facial expression.

This project combines eyebrow movement, eye widening, and jaw opening.

## Future Improvements

Possible next steps include:

Add blendshape scores for surprise recognition.
Add more facial expressions such as neutral, positive, confused, and low visible engagement.
Use temporal information instead of classifying individual frames.
Calculate a person's neutral baseline.
Smooth the blendshape values over several frames.
Train a machine-learning classifier using labeled facial-expression data.
Compare rule-based classification with Random Forest, SVM, or neural-network approaches.
Support multiple faces.
Save blendshape data to CSV for later analysis.
Create a classroom-level reaction dashboard.

## Ethical Considerations

Facial expressions should not be treated as a direct measurement of a person's internal emotional state.

For example, a person who looks neutral may still be interested, confused, tired, or thinking.

Therefore, this project should be described as detecting apparent facial expressions or visible facial cues, rather than determining exactly how a person feels.

For classroom applications, avoid using facial-expression predictions as high-stakes judgments about individual students.

## References
Google MediaPipe Face Landmarker documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
MediaPipe Face Landmarker blendshape documentation:
https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes
Ekman & Friesen, Facial Action Coding System (FACS).

## License
This project is intended for educational and experimental purposes.
