import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
tflite_model_path = 'C:/Users/john/OneDrive/Documents/THESIS MODEL/educational-tool/Feedforward Neural Network - BlazePose Model/model_fnn.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize BlazePose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with BlazePose
    results = pose.process(frame_rgb)

    # Recolor image back to BGR for rendering
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Detect Taijiquan Stances (class)
    if results.pose_landmarks:
        # Extract Pose landmarks
        pose_landmarks = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())

        # Make Detections
        X = pd.DataFrame([pose_row])

        # Convert X to numpy array
        input_data = X.to_numpy().astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class and probabilities
        body_language_class = np.argmax(output_data)
        body_language_prob = output_data[0]

        # Convert landmark coordinates to integers
        landmarks_as_pixels = np.array([(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in results.pose_landmarks.landmark])

        # Calculate bounding rectangle
        bbox_c = cv2.boundingRect(landmarks_as_pixels)

        # Draw the bounding box
        cv2.rectangle(frame, (int(bbox_c[0]), int(bbox_c[1])), (int(bbox_c[0] + bbox_c[2]), int(bbox_c[1] + bbox_c[3])), (0, 255, 0), 2)

        # Display Probability
        cv2.putText(frame, 'PROB', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(body_language_prob[body_language_class], 2)),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display detected class
        cv2.putText(frame, f'CLASS: {body_language_class}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 191, 255), thickness=2, circle_radius=2)
        )

        # Add class label on the bounding box
        cv2.putText(frame, f'CLASS: {body_language_class}', (int(bbox_c[0]), int(bbox_c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Pose Detection', frame)

    # Check for exit key (q)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
