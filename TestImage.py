import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('emotion_recognition_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

image_path = 'path/to/your/image.jpg'

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_resized = cv2.resize(gray_image, (48, 48))
    face_normalized = face_resized / 255.0  # Normalize to [0, 1]
    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))  # Reshape for model input

    predictions = model.predict(face_reshaped)
    emotion_index = np.argmax(predictions)
    emotion_label = emotion_labels[emotion_index]

    confidence = np.max(predictions) * 100

    print(f"Predicted Emotion: {emotion_label} with {confidence:.2f}% confidence")


    label = f"{emotion_label} ({confidence:.2f}%)"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
