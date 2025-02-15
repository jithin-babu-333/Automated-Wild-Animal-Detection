from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os

app = Flask(__name__, static_folder='static')

# Load your pre-trained CNN model
model = tf.keras.models.load_model('C:\\Users\\ajmal\\Desktop\\final 4\\Model 3, with Resnet early stopping(82 acc,loss 0.68).h5')

# Define a list of class labels
class_labels = ['Notwild', 'Buffalo', 'Elephant', 'Monkey', 'Tiger', 'Wild boar']

# Define a dictionary mapping class labels to descriptions
class_descriptions = {
    'Monkey': "Monkeys are agile and intelligent animals known for their curiosity and mischievous behavior. While they may seem harmless at first glance, they can become dangerous when provoked or when they feel threatened.",
    'Tiger': "Tigers are apex predators and one of the most feared animals in the wild. With their immense strength, sharp claws, and powerful jaws, they are capable of inflicting fatal injuries to humans.",
    'Buffalo': "Buffaloes, also known as water buffalo or bison, are robust and formidable animals. They possess a strong build, sharp horns, and a tendency to become aggressive if they feel threatened.",
    'Wild boar': "Wild boars, also called feral pigs, are sturdy and powerful animals that can pose a threat to human safety.",
    'Elephant': "Elephants are enormous and powerful creatures that command respect in the animal kingdom. While they are generally peaceful, they can become dangerous if they feel endangered or if their young ones are threatened."
}

# Set a threshold value for confidence
confidence_threshold = 0.5  # Adjust as per your preference

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust the size as per your model's input size
    image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform inference on the uploaded image
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence >= confidence_threshold:
        predicted_class = class_labels[predicted_class_index]
        description = class_descriptions.get(predicted_class, 'Description not available')
    else:
        predicted_class = 'Not a wild animal belonging to the 5 classes'
        description = ''

    return predicted_class, confidence, description

# Create a function to capture video from the webcam and process it
def process_webcam():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Check for key press to stop capturing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Preprocess the frame as an image
        image = Image.fromarray(frame)
        predicted_class, confidence, description = predict_image(image)

        # Show the predicted class and confidence on the frame
        cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Description: {description}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the frame with predictions
        cv2.imshow('Webcam with Predictions', frame)

    cap.release()
    cv2.destroyAllWindows()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the user selected the webcam option
        if request.form.get('option') == 'webcam':
            process_webcam()
            return render_template('index.html')

        if 'image' not in request.files:
            return render_template('index.html', message='No image file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', message='No image file selected.')

        try:
            image = Image.open(file)

            # Save the uploaded image to the 'uploads' directory
            image_path = f"uploads/{file.filename}"
            image.save(os.path.join(app.static_folder, image_path))

            predicted_class, confidence, description = predict_image(image)
            return render_template('result.html', image=image_path, predicted_class=predicted_class,
                                   confidence=confidence, description=description)

        except:
            return render_template('index.html', message='Error processing the image.')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


    
