from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your trained model (assuming you already have it trained)
model_path = r"C:\Users\vivek\Downloads\trained_model.h5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        img = Image.open(image)
        img = img.resize((28, 28))  # Resize the image to match model input size
        img = img.convert('L')  # Convert to grayscale
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
        return img_array
    except (IOError, OSError, ValueError) as e:
        # Handle errors during image processing
        print("Error: Unable to preprocess image:", e)
        return None

# Function to predict the digit in the image
def predict_digit(image):
    img_array = preprocess_image(image)
    if img_array is not None:
        predicted_probabilities = model.predict(img_array)
        predicted_class = np.argmax(predicted_probabilities)
        return predicted_class
    else:
        return None

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            predicted_digit = predict_digit(file)
            if predicted_digit is not None:
                return render_template('result.html', digit=predicted_digit)
            else:
                return "Error: Unable to predict digit."
    return "Error: No file provided."

if __name__ == '__main__':
    app.run(debug=True)
