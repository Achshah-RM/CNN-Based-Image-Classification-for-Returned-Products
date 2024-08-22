from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import base64
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Define the API endpoint for batch processing
@app.route('/process_batch', methods=['POST'])
def process_batch():
    data = request.get_json()
    image_list = data.get('images', [])
    predictions = []

    for img_str in image_list:
        # Decode the image from base64
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))  
        img = np.array(img) / 255.0  
        img = img.reshape(1, 28, 28, 1)  

        # Predict the class
        pred = model.predict(img)
        pred_class = int(np.argmax(pred, axis=1)[0]) 
        predictions.append(pred_class)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)