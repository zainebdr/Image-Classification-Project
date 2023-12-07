from __future__ import division, print_function
import sys
import os
from os import path
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = 'my_model'
print(MODEL_PATH)

# Load trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  

# Define the uploads directory
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

def process_results(preds):
    class_mapping = {0: "COVID", 1: "NORMAL", 2: "PNEUMONIA"}
    predicted_class = class_mapping[np.argmax(preds)]
    return predicted_class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("fillle ", f)

        # Check if the uploads directory exists, create it if not
        if not path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the file to the uploads directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        print("file path ", file_path)
        f.save(file_path)
        print("fillle ", f)
        # Make prediction
        preds = model_predict(file_path, model)

        # Process the result
        result = process_results(preds)
        return result
    return None
    

if __name__ == '__main__':
    app.run(debug=True)
