from flask import Flask, flash, request, redirect, url_for, render_template, session
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import sklearn
from collections import deque
import pickle
import numpy as np
import cv2
from PIL import Image
import keras.utils as image



# Initializing flask for website, giving name "app", "app" name will be used for configuring.
# static_url_path -> is the image folder location
app = Flask(__name__, static_url_path='/static')
app.secret_key = "112233"

# linking image upload folder
UPLOAD_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Defining image format file only png, jpg, jpeg, gif file will supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


# This function will be used for checking a file is image or not.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home page linking using route
# render_template() -> this function used for calling html pages.
# It search the templates folder for index.html file
# route('/') refers the homepage or index.html page
@app.route('/')
def home():
    return render_template('index.html')


# This is used for image upload and model prediction.
# GET, POST methods is used for take something from html file and return something to html file.
@app.route('/', methods=['POST', 'GET'])
def upload_image():
    # Step 1 : Image upload
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    # Checks the file is image or not
    if file and allowed_file(file.filename):
        # get the filename
        filename = secure_filename(file.filename)
        # save the file in upload folder variable. which is static->images.
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # flash used for display messages in html file
        flash('Image successfully uploaded and displayed below')


        # Step 2 : model loading
        model_path = "model.h5"
        model = load_model(model_path)
        # load the label file for disease name
        lb = pickle.loads(open("label", "rb").read())

        # Step 3 : Image resizing. this is used for image resizing
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
        Q = deque(maxlen=128)

        # Step 4 : image load from upload folder
        img = "static/images/"+filename
        img = image.load_img(img)
        x = image.img_to_array(img)
        frame = cv2.resize(x, (224, 224))
        frame -= mean
        # frame is the resized image

        # Step 5 : Prediction
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]

        # Step 6 : text -> Prediction result save in text variable
        text = "DETECTED DISEASE : {}".format(label.upper())

        # Step 7 : Passing the image filename and prediction result to index.html file
        # image filename saved in "imagefilename" and prediction result saved in "result" variable
        return render_template('index.html', imagefilename = filename , result = text)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


# same as main function.
if __name__ == "__main__":
    app.run()
