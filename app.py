from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# from PIL import Image
import os

app = Flask(__name__)
model = load_model('SKIN Diseases.h5')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#classes = ['Eczema', 'Warts Molluscum and other Viral Infections', 'Atopic Dermatitis','Melanocytic Nevi', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors','Tinea Ringworm Candidiasis and other Fungal Infections']
classes = ['Atopic Dermatitis', 'Eczema', 'Melanocytic Nevi', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Warts Molluscum and other Viral Infections']

def prepare_image(file_path):
    img = image.load_img(file_path,target_size = (244,244))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image to [0, 1] range
    return img

def getPrediction(pred):
    return np.argmax(pred)

def getClass(pred):
    return classes[pred]

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = prepare_image(file_path)
            prediction1 =getPrediction(model.predict(img))
            prediction2 =getPrediction(model.predict(img))
            meanPred=round((prediction1+prediction2)/2)
            prediction=getClass(meanPred)
            return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
