from flask import Flask, render_template, request, redirect
import tensorflow as tf
import cv2
import os

model = tf.keras.models.load_model('/Users/mm22/Documents/Development/catdog_demo')

CATEGORIES = ['Cat', 'Dog']

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(f'uploads/{file.filename}')
    prediction = model.predict([prepare(f'uploads/{file.filename}')])
    result = CATEGORIES[int(prediction[0][0])]
    #return redirect('/')
    return render_template('index.html', variable=result)

if __name__ == '__main__':
    app.run(debug=True)