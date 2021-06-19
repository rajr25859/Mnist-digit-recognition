from flask import Flask, render_template, request, redirect
from PIL import Image
import joblib
from matplotlib import pyplot as plt 
import os
import pickle
import png
import numpy as np

app = Flask(__name__)


knn = joblib.load('D:\\DATA FOR MACHINE LEARNING PROJECT\\MNIST_model_project\\webapp\\model\\knn.pickle')

@app.route('/', methods = ['GET', 'POST'])
def home():
    msg = ''
    msg2 = ''
    valid_exts = ['png']
    flag = None
    if request.method == 'POST':
        image = request.files['uploaded_item']
        if image.filename[-3:] in valid_exts:
            msg = 'Accepted file type'
            flag  = True
        else:
            msg = 'File not accepted'
        
        img = Image.open(image)
        size1 = img.size[0]
        size2 = img.size[1]

        if size1 == size2:
            msg2 = 'Size accepted'
        else:
            msg2 = 'Size Not Accepted'


        image.save('D:\\DATA FOR MACHINE LEARNING PROJECT\\MNIST_model_project\\webapp\\uploads\\image.png')

        ifile = png.Reader('D:\\DATA FOR MACHINE LEARNING PROJECT\\MNIST_model_project\\webapp\\uploads\\image.png')
        ip = ifile.read()
        img = []
        for ar in list(ip[2]):
            img.append([x for x in ar])
        img = np.asarray(img)
        img = img.flatten()
        pred = knn.predict([img])
        msg = 'Predicted Digit is {}'.format(pred)

        os.system('del "D:\\DATA FOR MACHINE LEARNING PROJECT\\MNIST_model_project\\webapp\\uploads\\image.png"')


    return render_template('upload.html', msg=msg,msg2 = msg2)


if __name__ == '__main__':
    app.run(debug=True)