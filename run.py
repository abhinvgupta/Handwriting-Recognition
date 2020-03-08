from flask import Flask, render_template, request

from skimage.transform import resize
import imageio
import numpy as np 
import keras.models
import re
import sys
import base64
import os
sys.path.append(os.path.abspath('./model'))
from loadModel import *

#init flask app

app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData):
	imgstr = re.search(b'base64,(.*)',imgData).group(1)
	with open('output.png','wb') as output:
		print(base64.b64decode(imgstr))
		output.write(base64.b64decode(imgstr))


def imgReshape(x):
	imgArray = np.zeros((28,28,1))
	for i1, u in enumerate(x):
		for i2, v in enumerate(u):
			imgArray[i1,i2,0] = v[0]

	imgArray = imgArray.reshape(1,28,28,1)
	return imgArray


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict/', methods = ['GET','POST'])
def predict():
	imgData = request.get_data() 
	convertImage(imgData)
	x = imageio.imread('output.png')
	print (str(x[0][100]), x.shape)
	x = np.invert(x)
	x = resize(x, (28,28))
	x = imgReshape(x)
	with graph.as_default():
		out = model.predict(x)
		response = str(np.argmax(out))
		return response

print (model)
if __name__ == '__main__':
	port = 5000
	app.run(port=port)





