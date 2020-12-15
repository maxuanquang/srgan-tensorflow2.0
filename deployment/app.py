from flask import Flask, render_template,request, url_for
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from model import get_G
import cv2
import tensorlayer as tl

app = Flask(__name__)
app.debug=True
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
	input_g_shape =([None, None, 3])
	G = get_G(input_g_shape)
	name = ""
	with tf.device('cpu'):
		G.load_weights("models/g_maevgg_60.h5")
	if request.method == 'POST':
		image = request.files['imgInp'].read()
		npimg = np.fromstring(image, np.uint8)
		image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		img = image[:,:,::-1]
		normalized_img = img / 127.5 - 1
		normalized_img = normalized_img[np.newaxis,:,:,:]
		my_prediction = G(normalized_img).numpy()
		name = request.files['imgInp']
		name = name.filename
		# cv2.imwrite('static/images/'+str(name),img)
		tl.vis.save_image(img, "static/images/"+str(name))
		tl.vis.save_image(my_prediction[0], "static/images_target/"+str(name))
	return render_template('home.html',filename=str(name))


if __name__ == '__main__':
	app.run()
