import keras
from keras.preprocessing import image 
import os
import numpy as np

model_path = os.path.join(os.getcwd(), "models/inception_kaufland_val_acc_80.h5")
model = keras.models.load_model(model_path)

eans = [4001686301265,
		4300175162883,
		4300175162920,
		4337185077241,
		4337185153013,
		4337185240386,
		4337185276682,
		4337185303739,
		4337185373619,
		4337185396243,
		4337185396748,
		4337185558740]

eans_dict = {
				4001686301265: "Haribo Goldbären",
				4300175162883: "Sauerkirschen",
				4300175162920: "Aprikosen (halbe Frucht)",
				4337185077241: "Crazy Wolf Energy Drink",
				4337185153013: "Choco Happs",
				4337185240386: "Cornflakes",
				4337185276682: "Pfirsiche (halbe Frucht)",
				4337185303739: "Choc It!",
				4337185373619: "Chips (Paprika)",
				4337185396243: "Weizenmehl Typ 405",
				4337185396748: "Feinzucker",
				4337185558740: "Williams Christ Birnen"
			}


def predict_class_num(path):
	data = getImageData(path)
	pred = model.predict(data)
	print(pred)
	return np.argmax(pred)

def getEAN(path):
	num_class = predict_class_num(path)
	ean = eans[num_class]
	name = eans_dict[ean]
	return (ean, name)

def getImageData(path):
	# predicting multiple images at once
	img = image.load_img(path, target_size=(299, 299))
	y = image.img_to_array(img)
	y = np.expand_dims(y, axis=0)
	y *= 1.0/255
	return y

def classify():
	path = os.path.join(os.getcwd(), "img_0")
	predicted = predict_class_num(path)
	print('Product ' + str(eans_dict.get(eans[predicted])) + ' with EAN-13: ' + str(eans[predicted]) + ' found.')
	return eans[predicted], eans_dict.get(eans[predicted])