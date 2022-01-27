from keras.models import load_model
import numpy as np
from tkinter import filedialog
from tkinter import *
import tensorflow as tf
import numpy as np


def predict_with_model(model, imgpath):
    
    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [150,150]) # (150,150,3) resizing to all images to 150x150 size
    image = tf.expand_dims(image, axis=0) # (1,150,150,3) increasing the dimension to match the predictor matrix
    prediction_dict = { #cross-reference the pseudo-label with the actual label, refer to data_reference.txt
        0:"Butterfly",
        1:"Dragonfly",
        2:"Grasshoper",
        3:"Ladybug",
        4:"Mosquito"
    }
    predictions = model.predict(image)
    predictions = prediction_dict[np.argmax(predictions)]

    return predictions


if __name__=="__main__":

    # load model folder
    model = load_model('./Mark1_Model') #load your current model here
    root = Tk()
    #change your initialdir to data folder
    root.filename =  filedialog.askopenfilename(initialdir = "E:\Projects\Python\InsectsRecognition\data\Test",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (f"opening {root.filename} file") 
    img_path = root.filename
    prediction = predict_with_model(model,img_path)
    print(f"prediction = {prediction}")