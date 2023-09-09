import tensorflow as tf
import numpy as np
import os, json

def preprocess_img(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.crop_to_bounding_box(image, int(image.shape[0]*0.2), 0, int(image.shape[0]*0.6), int(image.shape[1]))
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_img(image)

def class_names():
    classes_path = 'C://Users//7XIN//Desktop//N2//AI//DL//SuperCar Image Classification//checkpoint//'
    with open(os.path.join(classes_path,'label_to_index.txt'), 'r')as file:
        label_names = json.loads(file.read())
    label_names = {y:x for x,y in label_names.items()}
    return label_names

def model_importer():
    model = tf.keras.models.load_model('checkpoint/weightings.h5')
    return model

def predictor(image):
    image_ = load_and_preprocess_image(image)
    image2 = np.reshape(image_, [1, 224, 224, 3])
    labels = class_names()
    model = model_importer()
    predictions = labels[np.argmax(model.predict(image2))]
    return predictions