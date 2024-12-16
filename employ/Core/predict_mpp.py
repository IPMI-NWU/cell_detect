from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "Model", "mpp", "inception_v3_mpp_1024.h5")
model = load_model(model_path)
graph = tf.get_default_graph()


def predict_mpp(image):
    with graph.as_default():
        image = image[0:1024,0:1024,:]
        cv2.imwrite("test.png", image)
        image = np.expand_dims(image, axis=0)
        prediction = float(model.predict(image)[0][0])
        return prediction

