import numpy as np
import tensorflow as tf
from PIL import Image

# Carregar model entrenat
model = tf.keras.models.load_model("model/model.h5")

# Map de classes EMNIST Balanced
classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

def preprocess(img):
    img = img.convert("L")  # grisos
    img = img.resize((28,28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=(0,-1))
    return img_arr

def predict(img):
    arr = preprocess(img)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return classes[idx], preds[0][idx]
