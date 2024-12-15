import cv2
import numpy as np
import base64
import tensorflow as tf

def encode_image_to_base64(image):
    """Convert CV2 image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def decode_base64_to_image(base64_string):
    """Convert base64 string to CV2 image"""
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def preprocess_traffic(traffic_image, target_size=(80, 80)):
    """Preprocess traffic( image for classification"""
    traffic_image = cv2.resize(traffic_image, target_size)
    traffic_image = np.expand_dims(traffic_image, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(traffic_image)