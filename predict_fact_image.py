import cv2
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import io

conf_threshold = 0.5
target_size = (80, 80)

labels = ['DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106P', 'P.107a', 'P.112', 'P.115',
          'P.117', 'P.123a', 'P.123b', 'P.124a', 'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128', 'P.130', 'P.131a',
          'P.137', 'P.245a', 'R.301c', 'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303', 'R.407a', 'R.409', 'R.425',
          'R.434', 'S.509a', 'W.2s01a', 'W.201b', 'W.202a', 'W.202b', 'W.203b', 'W.203c', 'W.205a', 'W.205b',
          'W.205d', 'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210', 'W.219', 'W.221b', 'W.224', 'W.225',
          'W.227', 'W.233', 'W.235', 'W.245a']

# Load models
yolo_model = YOLO('exp/model/yolov8_traffic_sign.pt')
classifier_model = load_model('exp/model/traffic_classifier_best.h5')

sign_meanings = pd.read_csv('sign_meanings.csv')
sign_meanings.set_index('Mã biển báo', inplace=True)

def preprocess_frame(roi):
    roi = cv2.resize(roi, target_size)
    roi = np.expand_dims(roi, axis=0)
    roi = tf.keras.applications.mobilenet.preprocess_input(roi)
    return roi

def download_image_from_url(source):
    if source.startswith(('http://', 'https://')):  # Nếu là URL
        response = requests.get(source)
        img = Image.open(io.BytesIO(response.content))
        img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:  # Nếu là đường dẫn local
        img = cv2.imread(source)
    return img

def classify_traffic(image_url):
    frame = download_image_from_url(image_url)
    roi = preprocess_frame(frame)
    predictions = classifier_model.predict(roi, verbose=0)[0]
    label_id = np.argmax(predictions)
    confidence = predictions[label_id]
    cv2.imshow(frame)

    return labels[label_id], confidence

def detect_and_classify_traffic_signs_from_url(image_url, output_path=None):
    # Download and load image from URL
    frame = download_image_from_url(image_url)

    # Run detection
    detections = yolo_model(frame, conf=conf_threshold, imgsz=640)
    for detect in detections:
        for bbox in detect.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi = preprocess_frame(roi)
            predictions = classifier_model.predict(roi, verbose=0)[0]

            label_id = np.argmax(predictions)
            confidence = predictions[label_id]
            
            processed_label = sign_meanings.loc[labels[label_id], 'Ý nghĩa']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{labels[label_id]} {confidence:.2f}', (x1 + 10, y1 + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print("-"*30 + "\n\n")
            print(labels[label_id], processed_label, confidence)
            print("\n\n" + "-"*30)

    if output_path:
        cv2.imwrite(output_path, frame)

    # Display the result
    cv2.imshow('Traffic Signs Detection', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run with an image URL
image_url = "https://thietbigiaothongthanhtri.com/upload/images/bien-bao-di-cham-w245-thanh-tri-4.jpg"
output_path = "exp/output/output_fact_image.jpg"
predict_label = detect_and_classify_traffic_signs_from_url(image_url, output_path)