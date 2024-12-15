import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from traffic_sign.audio_manager import AudioManager
from traffic_sign.sign_tracker import SignTracker
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU memory growth to avoid CUDA errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth configuration failed: {e}")


def detect_and_classify_traffic_signs(video_path, output_path=None):
    conf_threshold = 0.5
    target_size = (80, 80)
    
    try:
        yolo_model = YOLO('exp/model/yolov8_traffic_sign.pt')
        classifier_model = load_model('exp/model/traffic_classifier_best.h5')
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return
    
    audio_manager = AudioManager()
    sign_tracker = SignTracker(audio_manager)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = yolo_model(frame, conf=conf_threshold, imgsz=640)

            for detect in detections:
                for bbox in detect.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    if y2 <= y1 or x2 <= x1:
                        continue
                        
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    try:
                        roi = cv2.resize(roi, target_size)
                        roi = np.expand_dims(roi, axis=0)
                        roi = tf.keras.applications.mobilenet.preprocess_input(roi)
                        predictions = classifier_model.predict(roi, verbose=0)[0]
                        
                        best_label, confidence, _, _ = sign_tracker.update((x1, y1, x2, y2), predictions)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if best_label:
                            cv2.putText(frame, f'{best_label} {confidence:.2f}', 
                                      (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                    except Exception as e:
                        logger.error(f"Error processing detection: {e}")
                        continue

            if output_path:
                out.write(frame)

            cv2.imshow('Traffic Signs Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error in main detection loop: {e}")
    
    finally:
        cap.release()
        if output_path:
            try:
                out.release()
            except:
                pass
        cv2.destroyAllWindows()
        audio_manager.stop()

if __name__ == "__main__":
    video_path = 'test_dataset/raw_data.mp4'
    detect_and_classify_traffic_signs(video_path)