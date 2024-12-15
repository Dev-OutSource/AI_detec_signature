import ast
import json
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
import numpy as np
import cv2
import logging
from typing import List
import sys
from pathlib import Path
import base64

# Add the root directory to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from ultralytics import YOLO
from api.schemas import ProcessingImageResponse, TrafficDetection

from traffic_sign.sign_tracker import SignTracker
from api.utils import encode_image_to_base64, decode_base64_to_image, preprocess_traffic
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize models
try:
    detector = YOLO('exp/model/yolov8_traffic_sign.pt')
    classifier_model = load_model('exp/model/traffic_classifier_best.h5')
    sign_tracker = SignTracker()
    labels = ['DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106P', 'P.107a', 'P.112', 'P.115',
          'P.117', 'P.123a', 'P.123b', 'P.124a', 'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128', 'P.130', 'P.131a',
          'P.137', 'P.245a', 'R.301c', 'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303', 'R.407a', 'R.409', 'R.425',
          'R.434', 'S.509a', 'W.2s01a', 'W.201b', 'W.202a', 'W.202b', 'W.203b', 'W.203c', 'W.205a', 'W.205b',
          'W.205d', 'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210', 'W.219', 'W.221b', 'W.224', 'W.225',
          'W.227', 'W.233', 'W.235', 'W.245a']
    
    conf_threshold = 0.25
    
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

@app.post("/process-image", response_model=ProcessingImageResponse)
async def process_image(base64_image: str,  tracked_signs: str = None):
    
    try:
        base64_image = base64_image.strip()
        
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]
            
        image_data = base64.b64decode(base64_image)

        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        traffics_info: List[TrafficDetection] = []

        # Detect faces
        detector = YOLO('exp/model/yolov8_traffic_sign.pt')
        detections = detector(image, conf=conf_threshold, imgsz=640)
        
        processed_tracked_signs = []
        if tracked_signs:
            # Chuyển string thành list Python
            processed_tracked_signs = ast.literal_eval(tracked_signs) if tracked_signs is not None else None
        
        sign_tracker.set_tracked_signs(processed_tracked_signs)
        for result in detections[0].boxes:
            bbox = result.xyxy[0].cpu().numpy()  # Convert to numpy array
            detect_confidence = float(result.conf[0])
            
            x1, y1, x2, y2 = map(int, bbox)
            
            if y2 <= y1 or x2 <= x1:
                continue
                
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            try:
                roi = preprocess_traffic(roi)
                predictions = classifier_model.predict(roi, verbose=0)[0]
                
                best_label, confidence, processed_tracked_signs, tracked_time = sign_tracker.update((x1, y1, x2, y2), predictions)
                traffics_info.append(TrafficDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=detect_confidence,
                    traffic=str(best_label),
                    traffic_confidence=confidence,
                    tracked_time=tracked_time
                ))
                
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{best_label} {confidence:.2f}', 
                            (x1 + 10, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                continue
        
        return ProcessingImageResponse(
            success=True,
            message="Image processed successfully",
            num_traffics=len(traffics_info),
            traffics=traffics_info,
            tracked_signs=processed_tracked_signs,
            processed_image=encode_image_to_base64(image)
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))