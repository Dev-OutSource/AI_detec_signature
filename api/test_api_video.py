import cv2
import requests
import json
from typing import Optional
import time
import numpy as np

from traffic_sign.audio_manager import AudioManager

class VideoProcessor:
    def __init__(self, api_url: str):
        self.api_url = api_url
        # Define color map for different traffic signs
        self.color_map = {
            'default': (0, 255, 0)  # Green for default
        }
        self.audio_manager = AudioManager()

    def process_frame(self, frame, tracked_signs = None) -> Optional[dict]:
        """Process a single frame by sending it to the API"""
        try:
            # Encode frame to jpg format
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Prepare the file for sending
            files = {
                'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            data = {
                'tracked_signs': str(tracked_signs) if tracked_signs is not None else None
            }
            
            # Send POST request to API
            response = requests.post(self.api_url, files=files, data=data)
            
            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error processing frame: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def draw_traffic_signs(self, frame, traffic_signs):
        """Draw bounding boxes and labels for traffic signs"""
        frame_with_boxes = frame.copy()
        
        for sign in traffic_signs:
            # Extract information
            bbox = sign['bbox']
            detect_confidence = sign['confidence']
            traffic_type = sign['traffic']
            traffic_confidence = sign['traffic_confidence']
            tracked_time = sign['tracked_time']
            

            # Convert bbox tuple string to integers
            x1, y1, x2, y2 = map(int, bbox)

            if traffic_confidence > 0.3 and tracked_time > 3:
                # Get color for this traffic sign type
                color = self.color_map.get(traffic_type, self.color_map['default'])

                # Draw bounding box
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)

                # Create label with traffic sign type and confidence
                label = f'{traffic_type} ({traffic_confidence:.2f})'
                
                # Calculate text size and position
                font_scale = 0.6
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame_with_boxes, (x1, y1 - text_height - 5), (x1 + text_width, y1), color)
                
                # Draw text
                cv2.putText(frame_with_boxes, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

            if traffic_confidence > 0.5 and tracked_time > 5:
                self.audio_manager.add_to_queue(traffic_type)
        return frame_with_boxes

    def process_video(self, video_path: str, output_path: str = None, fps: int = 30):
        """Process video file and show/save results"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if output path is specified
        out = None
        tracked_signs = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        process_every_n_frames = int(original_fps / fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Process only every nth frame to maintain desired FPS
            if frame_count % process_every_n_frames != 0:
                continue

            # Process frame
            result = self.process_frame(frame, tracked_signs)
            
            if result and result.get('success'):
                # Get traffic signs information
                traffic_signs = result.get('traffics', [])
                tracked_signs = result.get('tracked_signs', []) 
                
                # Draw traffic signs on frame
                processed_frame = self.draw_traffic_signs(frame, traffic_signs)
                
                # Display processing info
                print(f"Frame {frame_count}: Detected {len(traffic_signs)} traffic signs")
                for sign in traffic_signs:
                    print(f"  - Traffic sign: {sign['traffic']}, Confidence: {sign['traffic_confidence']:.2f}")
                
                # Write frame if output is specified
                if out:
                    out.write(processed_frame)
                
                # Show frame
                cv2.imshow('Processed Video', processed_frame)
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

def main():
    # Configuration
    API_URL = "http://localhost:8000/process-image"  # Change this to your API URL
    VIDEO_PATH = "test_dataset/raw_data.mp4"  # Change this to your input video path
    OUTPUT_PATH = "output_video.mp4"  # Optional: set to None to disable saving
    OUTPUT_FPS = 30  # Desired output FPS

    # Initialize video processor
    
    processor = VideoProcessor(API_URL)

    try:
        # Process video
        processor.process_video(VIDEO_PATH, OUTPUT_PATH, OUTPUT_FPS)
        print("Video processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()