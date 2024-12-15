import numpy as np
import time

class SignTracker:
    def __init__(self, audio_manager = None):
        self.tracked_signs = []  # [(box, predictions, time, announced, last_update)]
        self.position_threshold = 10
        self.audio_manager = audio_manager
        self.labels = ['DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106P', 'P.107a', 'P.112', 'P.115', 
          'P.117', 'P.123a', 'P.123b', 'P.124a', 'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128', 'P.130', 'P.131a', 
          'P.137', 'P.245a', 'R.301c', 'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303', 'R.407a', 'R.409', 'R.425', 
          'R.434', 'S.509a', 'W.201a', 'W.201b', 'W.202a', 'W.202b', 'W.203b', 'W.203c', 'W.205a', 'W.205b', 
          'W.205d', 'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210', 'W.219', 'W.221b', 'W.224', 'W.225', 
          'W.227', 'W.233', 'W.235', 'W.245a']
        
    def is_same_position(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        return (abs(x1_1 - x1_2) < self.position_threshold and 
                abs(y1_1 - y1_2) < self.position_threshold and 
                abs(x2_1 - x2_2) < self.position_threshold and 
                abs(y2_1 - y2_2) < self.position_threshold)
    
    def cleanup_old_tracks(self, current_time):
        """Xóa các track cũ không được cập nhật sau 2 giây"""
        self.tracked_signs = [
            track for track in self.tracked_signs 
            if (current_time - track[4]) <= 2.5
        ]
    
    def predict(self, predictions):
        label_id = np.argmax(predictions)
        sign_code = self.labels[label_id]
        conf = predictions[label_id]
        
        return sign_code, conf
    
    def update(self, bbox, predictions):
        current_time = time.time()
        
        # Dọn dẹp các track cũ
        self.cleanup_old_tracks(current_time)
        
        # Kiểm tra xem box mới có trùng với box nào đã tracked không
        for i, track in enumerate(self.tracked_signs):

            tracked_box, tracked_predictions, tracked_time, announced, _ = track
            tracked_predictions = np.array(tracked_predictions)
            if self.is_same_position(bbox, tracked_box):
                
                # Cập nhật predictions bằng cách cộng dồn
                new_time = tracked_time + 1
                updated_predictions = tracked_predictions + predictions
                
                sign_code, new_conf = self.predict(updated_predictions / new_time)
                
                # Kiểm tra điều kiện phát âm thanh
                if new_time >= 5 and new_conf > 0.5 and not announced and self.audio_manager is not None:
                    # Phát âm thanh
                    self.audio_manager.add_to_queue(sign_code)
                    announced = True
                
                self.tracked_signs[i] = (bbox, updated_predictions.tolist(), new_time, announced, current_time)
                return sign_code, new_conf, self.tracked_signs, new_time
        
        # Thêm track mới với timestamp
        self.tracked_signs.append((bbox, predictions.tolist(), 1, False, current_time))
        sign_code, new_conf = self.predict(predictions)
        
        return sign_code, new_conf, self.tracked_signs, 1
    
    def set_tracked_signs(self, tracked_signs):
        self.tracked_signs = tracked_signs if tracked_signs is not None else []
        