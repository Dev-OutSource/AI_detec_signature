import pygame
import threading
import time
import pandas as pd
import logging
import io
from queue import Queue
from traffic_sign.tts import TTSViettelAI

class AudioManager:
    def __init__(self):
        # Khởi tạo logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Khởi tạo các biến cơ bản
        self.tts = TTSViettelAI()
        self.audio_queue = Queue()
        self.is_running = True
        
        # Khởi tạo pygame mixer
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        except pygame.error as e:
            self.logger.error(f"Không thể khởi tạo pygame mixer: {e}")
            self.is_running = False
        
        # Đọc file ý nghĩa biển báo
        try:
            self.sign_meanings = pd.read_csv('sign_meanings.csv')
            self.sign_meanings.set_index('Mã biển báo', inplace=True)
        except Exception as e:
            self.logger.error(f"Không thể đọc file sign_meanings.csv: {e}")
            self.sign_meanings = None
        
        # Khởi động thread xử lý âm thanh
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def get_meaning(self, sign_code):
        """Lấy ý nghĩa của biển báo từ mã biển"""
        if self.sign_meanings is None:
            return "Không xác định"
        try:
            return self.sign_meanings.loc[sign_code, 'Ý nghĩa']
        except KeyError:
            return "Không xác định"

    def add_to_queue(self, sign_code):
        """Thêm biển báo vào hàng đợi phát âm thanh"""
        if not self.is_running:
            return

        meaning = self.get_meaning(sign_code)
        self.audio_queue.put(f"Phát hiện biển báo {meaning}")

    def _audio_worker(self):
        """Thread worker xử lý việc phát âm thanh"""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    text = self.audio_queue.get()
                    self.logger.info(f"Đang phát âm thanh: {text}")
                    
                    # Gọi API TTS
                    response = self.tts.handle(text)
                    
                    if response.status == 200 and response.audio_file:
                        audio_data = io.BytesIO(response.audio_file)
                        try:
                            pygame.mixer.music.load(audio_data)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                pygame.time.Clock().tick(10)
                        except pygame.error as e:
                            self.logger.error(f"Lỗi khi phát âm thanh: {e}")
                    else:
                        self.logger.warning(f"Lỗi API TTS, mã lỗi: {response.status}")
                    
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Lỗi trong audio worker: {e}")

    def stop(self):
        """Dừng AudioManager và giải phóng tài nguyên"""
        self.is_running = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        try:
            pygame.mixer.quit()
        except:
            pass