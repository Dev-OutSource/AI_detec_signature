from pydantic import BaseModel
from typing import List, Optional

# class SignData(BaseModel):
#     box: tuple[int, int, int, int]
#     predictions: list
#     time: int
#     announced: bool
#     last_update: float

class TrafficDetection(BaseModel):
    bbox: tuple[int, int, int, int]
    confidence: float
    traffic: str
    traffic_confidence: float
    tracked_time: int

class ProcessingImageResponse(BaseModel):
    success: bool
    message: str
    num_traffics: int
    traffics: List[TrafficDetection]
    tracked_signs: Optional[list] = None
    processed_image: Optional[str] = None