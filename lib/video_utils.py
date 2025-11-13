import cv2
import numpy as np
from typing import Optional, Tuple, Union
import PIL.Image as pil

class VideoSource:
    def __init__(self, source: Union[int, str], width: int = 640, height: int = 192):
        """Initialize video source (webcam or video file)
        
        Args:
            source: Camera index (int) or video file path (str)
            width: Desired frame width
            height: Desired frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source {self.source}")
        
        # Set resolution if using webcam
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        success, frame = self.cap.read()
        if not success:
            return False, None
            
        return True, frame

def frame_to_tensor(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB PIL Image and resize
    
    Args:
        frame: OpenCV BGR frame
        target_size: Desired (width, height)
        
    Returns:
        Resized RGB PIL Image
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = pil.fromarray(rgb)
    
    # Resize
    if img.size != target_size:
        img = img.resize(target_size, pil.LANCZOS)
        
    return img