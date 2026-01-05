"""
Pose Detector Module
Uses MediaPipe to detect body pose landmarks in real-time.
"""

import cv2
import mediapipe as mp


class PoseDetector:
    """
    A wrapper class for MediaPipe Pose detection.
    Provides easy-to-use methods for pose detection and landmark extraction.
    """
    
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the PoseDetector with MediaPipe Pose.
        
        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Complexity of pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.results = None
        self.landmarks = None
    
    def find_pose(self, img, draw=True):
        """
        Detect pose in the given image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw landmarks on the image
            
        Returns:
            Image with or without pose landmarks drawn
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return img
    
    def get_landmarks(self, img):
        """
        Get all pose landmarks as a list of (x, y, z, visibility) tuples.
        
        Args:
            img: Input image (used to get dimensions)
            
        Returns:
            List of landmarks with pixel coordinates, or empty list if no pose detected
        """
        landmarks_list = []
        
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append({
                    'id': id,
                    'x': cx,
                    'y': cy,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
        
        return landmarks_list
    
    def get_landmark_position(self, img, landmark_id):
        """
        Get the position of a specific landmark.
        
        Args:
            img: Input image (used to get dimensions)
            landmark_id: ID of the landmark (0-32)
            
        Returns:
            Tuple (x, y) of landmark position, or None if not found
        """
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            lm = self.results.pose_landmarks.landmark[landmark_id]
            return int(lm.x * w), int(lm.y * h)
        return None
    
    def is_pose_detected(self):
        """
        Check if a pose was detected in the last processed frame.
        
        Returns:
            Boolean indicating if pose is detected
        """
        return self.results is not None and self.results.pose_landmarks is not None


# MediaPipe Pose Landmark IDs for reference
class PoseLandmark:
    """
    Constants for MediaPipe Pose landmark IDs.
    Use these to access specific body parts.
    """
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
