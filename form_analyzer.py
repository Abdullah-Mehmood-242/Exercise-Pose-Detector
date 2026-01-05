"""
Form Analyzer Module
Analyzes exercise form and provides real-time feedback.
"""

from pose_detector import PoseLandmark
from angle_calculator import (
    calculate_body_angle, 
    get_landmark_coords, 
    calculate_vertical_angle,
    get_midpoint
)
from exercise_detector import ExerciseType


class FormFeedback:
    """Represents feedback about exercise form."""
    
    GOOD = 'good'
    WARNING = 'warning'
    BAD = 'bad'
    
    def __init__(self, status, message, detail=None):
        self.status = status
        self.message = message
        self.detail = detail
    
    def get_color(self):
        """Get BGR color based on feedback status."""
        colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow
            'bad': (0, 0, 255)         # Red
        }
        return colors.get(self.status, (255, 255, 255))
    
    def get_emoji(self):
        """Get emoji based on feedback status."""
        emojis = {
            'good': '✓',
            'warning': '⚠',
            'bad': '✗'
        }
        return emojis.get(self.status, '')


class FormAnalyzer:
    """
    Analyzes exercise form and provides corrective feedback.
    """
    
    def __init__(self):
        """Initialize the form analyzer."""
        self.current_exercise = ExerciseType.BICEP_CURL
        self.feedback_history = []
    
    def set_exercise(self, exercise_type):
        """Set the current exercise type."""
        self.current_exercise = exercise_type
        self.feedback_history = []
    
    def analyze(self, landmarks):
        """
        Analyze the current pose for form issues.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            FormFeedback object with analysis results
        """
        if not landmarks:
            return FormFeedback(FormFeedback.WARNING, "No pose detected", "Stand in front of camera")
        
        if self.current_exercise == ExerciseType.BICEP_CURL:
            return self._analyze_bicep_curl(landmarks)
        elif self.current_exercise == ExerciseType.SQUAT:
            return self._analyze_squat(landmarks)
        elif self.current_exercise == ExerciseType.PUSH_UP:
            return self._analyze_pushup(landmarks)
        elif self.current_exercise == ExerciseType.SHOULDER_PRESS:
            return self._analyze_shoulder_press(landmarks)
        
        return FormFeedback(FormFeedback.GOOD, "Good form!")
    
    def _analyze_bicep_curl(self, landmarks):
        """Analyze bicep curl form."""
        feedback_items = []
        
        # Check if elbow is stable (not moving forward/back)
        right_shoulder = get_landmark_coords(landmarks, PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_landmark_coords(landmarks, PoseLandmark.RIGHT_ELBOW)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if right_shoulder and right_elbow and right_hip:
            # Check elbow position relative to shoulder
            elbow_forward = right_elbow[0] - right_shoulder[0]
            
            if abs(elbow_forward) > 50:
                feedback_items.append("Keep elbow close to body")
        
        # Check for body sway
        left_shoulder = get_landmark_coords(landmarks, PoseLandmark.LEFT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
            hip_mid = get_midpoint(left_hip, right_hip)
            
            torso_angle = calculate_vertical_angle(hip_mid, shoulder_mid)
            
            if torso_angle > 15:
                feedback_items.append("Keep your back straight")
        
        if feedback_items:
            return FormFeedback(FormFeedback.WARNING, feedback_items[0])
        
        return FormFeedback(FormFeedback.GOOD, "Great form! Keep it up!")
    
    def _analyze_squat(self, landmarks):
        """Analyze squat form."""
        feedback_items = []
        
        # Check knee alignment
        right_knee = get_landmark_coords(landmarks, PoseLandmark.RIGHT_KNEE)
        right_ankle = get_landmark_coords(landmarks, PoseLandmark.RIGHT_ANKLE)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if right_knee and right_ankle:
            # Check if knees go past toes (in side view)
            knee_forward = right_knee[0] - right_ankle[0]
            
            # This check works best from side angle
            if abs(knee_forward) > 100:
                feedback_items.append("Don't let knees go too far past toes")
        
        # Check back angle
        left_shoulder = get_landmark_coords(landmarks, PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
            hip_mid = get_midpoint(left_hip, right_hip)
            
            torso_angle = calculate_vertical_angle(hip_mid, shoulder_mid)
            
            # Some forward lean is okay, but too much is bad
            if torso_angle > 45:
                feedback_items.append("Keep chest up, don't lean too far forward")
        
        # Check squat depth
        if right_hip and right_knee:
            knee_angle = calculate_body_angle(landmarks, 
                                              PoseLandmark.RIGHT_HIP, 
                                              PoseLandmark.RIGHT_KNEE, 
                                              PoseLandmark.RIGHT_ANKLE)
            if knee_angle and knee_angle > 120 and knee_angle < 160:
                feedback_items.append("Try to squat deeper for full range")
        
        if feedback_items:
            return FormFeedback(FormFeedback.WARNING, feedback_items[0])
        
        return FormFeedback(FormFeedback.GOOD, "Great squat form!")
    
    def _analyze_pushup(self, landmarks):
        """Analyze push-up form."""
        feedback_items = []
        
        # Check body alignment (should be straight line)
        left_shoulder = get_landmark_coords(landmarks, PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        left_ankle = get_landmark_coords(landmarks, PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coords(landmarks, PoseLandmark.RIGHT_ANKLE)
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]):
            shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
            hip_mid = get_midpoint(left_hip, right_hip)
            ankle_mid = get_midpoint(left_ankle, right_ankle)
            
            # Check if hips are sagging or piking
            # Calculate angle at hips
            body_angle = calculate_body_angle(landmarks,
                                              PoseLandmark.LEFT_SHOULDER,
                                              PoseLandmark.LEFT_HIP,
                                              PoseLandmark.LEFT_ANKLE)
            
            if body_angle and body_angle < 160:
                if hip_mid[1] > shoulder_mid[1] and hip_mid[1] > ankle_mid[1]:
                    feedback_items.append("Keep hips up, don't let them sag")
                else:
                    feedback_items.append("Don't pike hips too high")
        
        # Check hand position
        left_wrist = get_landmark_coords(landmarks, PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_coords(landmarks, PoseLandmark.RIGHT_WRIST)
        
        if left_wrist and right_wrist and left_shoulder and right_shoulder:
            # Hands should be roughly shoulder-width apart
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            hand_width = abs(right_wrist[0] - left_wrist[0])
            
            if hand_width < shoulder_width * 0.7:
                feedback_items.append("Widen hand placement")
            elif hand_width > shoulder_width * 1.5:
                feedback_items.append("Narrow hand placement")
        
        if feedback_items:
            return FormFeedback(FormFeedback.WARNING, feedback_items[0])
        
        return FormFeedback(FormFeedback.GOOD, "Great push-up form!")
    
    def _analyze_shoulder_press(self, landmarks):
        """Analyze shoulder press form."""
        feedback_items = []
        
        # Check for back arch
        left_shoulder = get_landmark_coords(landmarks, PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
            hip_mid = get_midpoint(left_hip, right_hip)
            
            # Check vertical alignment
            if shoulder_mid[0] - hip_mid[0] > 30:
                feedback_items.append("Don't arch your back")
        
        # Check for symmetric arm movement
        left_elbow = get_landmark_coords(landmarks, PoseLandmark.LEFT_ELBOW)
        right_elbow = get_landmark_coords(landmarks, PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_coords(landmarks, PoseLandmark.RIGHT_WRIST)
        
        if left_wrist and right_wrist:
            # Check if arms are at similar heights
            height_diff = abs(left_wrist[1] - right_wrist[1])
            if height_diff > 50:
                feedback_items.append("Keep arms at equal height")
        
        # Check elbow flare
        if left_elbow and right_elbow and left_shoulder and right_shoulder:
            left_flare = abs(left_elbow[0] - left_shoulder[0])
            right_flare = abs(right_elbow[0] - right_shoulder[0])
            
            # Elbows shouldn't flare out too much
            if left_flare > 80 or right_flare > 80:
                feedback_items.append("Keep elbows slightly in front")
        
        if feedback_items:
            return FormFeedback(FormFeedback.WARNING, feedback_items[0])
        
        return FormFeedback(FormFeedback.GOOD, "Great shoulder press form!")
