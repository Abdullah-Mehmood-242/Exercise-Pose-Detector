"""
Exercise Detector Module
Detects different types of exercises based on body pose landmarks.
"""

from enum import Enum
from pose_detector import PoseLandmark
from angle_calculator import calculate_body_angle


class ExerciseType(Enum):
    """Supported exercise types."""
    BICEP_CURL = "Bicep Curl"
    SQUAT = "Squat"
    PUSH_UP = "Push-up"
    SHOULDER_PRESS = "Shoulder Press"


class ExerciseDetector:
    """
    Detects and analyzes different exercises based on pose landmarks.
    """
    
    def __init__(self):
        """Initialize the exercise detector."""
        self.current_exercise = ExerciseType.BICEP_CURL
        
        # Angle thresholds for each exercise
        self.thresholds = {
            ExerciseType.BICEP_CURL: {
                'up_angle': 50,      # Arm bent (up position)
                'down_angle': 140,   # Arm extended (down position)
            },
            ExerciseType.SQUAT: {
                'up_angle': 160,     # Standing (up position)
                'down_angle': 90,    # Squatting (down position)
            },
            ExerciseType.PUSH_UP: {
                'up_angle': 160,     # Arms extended (up position)
                'down_angle': 90,    # Arms bent (down position)
            },
            ExerciseType.SHOULDER_PRESS: {
                'up_angle': 160,     # Arms extended above head
                'down_angle': 70,    # Arms at shoulder level
            }
        }
    
    def set_exercise(self, exercise_type):
        """
        Set the current exercise type.
        
        Args:
            exercise_type: ExerciseType enum value
        """
        self.current_exercise = exercise_type
    
    def get_exercise_angle(self, landmarks, side='right'):
        """
        Get the relevant angle for the current exercise.
        
        Args:
            landmarks: List of pose landmarks
            side: 'left' or 'right' side of body
            
        Returns:
            Angle in degrees, or None if landmarks not found
        """
        if not landmarks:
            return None
        
        if self.current_exercise == ExerciseType.BICEP_CURL:
            return self._get_arm_angle(landmarks, side)
        
        elif self.current_exercise == ExerciseType.SQUAT:
            return self._get_leg_angle(landmarks, side)
        
        elif self.current_exercise == ExerciseType.PUSH_UP:
            return self._get_arm_angle(landmarks, side)
        
        elif self.current_exercise == ExerciseType.SHOULDER_PRESS:
            return self._get_shoulder_angle(landmarks, side)
        
        return None
    
    def _get_arm_angle(self, landmarks, side='right'):
        """Calculate the elbow angle (shoulder-elbow-wrist)."""
        if side == 'right':
            shoulder = PoseLandmark.RIGHT_SHOULDER
            elbow = PoseLandmark.RIGHT_ELBOW
            wrist = PoseLandmark.RIGHT_WRIST
        else:
            shoulder = PoseLandmark.LEFT_SHOULDER
            elbow = PoseLandmark.LEFT_ELBOW
            wrist = PoseLandmark.LEFT_WRIST
        
        return calculate_body_angle(landmarks, shoulder, elbow, wrist)
    
    def _get_leg_angle(self, landmarks, side='right'):
        """Calculate the knee angle (hip-knee-ankle)."""
        if side == 'right':
            hip = PoseLandmark.RIGHT_HIP
            knee = PoseLandmark.RIGHT_KNEE
            ankle = PoseLandmark.RIGHT_ANKLE
        else:
            hip = PoseLandmark.LEFT_HIP
            knee = PoseLandmark.LEFT_KNEE
            ankle = PoseLandmark.LEFT_ANKLE
        
        return calculate_body_angle(landmarks, hip, knee, ankle)
    
    def _get_shoulder_angle(self, landmarks, side='right'):
        """Calculate the shoulder angle (hip-shoulder-elbow)."""
        if side == 'right':
            hip = PoseLandmark.RIGHT_HIP
            shoulder = PoseLandmark.RIGHT_SHOULDER
            elbow = PoseLandmark.RIGHT_ELBOW
        else:
            hip = PoseLandmark.LEFT_HIP
            shoulder = PoseLandmark.LEFT_SHOULDER
            elbow = PoseLandmark.LEFT_ELBOW
        
        return calculate_body_angle(landmarks, hip, shoulder, elbow)
    
    def get_stage(self, angle):
        """
        Determine the stage of the exercise based on angle.
        
        Args:
            angle: Current angle in degrees
            
        Returns:
            String 'up', 'down', or 'transition'
        """
        if angle is None:
            return 'unknown'
        
        thresholds = self.thresholds[self.current_exercise]
        
        # For squats and push-ups, lower angle means "down"
        if self.current_exercise in [ExerciseType.SQUAT, ExerciseType.PUSH_UP]:
            if angle <= thresholds['down_angle']:
                return 'down'
            elif angle >= thresholds['up_angle']:
                return 'up'
            else:
                return 'transition'
        
        # For bicep curls, lower angle means "up" (arm bent)
        elif self.current_exercise == ExerciseType.BICEP_CURL:
            if angle <= thresholds['up_angle']:
                return 'up'
            elif angle >= thresholds['down_angle']:
                return 'down'
            else:
                return 'transition'
        
        # For shoulder press, higher angle means "up" (arms extended)
        elif self.current_exercise == ExerciseType.SHOULDER_PRESS:
            if angle >= thresholds['up_angle']:
                return 'up'
            elif angle <= thresholds['down_angle']:
                return 'down'
            else:
                return 'transition'
        
        return 'unknown'
    
    def get_exercise_info(self):
        """
        Get information about the current exercise.
        
        Returns:
            Dictionary with exercise details
        """
        info = {
            ExerciseType.BICEP_CURL: {
                'name': 'Bicep Curl',
                'description': 'Curl your arm up, keeping elbow stationary',
                'target_muscles': 'Biceps',
                'key_points': ['Keep elbow close to body', 'Control the movement']
            },
            ExerciseType.SQUAT: {
                'name': 'Squat',
                'description': 'Lower your body by bending knees',
                'target_muscles': 'Quadriceps, Glutes',
                'key_points': ['Keep back straight', 'Knees over toes']
            },
            ExerciseType.PUSH_UP: {
                'name': 'Push-up',
                'description': 'Lower and push your body using arms',
                'target_muscles': 'Chest, Triceps',
                'key_points': ['Keep body straight', 'Full range of motion']
            },
            ExerciseType.SHOULDER_PRESS: {
                'name': 'Shoulder Press',
                'description': 'Press arms overhead from shoulder level',
                'target_muscles': 'Shoulders, Triceps',
                'key_points': ['Keep core tight', 'Full extension at top']
            }
        }
        
        return info.get(self.current_exercise, {})
