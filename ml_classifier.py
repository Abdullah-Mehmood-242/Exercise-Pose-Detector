"""
ML Classifier Module
Machine Learning based exercise classification and form quality scoring.

This module demonstrates custom ML work for the Computer Vision project:
1. Feature extraction from pose landmarks
2. Exercise type classification using Random Forest
3. Form quality scoring using trained models
"""

import numpy as np
import os
import json
from datetime import datetime
from pose_detector import PoseLandmark
from angle_calculator import (
    calculate_body_angle,
    calculate_distance,
    get_landmark_coords,
    calculate_vertical_angle,
    get_midpoint
)

# Try to import sklearn, provide fallback message if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. ML features will use rule-based fallback.")


class FeatureExtractor:
    """
    Extracts meaningful features from pose landmarks for ML classification.
    
    Features include:
    - Joint angles (elbow, knee, shoulder, hip)
    - Body proportions and ratios
    - Landmark positions (normalized)
    - Body orientation metrics
    """
    
    # Feature names for interpretability
    FEATURE_NAMES = [
        'right_elbow_angle',
        'left_elbow_angle',
        'right_knee_angle',
        'left_knee_angle',
        'right_shoulder_angle',
        'left_shoulder_angle',
        'right_hip_angle',
        'left_hip_angle',
        'torso_vertical_angle',
        'shoulder_hip_ratio',
        'arm_body_ratio',
        'knee_hip_vertical_diff',
        'wrist_shoulder_vertical_diff',
        'body_symmetry_score',
        'pose_compactness'
    ]
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.num_features = len(self.FEATURE_NAMES)
    
    def extract_features(self, landmarks):
        """
        Extract feature vector from pose landmarks.
        
        Args:
            landmarks: List of pose landmarks from PoseDetector
            
        Returns:
            numpy array of features, or None if extraction fails
        """
        if not landmarks or len(landmarks) < 33:
            return None
        
        features = []
        
        # 1. Joint Angles (8 features)
        # Right elbow angle (shoulder-elbow-wrist)
        right_elbow = calculate_body_angle(
            landmarks,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_WRIST
        )
        features.append(right_elbow if right_elbow else 0)
        
        # Left elbow angle
        left_elbow = calculate_body_angle(
            landmarks,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_WRIST
        )
        features.append(left_elbow if left_elbow else 0)
        
        # Right knee angle (hip-knee-ankle)
        right_knee = calculate_body_angle(
            landmarks,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_ANKLE
        )
        features.append(right_knee if right_knee else 0)
        
        # Left knee angle
        left_knee = calculate_body_angle(
            landmarks,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_ANKLE
        )
        features.append(left_knee if left_knee else 0)
        
        # Right shoulder angle (hip-shoulder-elbow)
        right_shoulder = calculate_body_angle(
            landmarks,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_ELBOW
        )
        features.append(right_shoulder if right_shoulder else 0)
        
        # Left shoulder angle
        left_shoulder = calculate_body_angle(
            landmarks,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_ELBOW
        )
        features.append(left_shoulder if left_shoulder else 0)
        
        # Right hip angle (shoulder-hip-knee)
        right_hip = calculate_body_angle(
            landmarks,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE
        )
        features.append(right_hip if right_hip else 0)
        
        # Left hip angle
        left_hip = calculate_body_angle(
            landmarks,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE
        )
        features.append(left_hip if left_hip else 0)
        
        # 2. Body Orientation (1 feature)
        left_shoulder_pos = get_landmark_coords(landmarks, PoseLandmark.LEFT_SHOULDER)
        right_shoulder_pos = get_landmark_coords(landmarks, PoseLandmark.RIGHT_SHOULDER)
        left_hip_pos = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        right_hip_pos = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if all([left_shoulder_pos, right_shoulder_pos, left_hip_pos, right_hip_pos]):
            shoulder_mid = get_midpoint(left_shoulder_pos, right_shoulder_pos)
            hip_mid = get_midpoint(left_hip_pos, right_hip_pos)
            torso_angle = calculate_vertical_angle(hip_mid, shoulder_mid)
            features.append(torso_angle)
        else:
            features.append(0)
        
        # 3. Body Proportions (2 features)
        if all([left_shoulder_pos, right_shoulder_pos, left_hip_pos, right_hip_pos]):
            shoulder_width = calculate_distance(left_shoulder_pos, right_shoulder_pos)
            hip_width = calculate_distance(left_hip_pos, right_hip_pos)
            shoulder_hip_ratio = shoulder_width / max(hip_width, 1)
            features.append(shoulder_hip_ratio)
        else:
            features.append(1.0)
        
        # Arm to body ratio
        right_wrist = get_landmark_coords(landmarks, PoseLandmark.RIGHT_WRIST)
        if right_shoulder_pos and right_wrist and right_hip_pos:
            arm_length = calculate_distance(right_shoulder_pos, right_wrist)
            torso_length = calculate_distance(right_shoulder_pos, right_hip_pos)
            arm_body_ratio = arm_length / max(torso_length, 1)
            features.append(arm_body_ratio)
        else:
            features.append(1.0)
        
        # 4. Vertical Differences (2 features)
        right_knee_pos = get_landmark_coords(landmarks, PoseLandmark.RIGHT_KNEE)
        if right_knee_pos and right_hip_pos:
            knee_hip_diff = (right_knee_pos[1] - right_hip_pos[1]) / 480  # Normalized
            features.append(knee_hip_diff)
        else:
            features.append(0)
        
        if right_wrist and right_shoulder_pos:
            wrist_shoulder_diff = (right_wrist[1] - right_shoulder_pos[1]) / 480
            features.append(wrist_shoulder_diff)
        else:
            features.append(0)
        
        # 5. Symmetry Score (1 feature)
        symmetry = self._calculate_symmetry(landmarks)
        features.append(symmetry)
        
        # 6. Pose Compactness (1 feature)
        compactness = self._calculate_compactness(landmarks)
        features.append(compactness)
        
        return np.array(features)
    
    def _calculate_symmetry(self, landmarks):
        """Calculate body symmetry score (0-1, higher is more symmetric)."""
        pairs = [
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW),
            (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST),
            (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
            (PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE),
            (PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE),
        ]
        
        # Get body center (midpoint of hips)
        left_hip = get_landmark_coords(landmarks, PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP)
        
        if not left_hip or not right_hip:
            return 0.5
        
        center_x = (left_hip[0] + right_hip[0]) / 2
        
        symmetry_scores = []
        for left_id, right_id in pairs:
            left_pos = get_landmark_coords(landmarks, left_id)
            right_pos = get_landmark_coords(landmarks, right_id)
            
            if left_pos and right_pos:
                left_dist = abs(left_pos[0] - center_x)
                right_dist = abs(right_pos[0] - center_x)
                
                if max(left_dist, right_dist) > 0:
                    pair_symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)
                    symmetry_scores.append(pair_symmetry)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.5
    
    def _calculate_compactness(self, landmarks):
        """Calculate pose compactness (how spread out the body is)."""
        key_points = [
            PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.NOSE
        ]
        
        positions = []
        for point_id in key_points:
            pos = get_landmark_coords(landmarks, point_id)
            if pos:
                positions.append(pos)
        
        if len(positions) < 3:
            return 0.5
        
        # Calculate bounding box area
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # Normalize by frame size (assume 640x480)
        normalized_area = (width * height) / (640 * 480)
        
        # Invert so higher = more compact
        compactness = 1 - min(normalized_area, 1)
        
        return compactness


class ExerciseClassifier:
    """
    ML-based exercise classifier using Random Forest.
    
    Classifies poses into exercise types:
    0 - Bicep Curl
    1 - Squat
    2 - Push-up
    3 - Shoulder Press
    4 - Unknown/Standing
    """
    
    EXERCISE_LABELS = {
        0: 'Bicep Curl',
        1: 'Squat',
        2: 'Push-up',
        3: 'Shoulder Press',
        4: 'Unknown'
    }
    
    def __init__(self, model_path=None):
        """
        Initialize the exercise classifier.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif SKLEARN_AVAILABLE:
            # Initialize with default model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
    
    def train(self, training_data, labels):
        """
        Train the classifier on labeled data.
        
        Args:
            training_data: List of feature vectors
            labels: List of exercise labels (0-4)
        """
        if not SKLEARN_AVAILABLE:
            print("Cannot train: scikit-learn not installed")
            return False
        
        X = np.array(training_data)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Model trained on {len(X)} samples")
        return True
    
    def predict(self, landmarks):
        """
        Predict exercise type from landmarks.
        
        Args:
            landmarks: Pose landmarks from detector
            
        Returns:
            Tuple of (exercise_label, confidence, all_probabilities)
        """
        features = self.feature_extractor.extract_features(landmarks)
        
        if features is None:
            return 'Unknown', 0.0, {}
        
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Use rule-based fallback
            return self._rule_based_predict(features)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        confidence = max(probabilities)
        label = self.EXERCISE_LABELS.get(prediction, 'Unknown')
        
        # Create probability dict
        prob_dict = {
            self.EXERCISE_LABELS[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        
        return label, confidence, prob_dict
    
    def _rule_based_predict(self, features):
        """
        Rule-based fallback when ML model is not trained.
        Uses heuristics based on joint angles.
        """
        right_elbow = features[0]
        right_knee = features[2]
        right_shoulder = features[4]
        torso_angle = features[8]
        wrist_shoulder_diff = features[12]  # wrist_shoulder_vertical_diff
        
        probabilities = {'Bicep Curl': 0.1, 'Squat': 0.1, 
                        'Push-up': 0.1, 'Shoulder Press': 0.1, 'Unknown': 0.6}
        
        # Bicep curl: bent elbow, standing upright
        if right_elbow < 90 and torso_angle < 20 and right_knee > 150:
            probabilities['Bicep Curl'] = 0.7
            probabilities['Unknown'] = 0.1
        
        # Squat: bent knees, upright torso
        elif right_knee < 120 and torso_angle < 45:
            probabilities['Squat'] = 0.7
            probabilities['Unknown'] = 0.1
        
        # Push-up: horizontal body, bent elbows
        elif torso_angle > 60 and right_elbow < 120:
            probabilities['Push-up'] = 0.7
            probabilities['Unknown'] = 0.1
        
        # Shoulder press: arms above shoulders
        elif wrist_shoulder_diff < -0.1 and right_shoulder > 90:
            probabilities['Shoulder Press'] = 0.7
            probabilities['Unknown'] = 0.1
        
        best_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_label]
        
        return best_label, confidence, probabilities
    
    def save_model(self, path):
        """Save trained model to file."""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return False
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
        return True
    
    def load_model(self, path):
        """Load model from file."""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = True
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


class FormQualityScorer:
    """
    ML-based form quality scoring.
    
    Scores exercise form on a scale of 0-100 based on:
    - Body symmetry
    - Joint angles compared to ideal
    - Movement consistency
    """
    
    # Ideal angles for each exercise (approximate)
    IDEAL_ANGLES = {
        'Bicep Curl': {
            'up_elbow': 40,
            'down_elbow': 160,
            'torso': 0,
        },
        'Squat': {
            'down_knee': 90,
            'up_knee': 170,
            'torso': 30,
        },
        'Push-up': {
            'down_elbow': 90,
            'up_elbow': 170,
            'body_line': 180,
        },
        'Shoulder Press': {
            'up_shoulder': 170,
            'down_shoulder': 70,
            'torso': 0,
        }
    }
    
    def __init__(self):
        """Initialize the form scorer."""
        self.feature_extractor = FeatureExtractor()
        self.score_history = []
        self.max_history = 30  # Keep last 30 scores for smoothing
    
    def score_form(self, landmarks, exercise_type='Bicep Curl', stage='unknown'):
        """
        Score the current form quality.
        
        Args:
            landmarks: Pose landmarks
            exercise_type: Current exercise name
            stage: Current exercise stage ('up', 'down', etc.)
            
        Returns:
            FormScore object with score and breakdown
        """
        features = self.feature_extractor.extract_features(landmarks)
        
        if features is None:
            return FormScore(0, "No pose detected", {})
        
        scores = {}
        
        # 1. Symmetry Score (25 points)
        symmetry = features[13]  # symmetry_score feature
        scores['symmetry'] = int(symmetry * 25)
        
        # 2. Angle Accuracy Score (50 points)
        angle_score = self._score_angles(features, exercise_type, stage)
        scores['angle_accuracy'] = int(angle_score * 50)
        
        # 3. Stability Score (25 points)
        stability = self._score_stability(features)
        scores['stability'] = int(stability * 25)
        
        # Calculate total
        total_score = sum(scores.values())
        
        # Smooth score using history
        self.score_history.append(total_score)
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)
        
        smoothed_score = int(np.mean(self.score_history))
        
        # Generate feedback message
        feedback = self._generate_feedback(smoothed_score, scores)
        
        return FormScore(smoothed_score, feedback, scores)
    
    def _score_angles(self, features, exercise_type, stage):
        """Score how close angles are to ideal."""
        if exercise_type not in self.IDEAL_ANGLES:
            return 0.5
        
        ideals = self.IDEAL_ANGLES[exercise_type]
        
        # Get relevant angle from features
        if exercise_type == 'Bicep Curl':
            current_angle = features[0]  # right_elbow_angle
            if stage == 'up':
                ideal = ideals['up_elbow']
            else:
                ideal = ideals['down_elbow']
        
        elif exercise_type == 'Squat':
            current_angle = features[2]  # right_knee_angle
            if stage == 'down':
                ideal = ideals['down_knee']
            else:
                ideal = ideals['up_knee']
        
        elif exercise_type == 'Push-up':
            current_angle = features[0]  # right_elbow_angle
            if stage == 'down':
                ideal = ideals['down_elbow']
            else:
                ideal = ideals['up_elbow']
        
        elif exercise_type == 'Shoulder Press':
            current_angle = features[4]  # right_shoulder_angle
            if stage == 'up':
                ideal = ideals['up_shoulder']
            else:
                ideal = ideals['down_shoulder']
        else:
            return 0.5
        
        # Calculate score based on deviation from ideal
        deviation = abs(current_angle - ideal)
        max_deviation = 60  # Maximum expected deviation
        
        score = max(0, 1 - (deviation / max_deviation))
        return score
    
    def _score_stability(self, features):
        """Score body stability based on torso angle."""
        torso_angle = features[8]
        
        # Lower torso angle = more stable (for standing exercises)
        if torso_angle < 10:
            return 1.0
        elif torso_angle < 20:
            return 0.8
        elif torso_angle < 30:
            return 0.6
        elif torso_angle < 45:
            return 0.4
        else:
            return 0.2
    
    def _generate_feedback(self, score, breakdown):
        """Generate human-readable feedback based on score."""
        if score >= 90:
            return "Excellent form! Perfect execution!"
        elif score >= 75:
            return "Great form! Minor improvements possible."
        elif score >= 60:
            return "Good form. Focus on technique."
        elif score >= 40:
            return "Fair form. Check your posture."
        else:
            return "Needs improvement. Review exercise guide."
    
    def reset(self):
        """Reset score history."""
        self.score_history = []


class FormScore:
    """Data class for form quality score."""
    
    def __init__(self, score, feedback, breakdown):
        """
        Initialize form score.
        
        Args:
            score: Overall score (0-100)
            feedback: Feedback message
            breakdown: Dict of component scores
        """
        self.score = score
        self.feedback = feedback
        self.breakdown = breakdown
    
    def get_grade(self):
        """Get letter grade based on score."""
        if self.score >= 90:
            return 'A'
        elif self.score >= 80:
            return 'B'
        elif self.score >= 70:
            return 'C'
        elif self.score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_color(self):
        """Get BGR color based on score."""
        if self.score >= 80:
            return (0, 255, 0)  # Green
        elif self.score >= 60:
            return (0, 255, 255)  # Yellow
        elif self.score >= 40:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red


class TrainingDataCollector:
    """
    Utility class for collecting training data for the ML classifier.
    
    Saves pose features with labels for later training.
    """
    
    def __init__(self, save_dir='training_data'):
        """Initialize the data collector."""
        self.save_dir = save_dir
        self.feature_extractor = FeatureExtractor()
        self.collected_data = []
        
        # Create save directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def collect_sample(self, landmarks, label):
        """
        Collect a training sample.
        
        Args:
            landmarks: Pose landmarks
            label: Exercise label (0-4)
        """
        features = self.feature_extractor.extract_features(landmarks)
        
        if features is not None:
            self.collected_data.append({
                'features': features.tolist(),
                'label': label,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def save_data(self, filename=None):
        """Save collected data to file."""
        if not filename:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        
        print(f"Saved {len(self.collected_data)} samples to {filepath}")
        return filepath
    
    def load_data(self, filepath):
        """Load training data from file."""
        with open(filepath, 'r') as f:
            self.collected_data = json.load(f)
        
        print(f"Loaded {len(self.collected_data)} samples from {filepath}")
        return self.collected_data
    
    def get_training_arrays(self):
        """Get training data as numpy arrays."""
        if not self.collected_data:
            return None, None
        
        X = np.array([d['features'] for d in self.collected_data])
        y = np.array([d['label'] for d in self.collected_data])
        
        return X, y
    
    def clear(self):
        """Clear collected data."""
        self.collected_data = []


# Demo usage
if __name__ == "__main__":
    print("ML Classifier Module")
    print("=" * 50)
    
    # Test feature extractor
    extractor = FeatureExtractor()
    print(f"Feature names: {extractor.FEATURE_NAMES}")
    print(f"Number of features: {extractor.num_features}")
    
    # Test classifier initialization
    classifier = ExerciseClassifier()
    print(f"\nClassifier initialized. sklearn available: {SKLEARN_AVAILABLE}")
    
    # Test form scorer
    scorer = FormQualityScorer()
    print(f"Form scorer initialized")
    
    print("\nModule ready for integration!")
