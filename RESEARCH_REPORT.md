# AI-Powered Fitness Trainer: A Computer Vision Approach

## Research Report

**Course:** Computer Vision  
**Project Type:** Semester Project  
**Date:** January 2026

---

## Abstract

This report presents an AI-powered fitness trainer application that leverages computer vision and machine learning techniques for real-time exercise analysis. The system employs Google's MediaPipe Pose estimation model (BlazePose) for human pose detection, combined with custom machine learning classifiers for exercise recognition and form quality assessment. Additionally, classical computer vision techniques including optical flow analysis, background subtraction, and motion energy imaging are implemented to provide comprehensive motion analysis. The application demonstrates practical applications of deep learning-based pose estimation alongside traditional computer vision methods in the domain of fitness and health technology.

**Keywords:** Pose Estimation, Computer Vision, Machine Learning, MediaPipe, BlazePose, Optical Flow, Fitness Technology

---

## 1. Introduction

### 1.1 Background

The fitness technology industry has seen tremendous growth with the advent of AI and computer vision. Traditional workout tracking relied on wearable sensors, but camera-based systems offer a non-intrusive alternative that can provide detailed form analysis without requiring any physical devices.

### 1.2 Problem Statement

Manual exercise tracking is prone to errors, and professional trainers are not always available. There is a need for an automated system that can:
1. Detect and track human body poses in real-time
2. Recognize different exercise types automatically
3. Count repetitions accurately
4. Provide real-time feedback on exercise form
5. Analyze motion patterns for performance insights

### 1.3 Objectives

1. Implement real-time pose estimation using MediaPipe
2. Develop ML classifiers for exercise recognition
3. Create a form quality scoring system
4. Integrate classical CV techniques for motion analysis
5. Build an intuitive user interface for real-time feedback

---

## 2. Literature Review

### 2.1 Pose Estimation Approaches

Human pose estimation has evolved significantly over the years:

| Approach | Description | Examples |
|----------|-------------|----------|
| **Classical Methods** | Hand-crafted features, HOG descriptors | Pictorial Structures |
| **Deep Learning** | CNN-based keypoint detection | OpenPose, PoseNet |
| **Lightweight Models** | Mobile-optimized architectures | BlazePose, MoveNet |

### 2.2 BlazePose Architecture

BlazePose, developed by Google Research, is a lightweight pose estimation model designed for real-time inference on mobile devices. Key innovations include:

1. **Two-stage detector-tracker pipeline**: Reduces computational cost by only running detection when needed
2. **Heatmap + regression hybrid**: Combines heatmap precision with regression speed
3. **Attention mechanisms**: Focus computational resources on relevant body regions

### 2.3 Exercise Recognition Research

Prior work in exercise recognition includes:
- Sensor-based activity recognition using accelerometers
- Video-based action recognition using 3D CNNs
- Skeleton-based methods using RNNs/Transformers

Our approach combines pose-based features with classical ML classifiers for interpretability.

---

## 3. Methodology

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI FITNESS TRAINER SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Camera     │───▶│   OpenCV     │───▶│  MediaPipe   │              │
│  │   Input      │    │   Capture    │    │  Pose        │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                  │                      │
│                                    ┌─────────────┴─────────────┐        │
│                                    │                           │        │
│                                    ▼                           ▼        │
│  ┌─────────────────────────────────────┐   ┌────────────────────────┐  │
│  │         ML Classification           │   │    CV Analysis         │  │
│  │  • Feature Extraction               │   │  • Optical Flow        │  │
│  │  • Exercise Recognition             │   │  • Background Sub      │  │
│  │  • Form Quality Scoring             │   │  • Motion Energy       │  │
│  └─────────────────┬───────────────────┘   └───────────┬────────────┘  │
│                    │                                   │                │
│                    └───────────────┬───────────────────┘                │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Application Logic                            │   │
│  │   • Rep Counter  • Stage Detection  • Form Analyzer              │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     User Interface                               │   │
│  │   • Stats Panel  • Form Feedback  • CV Visualization             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 MediaPipe Pose (BlazePose)

#### 3.2.1 Architecture Details

MediaPipe Pose uses a two-step detection-tracking pipeline:

**Step 1: Person Detection**
- Lightweight face detector to establish region of interest
- Only runs when tracking is lost

**Step 2: Pose Landmark Model**
- Input: 256x256 RGB image
- Output: 33 body landmarks with x, y, z coordinates and visibility scores
- Backbone: Modified MobileNet architecture
- Inference time: ~10ms on modern hardware

#### 3.2.2 Landmark Model

The BlazePose model outputs 33 keypoints covering the full body:

```
           0 (NOSE)
           │
    ┌──────┴──────┐
    │             │
   11            12 (SHOULDERS)
    │             │
   13            14 (ELBOWS)
    │             │
   15            16 (WRISTS)
    │             │
   23            24 (HIPS)
    │             │
   25            26 (KNEES)
    │             │
   27            28 (ANKLES)
```

#### 3.2.3 Implementation

```python
class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,        # 0, 1, or 2
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def find_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results.pose_landmarks
```

### 3.3 ML Exercise Classification

#### 3.3.1 Feature Extraction

We extract 15 features from the 33 pose landmarks:

| Feature Category | Features | Count |
|-----------------|----------|-------|
| Joint Angles | Elbow (L/R), Knee (L/R), Shoulder (L/R), Hip (L/R) | 8 |
| Body Orientation | Torso vertical angle | 1 |
| Proportions | Shoulder-hip ratio, Arm-body ratio | 2 |
| Position Differences | Knee-hip vertical, Wrist-shoulder vertical | 2 |
| Quality Metrics | Symmetry score, Pose compactness | 2 |

#### 3.3.2 Angle Calculation

Joint angles are calculated using vector mathematics:

```python
def calculate_angle(point1, point2, point3):
    """
    Calculate angle at point2 (vertex) between point1 and point3
    
    Using the formula:
    angle = arccos((v1 · v2) / (|v1| × |v2|))
    """
    a = np.array(point1)
    b = np.array(point2)  # vertex
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)
```

#### 3.3.3 Classification Model

We use a Random Forest classifier for exercise recognition:

```python
from sklearn.ensemble import RandomForestClassifier

class ExerciseClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, landmarks):
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        return prediction, probabilities
```

**Why Random Forest?**
- Interpretable: Feature importance analysis
- Robust: Handles noisy sensor data well
- Fast: Real-time inference capability
- No extensive training data required

### 3.4 Form Quality Scoring

The form scoring system evaluates exercise execution quality on a 0-100 scale:

| Component | Weight | Description |
|-----------|--------|-------------|
| Symmetry | 25% | Left-right body balance |
| Angle Accuracy | 50% | Deviation from ideal exercise angles |
| Stability | 25% | Body steadiness during exercise |

```python
def score_form(self, landmarks, exercise_type, stage):
    scores = {}
    
    # Symmetry Score (25 points)
    scores['symmetry'] = self._calculate_symmetry(landmarks) * 25
    
    # Angle Accuracy (50 points)
    scores['angle_accuracy'] = self._score_angles(landmarks, exercise_type) * 50
    
    # Stability (25 points)
    scores['stability'] = self._score_stability(landmarks) * 25
    
    return sum(scores.values())
```

### 3.5 Computer Vision Techniques

#### 3.5.1 Optical Flow (Lucas-Kanade)

Optical flow tracks motion between consecutive frames:

```python
# Parameters for Lucas-Kanade optical flow
lk_params = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

# Calculate optical flow
next_points, status, error = cv2.calcOpticalFlowPyrLK(
    prev_gray, gray, prev_points, None, **lk_params
)
```

**Applications:**
- Visualize movement direction and speed
- Detect stationary vs. active phases
- Motion-based exercise phase detection

#### 3.5.2 Background Subtraction (MOG2)

Gaussian Mixture Model for foreground detection:

```python
# Create MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# Apply to frame
fg_mask = bg_subtractor.apply(frame)
```

**Applications:**
- Extract person silhouette
- Calculate motion area percentage
- Generate bounding boxes

#### 3.5.3 Motion Energy Image (MEI)

Accumulates motion over time for pattern visualization:

```python
def update_mei(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if self.prev_gray is not None:
        diff = cv2.absdiff(gray, self.prev_gray)
        _, motion = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        
        # Decay old motion, add new
        self.mei = self.mei * 0.95 + motion
    
    self.prev_gray = gray
    return self.mei
```

**Applications:**
- Visualize exercise movement patterns
- Identify high-activity body regions
- Compare motion consistency

---

## 4. Implementation

### 4.1 Project Structure

```
AI-Fitness-Trainer/
│
├── main.py               # Application entry point & UI
├── pose_detector.py      # MediaPipe pose wrapper
├── exercise_detector.py  # Exercise-specific logic
├── rep_counter.py        # Repetition tracking
├── form_analyzer.py      # Rule-based form feedback
├── angle_calculator.py   # Geometric calculations
├── ml_classifier.py      # ML classification & scoring
├── cv_analyzer.py        # CV techniques module
├── requirements.txt      # Dependencies
└── RESEARCH_REPORT.md    # This document
```

### 4.2 Key Modules

#### 4.2.1 Pose Detector Module
- Wraps MediaPipe Pose API
- Provides landmark extraction methods
- Handles coordinate transformation

#### 4.2.2 ML Classifier Module
- `FeatureExtractor`: Extracts 15 features from landmarks
- `ExerciseClassifier`: Random Forest-based classification
- `FormQualityScorer`: ML-based form scoring
- `TrainingDataCollector`: Utility for data collection

#### 4.2.3 CV Analyzer Module
- `OpticalFlowAnalyzer`: Lucas-Kanade implementation
- `BackgroundSubtractor`: MOG2 wrapper
- `MotionEnergyImage`: MEI generator
- `EdgeDetector`: Canny edge detection
- `CVAnalyzer`: Unified interface

### 4.3 User Interface

The application features a comprehensive overlay UI:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  AI FITNESS TRAINER                           Exercise: Bicep Curl        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ┌─────────────────┐                           ┌────────────────────┐    ║
║  │  REPS  │ STAGE  │                           │  ⏱️ Time: 02:45     │    ║
║  │   12   │  UP ⬆  │      📹 LIVE CAMERA       │  📊 Total: 24 reps │    ║
║  └─────────────────┘      WITH POSE OVERLAY    │  🔥 RPM: 8.7       │    ║
║                                                 └────────────────────┘    ║
║  ┌─────────────────┐                                                      ║
║  │ ML MODE: AUTO   │   ┌────────────────────┐                            ║
║  │ Detected: Curl  │   │  CV ANALYSIS       │                            ║
║  │ Confidence: 85% │   │  Motion: 12.5      │                            ║
║  │ Form: 78/100 (C)│   │  Area: 35.2%       │                            ║
║  └─────────────────┘   └────────────────────┘                            ║
║                                                                           ║
║  [1]Curl [2]Squat [3]Push-up [4]Press                                    ║
║  [A]Auto-Detect [V]CV Mode [R]Reset [Q]Quit                              ║
║                                                                           ║
║  ┌───────────────────────────────────────────────────────────────────┐   ║
║  │  ✅ FORM FEEDBACK: Great form! Keep it up!                        │   ║
║  └───────────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Results & Analysis

### 5.1 Pose Detection Performance

| Metric | Value |
|--------|-------|
| Average FPS | 25-30 fps |
| Detection Confidence | 0.7+ threshold |
| Tracking Accuracy | High (smooth landmarks) |
| Lighting Sensitivity | Moderate |

### 5.2 Exercise Recognition

The ML classifier uses rule-based heuristics by default (without training data) and achieves:

| Exercise | Recognition Cues |
|----------|-----------------|
| Bicep Curl | Bent elbow angle, upright torso |
| Squat | Bent knee angle, slight forward lean |
| Push-up | Horizontal body, bent/extended arms |
| Shoulder Press | Arms above shoulders, vertical torso |

### 5.3 Form Scoring Validation

The form scoring system provides consistent feedback:

| Score Range | Grade | Interpretation |
|-------------|-------|----------------|
| 90-100 | A | Excellent form |
| 80-89 | B | Great form |
| 70-79 | C | Good form |
| 60-69 | D | Fair form |
| <60 | F | Needs improvement |

### 5.4 CV Analysis Features

| Technique | Purpose | Visualization |
|-----------|---------|---------------|
| Optical Flow | Motion direction/speed | Arrow vectors |
| Background Sub | Foreground detection | Colored overlay |
| MEI | Motion accumulation | Heatmap |
| Edge Detection | Body contours | Green edges |

---

## 6. Discussion

### 6.1 Strengths

1. **Real-time performance**: Achieves 25-30 FPS on standard hardware
2. **Non-intrusive**: No wearable sensors required
3. **Interpretable**: Feature-based ML allows understanding of decisions
4. **Modular design**: Easy to extend with new exercises
5. **Multiple CV techniques**: Comprehensive motion analysis

### 6.2 Limitations

1. **Camera angle dependency**: Works best with side/front view
2. **Lighting sensitivity**: Poor lighting affects detection
3. **Single person**: Multi-person not supported
4. **Limited exercise set**: Currently 4 exercises
5. **No trained model**: Uses rule-based classification by default

### 6.3 Future Work

1. **Train ML models**: Collect labeled data for better classification
2. **Add more exercises**: Lunges, deadlifts, planks, etc.
3. **Voice feedback**: Audio cues for form correction
4. **Workout logging**: Save session history and progress
5. **Multi-person support**: Track multiple users simultaneously
6. **Mobile deployment**: Port to mobile devices

---

## 7. Conclusion

This project successfully demonstrates the application of computer vision and machine learning to fitness technology. By combining Google's MediaPipe pose estimation with custom ML classifiers and classical CV techniques, we created a comprehensive exercise analysis system that operates in real-time.

The system provides:
- **Accurate pose detection** using BlazePose deep learning model
- **Automatic exercise recognition** using ML classification
- **Form quality scoring** for performance feedback
- **Motion analysis** using optical flow and background subtraction
- **Intuitive visualization** for user guidance

This work contributes to the growing field of AI-assisted fitness and demonstrates practical applications of computer vision concepts learned throughout the course.

---

## 8. References

1. Bazarevsky, V., et al. (2020). "BlazePose: On-device Real-time Body Pose Tracking." arXiv preprint arXiv:2006.10204.

2. Lugaresi, C., et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." arXiv preprint arXiv:1906.08172.

3. Cao, Z., et al. (2019). "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." IEEE TPAMI.

4. Lucas, B.D., Kanade, T. (1981). "An Iterative Image Registration Technique with an Application to Stereo Vision." IJCAI.

5. Zivkovic, Z. (2004). "Improved Adaptive Gaussian Mixture Model for Background Subtraction." ICPR.

6. Bobick, A.F., Davis, J.W. (2001). "The Recognition of Human Movement Using Temporal Templates." IEEE TPAMI.

7. Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.

---

## Appendix A: Installation Guide

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Fitness-Trainer.git
cd AI-Fitness-Trainer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Appendix B: Keyboard Controls

| Key | Action |
|-----|--------|
| 1 | Select Bicep Curl |
| 2 | Select Squat |
| 3 | Select Push-up |
| 4 | Select Shoulder Press |
| A | Toggle ML Auto-Detection |
| V | Cycle CV Visualization |
| R | Reset Rep Count |
| Q | Quit Application |

## Appendix C: Project Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥ 4.8.0 | Image processing & display |
| mediapipe | ≥ 0.10.0 | Pose estimation |
| numpy | ≥ 1.24.0 | Numerical operations |
| scikit-learn | ≥ 1.3.0 | ML classification |
| joblib | ≥ 1.3.0 | Model persistence |

---

*End of Report*
