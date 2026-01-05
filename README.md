# AI Fitness Trainer 🏋️

A real-time AI-powered fitness trainer that uses computer vision to detect body poses, count exercise repetitions, and provide form feedback.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

## ✨ Features

- **Real-time Pose Detection** - Track 33 body landmarks using Google's MediaPipe
- **Automatic Rep Counting** - Counts repetitions for multiple exercises
- **Form Feedback** - Real-time analysis with corrective suggestions
- **Session Statistics** - Track duration, total reps, and reps per minute
- **4 Exercises Supported**:
  - 💪 Bicep Curls
  - 🦵 Squats  
  - 🫸 Push-ups
  - 🙆 Shoulder Press

## 📸 Demo

```
┌─────────────────────────────────────────────────────────────┐
│  AI FITNESS TRAINER                    Exercise: Bicep Curl │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────┐                      ┌──────────────────┐  │
│ │ REPS  │STAGE │                      │ Time: 2:45       │  │
│ │  12   │ UP ⬆ │    [VIDEO FEED]      │ Total Reps: 24   │  │
│ │       │      │    [WITH POSE]       │ RPM: 8.7         │  │
│ └──────────────┘                      └──────────────────┘  │
│                                                             │
│  [1]Curl [2]Squat [3]Push-up [4]Press | [R]Reset [Q]Quit   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ● FORM FEEDBACK: Great form! Keep it up!                │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-fitness-trainer.git
   cd ai-fitness-trainer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Bicep Curl |
| `2` | Switch to Squat |
| `3` | Switch to Push-up |
| `4` | Switch to Shoulder Press |
| `R` | Reset rep count |
| `Q` | Quit application |

## 🏗️ Project Structure

```
ai-fitness-trainer/
├── main.py               # Main application entry point
├── pose_detector.py      # MediaPipe pose detection wrapper
├── angle_calculator.py   # Angle calculation utilities
├── exercise_detector.py  # Exercise-specific detection logic
├── rep_counter.py        # Rep counting system
├── form_analyzer.py      # Form feedback analyzer
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🔧 How It Works

1. **Pose Detection**: Uses MediaPipe Pose to detect 33 body landmarks in real-time
2. **Angle Calculation**: Calculates joint angles (elbow, knee, shoulder) using vector mathematics
3. **Stage Detection**: Determines exercise stage (up/down) based on angle thresholds
4. **Rep Counting**: Counts a rep when transitioning from down → up position
5. **Form Analysis**: Checks body alignment and provides corrective feedback

## 📊 Technologies Used

- **Python 3.8+** - Programming language
- **OpenCV** - Video capture and display
- **MediaPipe** - Pose detection AI model
- **NumPy** - Mathematical operations

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add more exercises
- Improve form detection

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created as a Computer Vision semester project.

---

⭐ If you found this project helpful, please give it a star!
