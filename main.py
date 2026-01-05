"""
AI Fitness Trainer
Real-time fitness tracking with pose detection, rep counting, and form feedback.

Controls:
    1 - Bicep Curl
    2 - Squat
    3 - Push-up
    4 - Shoulder Press
    R - Reset rep count
    Q - Quit
"""

import cv2
from datetime import datetime

from pose_detector import PoseDetector
from exercise_detector import ExerciseDetector, ExerciseType
from rep_counter import RepCounter
from form_analyzer import FormAnalyzer


class FitnessTrainerUI:
    """
    UI rendering for the AI Fitness Trainer.
    """
    
    # Colors (BGR format)
    COLORS = {
        'primary': (255, 100, 50),      # Blue
        'secondary': (50, 200, 50),     # Green
        'accent': (50, 200, 255),       # Yellow/Orange
        'warning': (0, 165, 255),       # Orange
        'danger': (0, 0, 255),          # Red
        'success': (0, 255, 0),         # Green
        'dark': (40, 40, 40),           # Dark gray
        'light': (220, 220, 220),       # Light gray
        'white': (255, 255, 255),
        'black': (0, 0, 0),
    }
    
    def __init__(self, frame_width, frame_height):
        """Initialize UI with frame dimensions."""
        self.width = frame_width
        self.height = frame_height
    
    def draw_header(self, frame, exercise_name):
        """Draw the header with app title and current exercise."""
        # Header background
        cv2.rectangle(frame, (0, 0), (self.width, 60), self.COLORS['dark'], -1)
        
        # App title
        cv2.putText(frame, "AI FITNESS TRAINER", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['accent'], 2)
        
        # Current exercise
        cv2.putText(frame, f"Exercise: {exercise_name}", (self.width - 300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['white'], 2)
    
    def draw_stats_panel(self, frame, rep_count, stage, angle):
        """Draw the statistics panel with rep count and stage."""
        panel_y = 80
        panel_height = 120
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (250, panel_y + panel_height), 
                      self.COLORS['dark'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Rep count
        cv2.putText(frame, "REPS", (30, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['light'], 1)
        cv2.putText(frame, str(rep_count), (30, panel_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, self.COLORS['accent'], 3)
        
        # Divider
        cv2.line(frame, (120, panel_y + 10), (120, panel_y + panel_height - 10),
                 self.COLORS['light'], 1)
        
        # Stage
        cv2.putText(frame, "STAGE", (140, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['light'], 1)
        
        stage_color = self.COLORS['success'] if stage == 'up' else self.COLORS['warning']
        stage_text = stage.upper() if stage else "READY"
        cv2.putText(frame, stage_text, (140, panel_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, stage_color, 2)
        
        # Angle display
        if angle:
            cv2.putText(frame, f"Angle: {int(angle)}°", (30, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)
    
    def draw_feedback_panel(self, frame, feedback):
        """Draw the form feedback panel."""
        panel_y = self.height - 80
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (self.width - 10, self.height - 10),
                      self.COLORS['dark'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Feedback icon
        status_color = feedback.get_color()
        cv2.circle(frame, (35, panel_y + 35), 15, status_color, -1)
        
        # Feedback text
        cv2.putText(frame, "FORM FEEDBACK:", (60, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)
        cv2.putText(frame, feedback.message, (60, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    def draw_controls(self, frame):
        """Draw keyboard controls info."""
        controls = "[1]Curl [2]Squat [3]Push-up [4]Press | [R]Reset [Q]Quit"
        
        # Controls background at bottom
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (self.width - text_size[0]) // 2
        
        cv2.putText(frame, controls, (text_x, self.height - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)
    
    def draw_progress_bar(self, frame, progress, x, y, width, height):
        """Draw a progress bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['dark'], -1)
        
        # Progress fill
        fill_width = int((progress / 100) * width)
        if fill_width > 0:
            color = self.COLORS['success'] if progress >= 100 else self.COLORS['accent']
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['light'], 1)
    
    def draw_session_stats(self, frame, stats):
        """Draw session statistics panel."""
        panel_x = self.width - 200
        panel_y = 80
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (self.width - 10, panel_y + 100),
                      self.COLORS['dark'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Session time
        cv2.putText(frame, f"Time: {stats['duration']}", (panel_x + 15, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)
        
        # Total reps
        cv2.putText(frame, f"Total Reps: {stats['total_reps']}", (panel_x + 15, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)
        
        # Reps per minute
        cv2.putText(frame, f"RPM: {stats['reps_per_minute']}", (panel_x + 15, panel_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['light'], 1)


class AIFitnessTrainer:
    """
    Main AI Fitness Trainer application.
    """
    
    def __init__(self):
        """Initialize the fitness trainer."""
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.exercise_detector = ExerciseDetector()
        self.rep_counter = RepCounter()
        self.form_analyzer = FormAnalyzer()
        
        self.ui = None
        self.running = False
        
        # Exercise key mappings
        self.exercise_keys = {
            ord('1'): ExerciseType.BICEP_CURL,
            ord('2'): ExerciseType.SQUAT,
            ord('3'): ExerciseType.PUSH_UP,
            ord('4'): ExerciseType.SHOULDER_PRESS,
        }
    
    def set_exercise(self, exercise_type):
        """Change the current exercise."""
        self.exercise_detector.set_exercise(exercise_type)
        self.form_analyzer.set_exercise(exercise_type)
        self.rep_counter.reset()
        print(f"Switched to: {exercise_type.value}")
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with UI overlay
        """
        # Detect pose
        frame = self.pose_detector.find_pose(frame, draw=True)
        landmarks = self.pose_detector.get_landmarks(frame)
        
        # Get exercise angle
        angle = self.exercise_detector.get_exercise_angle(landmarks)
        
        # Determine stage
        stage = self.exercise_detector.get_stage(angle)
        
        # Update rep counter
        self.rep_counter.update(stage)
        
        # Analyze form
        feedback = self.form_analyzer.analyze(landmarks)
        
        # Draw UI
        exercise_info = self.exercise_detector.get_exercise_info()
        self.ui.draw_header(frame, exercise_info.get('name', 'Unknown'))
        self.ui.draw_stats_panel(frame, self.rep_counter.get_count(), 
                                  self.rep_counter.get_stage(), angle)
        self.ui.draw_session_stats(frame, self.rep_counter.get_session_stats())
        self.ui.draw_feedback_panel(frame, feedback)
        self.ui.draw_controls(frame)
        
        return frame
    
    def handle_key(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            Boolean indicating if app should continue running
        """
        if key == ord('q') or key == ord('Q'):
            return False
        
        if key == ord('r') or key == ord('R'):
            self.rep_counter.reset()
            print("Rep count reset!")
        
        if key in self.exercise_keys:
            self.set_exercise(self.exercise_keys[key])
        
        return True
    
    def run(self, camera_index=0):
        """
        Run the fitness trainer application.
        
        Args:
            camera_index: Index of the camera to use
        """
        print("=" * 50)
        print("    AI FITNESS TRAINER")
        print("=" * 50)
        print("\nInitializing camera...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            print("Please check your webcam connection.")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {frame_width}x{frame_height}")
        print("\nControls:")
        print("  [1] Bicep Curl")
        print("  [2] Squat")
        print("  [3] Push-up")
        print("  [4] Shoulder Press")
        print("  [R] Reset rep count")
        print("  [Q] Quit")
        print("\nStarting... Stand in front of the camera!")
        print("-" * 50)
        
        # Initialize UI
        self.ui = FitnessTrainerUI(frame_width, frame_height)
        
        self.running = True
        fps_start_time = datetime.now()
        fps_frame_count = 0
        current_fps = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Calculate and display FPS
            fps_frame_count += 1
            elapsed = (datetime.now() - fps_start_time).total_seconds()
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = datetime.now()
            
            cv2.putText(frame, f"FPS: {int(current_fps)}", (self.ui.width - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Display frame
            cv2.imshow('AI Fitness Trainer', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.running = self.handle_key(key)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        stats = self.rep_counter.get_session_stats()
        print("\n" + "=" * 50)
        print("    SESSION COMPLETE!")
        print("=" * 50)
        print(f"  Duration: {stats['duration']}")
        print(f"  Total Reps: {stats['total_reps']}")
        print(f"  Sets Completed: {stats['sets_completed']}")
        print(f"  Avg Reps/Min: {stats['reps_per_minute']}")
        print("=" * 50)


def main():
    """Main entry point."""
    trainer = AIFitnessTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
