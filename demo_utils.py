"""
Demo Utilities Module
Provides demo and presentation features for the AI Fitness Trainer.

Features:
1. Screenshot capture
2. Video recording
3. Training data collection with visual feedback
4. Performance benchmarking
"""

import cv2
import os
import time
import json
from datetime import datetime
import numpy as np


class ScreenshotCapture:
    """Captures and saves screenshots of the application."""
    
    def __init__(self, save_dir='screenshots'):
        """Initialize screenshot capture."""
        self.save_dir = save_dir
        self.screenshot_count = 0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def capture(self, frame, prefix='screenshot'):
        """
        Capture and save a screenshot.
        
        Args:
            frame: The frame to save
            prefix: Filename prefix
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}_{self.screenshot_count:03d}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, frame)
        self.screenshot_count += 1
        
        print(f"Screenshot saved: {filepath}")
        return filepath


class VideoRecorder:
    """Records video clips of exercise sessions."""
    
    def __init__(self, save_dir='recordings', fps=30):
        """Initialize video recorder."""
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        self.current_file = None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def start(self, frame_width, frame_height):
        """Start recording."""
        if self.is_recording:
            return False
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recording_{timestamp}.mp4"
        self.current_file = os.path.join(self.save_dir, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.current_file, 
            fourcc, 
            self.fps, 
            (frame_width, frame_height)
        )
        
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"Recording started: {self.current_file}")
        return True
    
    def write_frame(self, frame):
        """Write a frame to the recording."""
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1
    
    def stop(self):
        """Stop recording."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        duration = time.time() - self.start_time
        print(f"Recording saved: {self.current_file}")
        print(f"Duration: {duration:.1f}s, Frames: {self.frame_count}")
        
        return self.current_file
    
    def get_status(self):
        """Get recording status."""
        if not self.is_recording:
            return None
        
        duration = time.time() - self.start_time
        return {
            'recording': True,
            'duration': duration,
            'frames': self.frame_count,
            'file': self.current_file
        }


class PerformanceBenchmark:
    """Tracks and displays performance metrics."""
    
    def __init__(self, window_size=60):
        """Initialize benchmark tracker."""
        self.window_size = window_size
        self.frame_times = []
        self.processing_times = []
        self.pose_detection_times = []
        self.ml_times = []
        self.cv_times = []
        
        self.last_frame_time = None
    
    def start_frame(self):
        """Mark the start of a frame."""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
        return current_time
    
    def record_time(self, category, duration):
        """Record a timing measurement."""
        if category == 'pose':
            self.pose_detection_times.append(duration)
            if len(self.pose_detection_times) > self.window_size:
                self.pose_detection_times.pop(0)
        elif category == 'ml':
            self.ml_times.append(duration)
            if len(self.ml_times) > self.window_size:
                self.ml_times.pop(0)
        elif category == 'cv':
            self.cv_times.append(duration)
            if len(self.cv_times) > self.window_size:
                self.cv_times.pop(0)
    
    def get_fps(self):
        """Get current FPS."""
        if not self.frame_times:
            return 0
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def get_stats(self):
        """Get performance statistics."""
        stats = {
            'fps': self.get_fps(),
            'avg_frame_time_ms': np.mean(self.frame_times) * 1000 if self.frame_times else 0,
        }
        
        if self.pose_detection_times:
            stats['pose_time_ms'] = np.mean(self.pose_detection_times) * 1000
        if self.ml_times:
            stats['ml_time_ms'] = np.mean(self.ml_times) * 1000
        if self.cv_times:
            stats['cv_time_ms'] = np.mean(self.cv_times) * 1000
        
        return stats
    
    def draw_overlay(self, frame, x=10, y=340):
        """Draw performance overlay on frame."""
        stats = self.get_stats()
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 150, y + 70), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "PERFORMANCE", (x + 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # FPS
        fps = stats['fps']
        fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (x + 10, y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)
        
        # Frame time
        cv2.putText(frame, f"Frame: {stats['avg_frame_time_ms']:.1f}ms", (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


class TrainingModeUI:
    """UI for collecting training data with visual feedback."""
    
    EXERCISE_LABELS = {
        0: 'Bicep Curl',
        1: 'Squat',
        2: 'Push-up',
        3: 'Shoulder Press',
        4: 'Standing/Unknown'
    }
    
    def __init__(self):
        """Initialize training mode UI."""
        self.is_active = False
        self.current_label = 0
        self.samples_collected = {i: 0 for i in range(5)}
        self.collection_countdown = 0
        self.last_collection_time = 0
        self.auto_collect = False
        self.collection_interval = 0.5  # seconds
    
    def toggle(self):
        """Toggle training mode."""
        self.is_active = not self.is_active
        if self.is_active:
            print("\n" + "=" * 40)
            print("TRAINING DATA COLLECTION MODE")
            print("=" * 40)
            print("Use number keys (0-4) to select label:")
            for key, label in self.EXERCISE_LABELS.items():
                print(f"  [{key}] {label}")
            print("\nPress SPACE to collect sample")
            print("Press C to toggle auto-collect")
            print("Press T again to exit training mode")
            print("=" * 40 + "\n")
        else:
            print("Training mode disabled")
        return self.is_active
    
    def set_label(self, label_id):
        """Set current label for collection."""
        if 0 <= label_id <= 4:
            self.current_label = label_id
            print(f"Label set to: {self.EXERCISE_LABELS[label_id]}")
    
    def should_collect(self):
        """Check if we should collect a sample now."""
        if not self.is_active:
            return False
        
        if self.auto_collect:
            current_time = time.time()
            if current_time - self.last_collection_time >= self.collection_interval:
                self.last_collection_time = current_time
                return True
        
        return False
    
    def record_sample(self):
        """Record that a sample was collected."""
        self.samples_collected[self.current_label] += 1
        total = sum(self.samples_collected.values())
        print(f"Collected sample for '{self.EXERCISE_LABELS[self.current_label]}' "
              f"(Total: {total})")
    
    def toggle_auto_collect(self):
        """Toggle automatic collection."""
        self.auto_collect = not self.auto_collect
        status = "ON" if self.auto_collect else "OFF"
        print(f"Auto-collect: {status}")
        self.last_collection_time = time.time()
    
    def draw_overlay(self, frame, x=10, y=420):
        """Draw training mode overlay."""
        if not self.is_active:
            return
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 220, y + 80), (40, 40, 80), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border to indicate training mode
        cv2.rectangle(frame, (x, y), (x + 220, y + 80), (0, 0, 255), 2)
        
        # Title
        cv2.putText(frame, "TRAINING MODE", (x + 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Current label
        label_name = self.EXERCISE_LABELS[self.current_label]
        cv2.putText(frame, f"Label: {label_name}", (x + 10, y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Samples count
        count = self.samples_collected[self.current_label]
        cv2.putText(frame, f"Samples: {count}", (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Auto-collect indicator
        if self.auto_collect:
            cv2.circle(frame, (x + 200, y + 15), 8, (0, 255, 0), -1)
            cv2.putText(frame, "AUTO", (x + 160, y + 72),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    def get_summary(self):
        """Get collection summary."""
        summary = []
        for label_id, count in self.samples_collected.items():
            summary.append(f"{self.EXERCISE_LABELS[label_id]}: {count}")
        return "\n".join(summary)


class DemoMode:
    """Manages demo/presentation features."""
    
    def __init__(self):
        """Initialize demo mode."""
        self.screenshot = ScreenshotCapture()
        self.recorder = VideoRecorder()
        self.benchmark = PerformanceBenchmark()
        self.training_ui = TrainingModeUI()
        
        self.show_performance = False
        self.show_help = False
    
    def handle_key(self, key):
        """
        Handle demo-related key presses.
        
        Args:
            key: Key code
            
        Returns:
            True if key was handled, False otherwise
        """
        # S - Screenshot
        if key == ord('s') or key == ord('S'):
            return 'screenshot'
        
        # P - Toggle performance display
        if key == ord('p') or key == ord('P'):
            self.show_performance = not self.show_performance
            status = "ON" if self.show_performance else "OFF"
            print(f"Performance display: {status}")
            return 'toggle_perf'
        
        # O - Start/Stop recording
        if key == ord('o') or key == ord('O'):
            return 'toggle_record'
        
        # T - Toggle training mode
        if key == ord('t') or key == ord('T'):
            self.training_ui.toggle()
            return 'toggle_training'
        
        # Training mode specific keys
        if self.training_ui.is_active:
            # 0-4 for labels
            if ord('0') <= key <= ord('4'):
                self.training_ui.set_label(key - ord('0'))
                return 'set_label'
            
            # Space to collect
            if key == ord(' '):
                return 'collect_sample'
            
            # C for auto-collect
            if key == ord('c') or key == ord('C'):
                self.training_ui.toggle_auto_collect()
                return 'toggle_auto'
        
        # H - Toggle help
        if key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
            return 'toggle_help'
        
        return None
    
    def draw_overlays(self, frame):
        """Draw all demo overlays."""
        if self.show_performance:
            self.benchmark.draw_overlay(frame)
        
        self.training_ui.draw_overlay(frame)
        
        # Recording indicator
        if self.recorder.is_recording:
            status = self.recorder.get_status()
            duration = status['duration']
            
            # Red recording dot
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {duration:.1f}s", 
                       (frame.shape[1] - 110, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Help overlay
        if self.show_help:
            self._draw_help(frame)
    
    def _draw_help(self, frame):
        """Draw help overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Title
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (w//4 + 20, h//4 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        shortcuts = [
            ("1-4", "Select Exercise"),
            ("A", "Toggle Auto-Detect (ML)"),
            ("V", "Cycle CV Modes"),
            ("R", "Reset Rep Count"),
            ("S", "Take Screenshot"),
            ("O", "Start/Stop Recording"),
            ("P", "Toggle Performance Stats"),
            ("T", "Toggle Training Mode"),
            ("H", "Show/Hide This Help"),
            ("Q", "Quit"),
        ]
        
        y_offset = h//4 + 60
        for key, action in shortcuts:
            cv2.putText(frame, f"[{key}]", (w//4 + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(frame, action, (w//4 + 80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25


# Demo usage
if __name__ == "__main__":
    print("Demo Utilities Module")
    print("=" * 50)
    print("Features:")
    print("  - Screenshot capture (S key)")
    print("  - Video recording (O key)")
    print("  - Performance benchmark (P key)")
    print("  - Training data collection (T key)")
    print("  - Help overlay (H key)")
    print("\nModule ready for integration!")
