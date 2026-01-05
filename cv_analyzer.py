"""
CV Analyzer Module
Classical Computer Vision techniques for motion analysis and visualization.

This module demonstrates traditional CV techniques for the project:
1. Optical Flow (Lucas-Kanade) - Motion vector visualization
2. Background Subtraction (MOG2) - Motion detection
3. Motion Energy Image (MEI) - Movement accumulation
4. Edge Detection (Canny) - Body contour enhancement
"""

import cv2
import numpy as np
from collections import deque


class OpticalFlowAnalyzer:
    """
    Optical Flow analysis using Lucas-Kanade method.
    
    Tracks motion between consecutive frames and visualizes
    movement vectors to show direction and speed of motion.
    """
    
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7):
        """
        Initialize optical flow analyzer.
        
        Args:
            max_corners: Maximum number of corners to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
        """
        # Parameters for ShiTomasi corner detection
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': quality_level,
            'minDistance': min_distance,
            'blockSize': 7
        }
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        self.prev_gray = None
        self.prev_points = None
        self.motion_vectors = []
        self.frame_count = 0
        self.redetect_interval = 30  # Re-detect features every N frames
        
        # Motion statistics
        self.avg_motion = 0
        self.motion_direction = (0, 0)
    
    def process(self, frame):
        """
        Process frame and calculate optical flow.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (motion_magnitude, motion_direction, flow_vectors)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize or re-detect features periodically
        if self.prev_gray is None or self.frame_count % self.redetect_interval == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_gray = gray.copy()
            self.frame_count += 1
            return 0, (0, 0), []
        
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_gray = gray.copy()
            return 0, (0, 0), []
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if next_points is None:
            self.prev_gray = gray.copy()
            return 0, (0, 0), []
        
        # Select good points
        good_old = self.prev_points[status == 1]
        good_new = next_points[status == 1]
        
        # Calculate motion vectors
        self.motion_vectors = []
        total_dx, total_dy = 0, 0
        
        for old, new in zip(good_old, good_new):
            dx = new[0] - old[0]
            dy = new[1] - old[1]
            magnitude = np.sqrt(dx**2 + dy**2)
            
            self.motion_vectors.append({
                'start': tuple(old.astype(int)),
                'end': tuple(new.astype(int)),
                'magnitude': magnitude,
                'direction': (dx, dy)
            })
            
            total_dx += dx
            total_dy += dy
        
        # Calculate average motion
        num_vectors = len(self.motion_vectors)
        if num_vectors > 0:
            self.avg_motion = np.mean([v['magnitude'] for v in self.motion_vectors])
            self.motion_direction = (total_dx / num_vectors, total_dy / num_vectors)
        else:
            self.avg_motion = 0
            self.motion_direction = (0, 0)
        
        # Update for next frame
        self.prev_gray = gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.frame_count += 1
        
        return self.avg_motion, self.motion_direction, self.motion_vectors
    
    def visualize(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw optical flow vectors on frame.
        
        Args:
            frame: Input frame to draw on
            color: Vector color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with flow vectors drawn
        """
        output = frame.copy()
        
        for vector in self.motion_vectors:
            start = vector['start']
            end = vector['end']
            magnitude = vector['magnitude']
            
            # Only draw significant motion
            if magnitude > 2:
                # Color based on magnitude (green to red)
                intensity = min(magnitude / 20, 1.0)
                vec_color = (0, int(255 * (1 - intensity)), int(255 * intensity))
                
                cv2.arrowedLine(output, start, end, vec_color, thickness, tipLength=0.3)
        
        # Draw motion indicator
        if self.avg_motion > 5:
            cv2.putText(output, f"Motion: {self.avg_motion:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return output
    
    def reset(self):
        """Reset the analyzer state."""
        self.prev_gray = None
        self.prev_points = None
        self.motion_vectors = []
        self.frame_count = 0
        self.avg_motion = 0
        self.motion_direction = (0, 0)


class BackgroundSubtractor:
    """
    Background subtraction for motion detection using MOG2.
    
    Separates moving foreground (person) from static background
    to create motion masks and silhouettes.
    """
    
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        """
        Initialize background subtractor.
        
        Args:
            history: Number of frames for background model
            var_threshold: Variance threshold for foreground detection
            detect_shadows: Whether to detect shadows
        """
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Morphological kernels for noise removal
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Motion statistics
        self.motion_area = 0
        self.motion_percentage = 0
        self.bounding_box = None
    
    def process(self, frame):
        """
        Process frame and extract foreground mask.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary foreground mask
        """
        # Apply background subtraction
        fg_mask = self.subtractor.apply(frame)
        
        # Remove shadows (shadows are gray in MOG2)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_large)
        
        # Calculate motion statistics
        self.motion_area = cv2.countNonZero(fg_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        self.motion_percentage = (self.motion_area / total_pixels) * 100
        
        # Find bounding box of motion
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            self.bounding_box = cv2.boundingRect(largest)
        else:
            self.bounding_box = None
        
        return fg_mask
    
    def visualize(self, frame, mask, overlay_color=(0, 255, 0), alpha=0.3):
        """
        Visualize motion detection on frame.
        
        Args:
            frame: Input frame
            mask: Foreground mask
            overlay_color: Color for motion highlight
            alpha: Overlay transparency
            
        Returns:
            Frame with motion visualization
        """
        output = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = overlay_color
        
        # Blend with original
        output = cv2.addWeighted(overlay, alpha, output, 1, 0)
        
        # Draw bounding box
        if self.bounding_box:
            x, y, w, h = self.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw motion percentage
        cv2.putText(output, f"Motion: {self.motion_percentage:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return output
    
    def get_silhouette(self, frame, mask):
        """
        Extract silhouette from frame using mask.
        
        Args:
            frame: Input frame
            mask: Foreground mask
            
        Returns:
            Frame with only foreground visible
        """
        silhouette = cv2.bitwise_and(frame, frame, mask=mask)
        return silhouette
    
    def reset(self):
        """Reset background model."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )


class MotionEnergyImage:
    """
    Motion Energy Image (MEI) generator.
    
    Accumulates motion over time to create a visualization
    of movement patterns and exercise motion paths.
    """
    
    def __init__(self, decay_rate=0.95, threshold=25):
        """
        Initialize MEI generator.
        
        Args:
            decay_rate: How fast old motion fades (0-1)
            threshold: Frame difference threshold
        """
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.mei = None
        self.prev_gray = None
        self.motion_history = deque(maxlen=60)  # Store last 60 frames of motion
    
    def process(self, frame):
        """
        Process frame and update motion energy image.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Motion energy image (grayscale, 0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.mei = np.zeros_like(gray, dtype=np.float32)
            return (self.mei * 255).astype(np.uint8)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray, self.prev_gray)
        
        # Threshold to get motion
        _, motion = cv2.threshold(diff, self.threshold, 1, cv2.THRESH_BINARY)
        motion = motion.astype(np.float32)
        
        # Decay old motion and add new
        self.mei = self.mei * self.decay_rate + motion
        self.mei = np.clip(self.mei, 0, 1)
        
        # Store in history for motion analysis
        self.motion_history.append(motion.copy())
        
        self.prev_gray = gray
        
        return (self.mei * 255).astype(np.uint8)
    
    def visualize(self, frame):
        """
        Create colored MEI visualization overlay.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with MEI heatmap overlay
        """
        if self.mei is None:
            return frame
        
        output = frame.copy()
        
        # Create heatmap from MEI
        mei_uint8 = (self.mei * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mei_uint8, cv2.COLORMAP_JET)
        
        # Only show where there's motion
        mask = (mei_uint8 > 10).astype(np.uint8) * 255
        
        # Blend heatmap with frame
        for c in range(3):
            output[:, :, c] = np.where(
                mask > 0,
                cv2.addWeighted(heatmap[:, :, c], 0.7, output[:, :, c], 0.3, 0),
                output[:, :, c]
            )
        
        return output
    
    def get_motion_intensity(self):
        """Get overall motion intensity (0-1)."""
        if self.mei is None:
            return 0
        return np.mean(self.mei)
    
    def get_motion_center(self):
        """Get center of motion activity."""
        if self.mei is None:
            return None
        
        # Find centroid of motion
        moments = cv2.moments(self.mei)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None
    
    def reset(self):
        """Reset MEI accumulator."""
        self.mei = None
        self.prev_gray = None
        self.motion_history.clear()


class EdgeDetector:
    """
    Edge detection for body contour enhancement.
    
    Uses Canny edge detection to highlight body outlines
    and improve pose visibility.
    """
    
    def __init__(self, low_threshold=50, high_threshold=150):
        """
        Initialize edge detector.
        
        Args:
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def process(self, frame):
        """
        Detect edges in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Edge image (binary)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        return edges
    
    def visualize(self, frame, edges, color=(0, 255, 0)):
        """
        Overlay edges on frame.
        
        Args:
            frame: Input frame
            edges: Edge image
            color: Edge color
            
        Returns:
            Frame with colored edges
        """
        output = frame.copy()
        
        # Create colored edge overlay
        edge_colored = np.zeros_like(frame)
        edge_colored[edges > 0] = color
        
        # Blend
        output = cv2.addWeighted(output, 1, edge_colored, 0.5, 0)
        
        return output


class CVAnalyzer:
    """
    Main CV analysis class combining all techniques.
    
    Provides unified interface for all CV analysis features
    with easy visualization and control.
    """
    
    def __init__(self):
        """Initialize all CV analyzers."""
        self.optical_flow = OpticalFlowAnalyzer()
        self.bg_subtractor = BackgroundSubtractor()
        self.mei_generator = MotionEnergyImage()
        self.edge_detector = EdgeDetector()
        
        # Visualization modes
        self.show_optical_flow = False
        self.show_background_sub = False
        self.show_mei = False
        self.show_edges = False
        
        # Analysis results
        self.analysis_results = {}
    
    def process(self, frame):
        """
        Process frame through all enabled analyzers.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        # Optical Flow
        motion_mag, motion_dir, vectors = self.optical_flow.process(frame)
        results['optical_flow'] = {
            'motion_magnitude': motion_mag,
            'motion_direction': motion_dir,
            'num_vectors': len(vectors)
        }
        
        # Background Subtraction
        fg_mask = self.bg_subtractor.process(frame)
        results['background_sub'] = {
            'motion_area': self.bg_subtractor.motion_area,
            'motion_percentage': self.bg_subtractor.motion_percentage,
            'bounding_box': self.bg_subtractor.bounding_box,
            'mask': fg_mask
        }
        
        # Motion Energy Image
        mei = self.mei_generator.process(frame)
        results['mei'] = {
            'motion_intensity': self.mei_generator.get_motion_intensity(),
            'motion_center': self.mei_generator.get_motion_center(),
            'image': mei
        }
        
        # Edge Detection
        edges = self.edge_detector.process(frame)
        results['edges'] = {
            'image': edges,
            'edge_density': np.count_nonzero(edges) / edges.size
        }
        
        self.analysis_results = results
        return results
    
    def visualize(self, frame):
        """
        Apply enabled visualizations to frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with visualizations applied
        """
        output = frame.copy()
        
        if self.show_optical_flow:
            output = self.optical_flow.visualize(output)
        
        if self.show_background_sub and 'mask' in self.analysis_results.get('background_sub', {}):
            mask = self.analysis_results['background_sub']['mask']
            output = self.bg_subtractor.visualize(output, mask)
        
        if self.show_mei:
            output = self.mei_generator.visualize(output)
        
        if self.show_edges and 'image' in self.analysis_results.get('edges', {}):
            edges = self.analysis_results['edges']['image']
            output = self.edge_detector.visualize(output, edges)
        
        return output
    
    def draw_analysis_panel(self, frame, x=220, y=210, width=180, height=100):
        """
        Draw CV analysis statistics panel.
        
        Args:
            frame: Frame to draw on
            x, y: Panel position
            width, height: Panel dimensions
        """
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "CV ANALYSIS", (x + 10, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Motion info
        if 'optical_flow' in self.analysis_results:
            motion = self.analysis_results['optical_flow']['motion_magnitude']
            cv2.putText(frame, f"Motion: {motion:.1f}", (x + 10, y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Motion percentage
        if 'background_sub' in self.analysis_results:
            pct = self.analysis_results['background_sub']['motion_percentage']
            cv2.putText(frame, f"Area: {pct:.1f}%", (x + 10, y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # MEI intensity
        if 'mei' in self.analysis_results:
            intensity = self.analysis_results['mei']['motion_intensity']
            cv2.putText(frame, f"Energy: {intensity:.2f}", (x + 10, y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Active modes
        modes = []
        if self.show_optical_flow:
            modes.append("OF")
        if self.show_background_sub:
            modes.append("BG")
        if self.show_mei:
            modes.append("MEI")
        if self.show_edges:
            modes.append("E")
        
        mode_str = "+".join(modes) if modes else "None"
        cv2.putText(frame, f"Mode: {mode_str}", (x + 10, y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def toggle_optical_flow(self):
        """Toggle optical flow visualization."""
        self.show_optical_flow = not self.show_optical_flow
        return self.show_optical_flow
    
    def toggle_background_sub(self):
        """Toggle background subtraction visualization."""
        self.show_background_sub = not self.show_background_sub
        return self.show_background_sub
    
    def toggle_mei(self):
        """Toggle MEI visualization."""
        self.show_mei = not self.show_mei
        return self.show_mei
    
    def toggle_edges(self):
        """Toggle edge detection visualization."""
        self.show_edges = not self.show_edges
        return self.show_edges
    
    def cycle_mode(self):
        """Cycle through visualization modes."""
        current = (self.show_optical_flow, self.show_background_sub, 
                   self.show_mei, self.show_edges)
        
        modes = [
            (False, False, False, False),  # None
            (True, False, False, False),   # Optical Flow only
            (False, True, False, False),   # Background Sub only
            (False, False, True, False),   # MEI only
            (False, False, False, True),   # Edges only
            (True, True, True, True),      # All
        ]
        
        # Find current mode and go to next
        try:
            idx = modes.index(current)
            next_idx = (idx + 1) % len(modes)
        except ValueError:
            next_idx = 0
        
        (self.show_optical_flow, self.show_background_sub,
         self.show_mei, self.show_edges) = modes[next_idx]
        
        return modes[next_idx]
    
    def reset(self):
        """Reset all analyzers."""
        self.optical_flow.reset()
        self.bg_subtractor.reset()
        self.mei_generator.reset()
        self.analysis_results = {}


# Demo usage
if __name__ == "__main__":
    print("CV Analyzer Module")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CVAnalyzer()
    print("CV Analyzer initialized with:")
    print("  - Optical Flow (Lucas-Kanade)")
    print("  - Background Subtraction (MOG2)")
    print("  - Motion Energy Image (MEI)")
    print("  - Edge Detection (Canny)")
    
    print("\nControls:")
    print("  V - Cycle through visualization modes")
    print("  O - Toggle Optical Flow")
    print("  B - Toggle Background Subtraction")
    print("  M - Toggle Motion Energy Image")
    print("  E - Toggle Edge Detection")
    
    print("\nModule ready for integration!")
