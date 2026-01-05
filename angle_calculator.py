"""
Angle Calculator Module
Provides utility functions for calculating angles between body landmarks.
"""

import numpy as np
import math


def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points.
    The angle is calculated at point2 (the vertex).
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point (vertex) as (x, y) tuple
        point3: Third point as (x, y) tuple
        
    Returns:
        Angle in degrees (0-180)
    """
    # Convert points to numpy arrays
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate cosine of angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clamp to avoid numerical errors with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
        
    Returns:
        Distance in pixels
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def get_landmark_coords(landmarks, landmark_id):
    """
    Extract (x, y) coordinates from landmarks list for a specific landmark.
    
    Args:
        landmarks: List of landmark dictionaries from PoseDetector
        landmark_id: ID of the landmark to extract
        
    Returns:
        Tuple (x, y) or None if landmark not found
    """
    for lm in landmarks:
        if lm['id'] == landmark_id:
            return (lm['x'], lm['y'])
    return None


def calculate_body_angle(landmarks, id1, id2, id3):
    """
    Calculate angle between three landmarks by their IDs.
    
    Args:
        landmarks: List of landmark dictionaries
        id1: ID of first landmark
        id2: ID of second landmark (vertex)
        id3: ID of third landmark
        
    Returns:
        Angle in degrees, or None if any landmark not found
    """
    p1 = get_landmark_coords(landmarks, id1)
    p2 = get_landmark_coords(landmarks, id2)
    p3 = get_landmark_coords(landmarks, id3)
    
    if p1 is None or p2 is None or p3 is None:
        return None
    
    return calculate_angle(p1, p2, p3)


def calculate_vertical_angle(point1, point2):
    """
    Calculate the angle of a line relative to vertical.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
        
    Returns:
        Angle in degrees from vertical (0 = perfectly vertical)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # Calculate angle from vertical
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    
    return angle


def calculate_horizontal_angle(point1, point2):
    """
    Calculate the angle of a line relative to horizontal.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
        
    Returns:
        Angle in degrees from horizontal (0 = perfectly horizontal)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # Calculate angle from horizontal
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    
    return angle


def is_point_above(point1, point2):
    """
    Check if point1 is above point2 (lower y value means higher on screen).
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
        
    Returns:
        Boolean indicating if point1 is above point2
    """
    return point1[1] < point2[1]


def get_midpoint(point1, point2):
    """
    Calculate the midpoint between two points.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
        
    Returns:
        Tuple (x, y) of midpoint
    """
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
