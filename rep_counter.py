"""
Rep Counter Module
Tracks exercise repetitions based on stage transitions.
"""

from datetime import datetime


class RepCounter:
    """
    Counts exercise repetitions by tracking stage transitions.
    A rep is counted when completing a full up-down-up or down-up-down cycle.
    """
    
    def __init__(self):
        """Initialize the rep counter."""
        self.count = 0
        self.stage = None
        self.prev_stage = None
        self.half_rep = False
        
        # Session tracking
        self.session_start = None
        self.session_reps = []
        self.total_sets = 0
    
    def update(self, new_stage):
        """
        Update the counter with a new stage.
        
        Args:
            new_stage: Current exercise stage ('up', 'down', 'transition', 'unknown')
            
        Returns:
            Boolean indicating if a rep was just completed
        """
        if new_stage in ['transition', 'unknown']:
            return False
        
        if self.session_start is None:
            self.session_start = datetime.now()
        
        self.prev_stage = self.stage
        self.stage = new_stage
        
        # Check for rep completion
        rep_completed = False
        
        # A rep is completed when going from down -> up
        if self.prev_stage == 'down' and self.stage == 'up':
            self.count += 1
            self.session_reps.append(datetime.now())
            rep_completed = True
        
        return rep_completed
    
    def get_count(self):
        """
        Get the current rep count.
        
        Returns:
            Integer rep count
        """
        return self.count
    
    def get_stage(self):
        """
        Get the current stage.
        
        Returns:
            String stage name
        """
        return self.stage if self.stage else 'Ready'
    
    def reset(self):
        """Reset the rep counter for a new set."""
        if self.count > 0:
            self.total_sets += 1
        self.count = 0
        self.stage = None
        self.prev_stage = None
        self.half_rep = False
    
    def reset_session(self):
        """Reset all session data."""
        self.reset()
        self.session_start = None
        self.session_reps = []
        self.total_sets = 0
    
    def get_session_stats(self):
        """
        Get statistics for the current session.
        
        Returns:
            Dictionary with session statistics
        """
        if self.session_start is None:
            return {
                'duration': '0:00',
                'total_reps': 0,
                'sets_completed': 0,
                'reps_per_minute': 0
            }
        
        duration = datetime.now() - self.session_start
        total_seconds = int(duration.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        total_reps = len(self.session_reps)
        reps_per_minute = 0
        if total_seconds > 0:
            reps_per_minute = round((total_reps / total_seconds) * 60, 1)
        
        return {
            'duration': f'{minutes}:{seconds:02d}',
            'total_reps': total_reps,
            'sets_completed': self.total_sets,
            'reps_per_minute': reps_per_minute,
            'current_set_reps': self.count
        }
    
    def get_progress_percentage(self, target_reps=10):
        """
        Get progress towards target reps as a percentage.
        
        Args:
            target_reps: Target number of reps for the set
            
        Returns:
            Float percentage (0-100)
        """
        return min(100, (self.count / target_reps) * 100)
