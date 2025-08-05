#!/usr/bin/env python3
"""
Gaze processing and object selection
"""
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Optional, Tuple
try:
    from .utils import log_message
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message


class GazeProcessor:
    """Handles gaze data processing and object selection"""
    
    def __init__(self):
        pass
    
    def get_gaze_point_at_timestamp(self, gaze_data: pd.DataFrame, target_timestamp: float) -> Optional[Tuple[float, float]]:
        """Get gaze point at specific timestamp"""
        if gaze_data is None or gaze_data.empty:
            return None
        
        try:
            # Calculate target timestamp in nanoseconds
            video_start_time_ns = gaze_data['timestamp_ns'].iloc[0]
            target_timestamp_ns = video_start_time_ns + int(target_timestamp * 1e9)
            
            # Find closest timestamp
            time_diff = np.abs(gaze_data['timestamp_ns'] - target_timestamp_ns)
            closest_idx = time_diff.idxmin()
            
            closest_row = gaze_data.iloc[closest_idx]
            gaze_x, gaze_y = closest_row['gaze_x'], closest_row['gaze_y']
            
            # Check for valid gaze data
            if pd.isna(gaze_x) or pd.isna(gaze_y):
                return None
            
            return (float(gaze_x), float(gaze_y))
            
        except Exception as e:
            log_message(f"Failed to get gaze point: {str(e)}")
            return None
    
    def select_object_by_gaze(self, objects: List[Dict[str, Any]], gaze_point: Optional[Tuple[float, float]], 
                             sigma: float = 400) -> Optional[Dict[str, Any]]:
        """Select object based on gaze point using Gaussian probability"""
        if not objects:
            return None
        
        # If no gaze point, select randomly
        if gaze_point is None:
            selected = random.choice(objects)
            log_message(f"No gaze data, randomly selected: {selected['name']}")
            return selected
        
        gaze_x, gaze_y = gaze_point
        log_message(f"Gaze point at ({gaze_x:.1f}, {gaze_y:.1f})")
        
        # Calculate Gaussian probabilities based on distance from gaze to object centers
        probabilities = []
        object_info = []
        
        for obj in objects:
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance squared
            dist_sq = (center_x - gaze_x)**2 + (center_y - gaze_y)**2
            prob = np.exp(-dist_sq / (2 * sigma**2))
            
            probabilities.append(prob)
            object_info.append(f"{obj['name']} (dist: {np.sqrt(dist_sq):.1f})")
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # If all probabilities are 0, use uniform distribution
            probabilities = [1.0 / len(objects)] * len(objects)
        
        # Select object based on probabilities
        selected_idx = np.random.choice(len(objects), p=probabilities)
        selected_object = objects[selected_idx]
        
        log_message(f"Selected object: {selected_object['name']} (probability: {probabilities[selected_idx]:.3f})")
        
        return selected_object
    
    def calculate_selection_probabilities(self, objects: List[Dict[str, Any]], 
                                        gaze_point: Optional[Tuple[float, float]], 
                                        sigma: float = 400) -> List[float]:
        """Calculate selection probabilities for all objects"""
        if not objects:
            return []
        
        if gaze_point is None:
            return [1.0 / len(objects)] * len(objects)
        
        gaze_x, gaze_y = gaze_point
        probabilities = []
        
        for obj in objects:
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            dist_sq = (center_x - gaze_x)**2 + (center_y - gaze_y)**2
            prob = np.exp(-dist_sq / (2 * sigma**2))
            probabilities.append(prob)
        
        # Normalize
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(objects)] * len(objects)
        
        return probabilities


class ObjectSelector:
    """High-level object selection interface"""
    
    def __init__(self):
        self.gaze_processor = GazeProcessor()
    
    def select_key_object(self, objects: List[Dict[str, Any]], gaze_data: pd.DataFrame, 
                         timestamp: float, sigma: float = 400) -> Optional[Dict[str, Any]]:
        """Select key object based on gaze data at given timestamp"""
        if not objects:
            log_message("No objects detected")
            return None
        
        # Get gaze point at timestamp
        gaze_point = self.gaze_processor.get_gaze_point_at_timestamp(gaze_data, timestamp)
        
        # Select object based on gaze
        selected_object = self.gaze_processor.select_object_by_gaze(objects, gaze_point, sigma)
        
        if selected_object:
            log_message(f"Key object selected: {selected_object['name']} at {selected_object['bbox']}")
        
        return selected_object
    
    def get_selection_summary(self, objects: List[Dict[str, Any]], gaze_data: pd.DataFrame, 
                            timestamp: float, sigma: float = 400) -> Dict[str, Any]:
        """Get detailed selection summary"""
        gaze_point = self.gaze_processor.get_gaze_point_at_timestamp(gaze_data, timestamp)
        probabilities = self.gaze_processor.calculate_selection_probabilities(objects, gaze_point, sigma)
        
        summary = {
            'timestamp': timestamp,
            'gaze_point': gaze_point,
            'total_objects': len(objects),
            'objects_with_probabilities': []
        }
        
        for i, (obj, prob) in enumerate(zip(objects, probabilities)):
            summary['objects_with_probabilities'].append({
                'index': i,
                'name': obj['name'],
                'bbox': obj['bbox'],
                'selection_probability': prob
            })
        
        return summary


def main():
    """Test gaze processing functionality"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Test gaze processor')
    parser.add_argument('--gaze-csv', required=True, help='Gaze tracking CSV file')
    parser.add_argument('--timestamp', type=float, default=30.0, help='Test timestamp in seconds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gaze_csv):
        log_message(f"Gaze CSV file not found: {args.gaze_csv}")
        return
    
    # Load gaze data
    try:
        gaze_data = pd.read_csv(args.gaze_csv)
        log_message(f"Loaded gaze data: {len(gaze_data)} frames")
    except Exception as e:
        log_message(f"Failed to load gaze data: {str(e)}")
        return
    
    # Test gaze point extraction
    processor = GazeProcessor()
    
    gaze_point = processor.get_gaze_point_at_timestamp(gaze_data, args.timestamp)
    if gaze_point:
        log_message(f"Gaze point at {args.timestamp}s: ({gaze_point[0]:.1f}, {gaze_point[1]:.1f})")
    else:
        log_message(f"No valid gaze point at {args.timestamp}s")
    
    # Test with mock objects
    mock_objects = [
        {
            'name': 'object_1',
            'bbox': [100, 100, 200, 200],
            'normalized_bbox': [100, 100, 200, 200]
        },
        {
            'name': 'object_2', 
            'bbox': [300, 300, 400, 400],
            'normalized_bbox': [300, 300, 400, 400]
        }
    ]
    
    selected = processor.select_object_by_gaze(mock_objects, gaze_point)
    if selected:
        log_message(f"Selected object: {selected['name']}")
    
    # Test object selector
    selector = ObjectSelector()
    summary = selector.get_selection_summary(mock_objects, gaze_data, args.timestamp)
    
    log_message("Selection summary:")
    log_message(f"  Timestamp: {summary['timestamp']}")
    log_message(f"  Gaze point: {summary['gaze_point']}")
    log_message(f"  Total objects: {summary['total_objects']}")
    
    for obj_info in summary['objects_with_probabilities']:
        log_message(f"  {obj_info['name']}: {obj_info['selection_probability']:.3f}")


if __name__ == "__main__":
    main() 