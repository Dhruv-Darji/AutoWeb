"""
Evaluation Module for SeeAct-style Action Prediction

Prepares metrics for evaluation (structure only).
Actual evaluation happens after SeeAct baseline runs.

Metrics:
1. Action Accuracy (AA): Does predicted action_type match oracle?
2. Target Accuracy (TA): Does predicted target match oracle selector/bbox?
3. Value Accuracy (VA): For type actions, does value match?
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Step-level metrics
    action_accuracy: float = 0.0  # AA
    target_accuracy: float = 0.0  # TA
    value_accuracy: float = 0.0   # VA
    
    # Counts
    total_samples: int = 0
    correct_actions: int = 0
    correct_targets: int = 0
    correct_values: int = 0
    
    # Detailed breakdown
    action_type_breakdown: Dict[str, Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"""
Evaluation Metrics:
  Total Samples: {self.total_samples}
  
  Action Accuracy (AA): {self.action_accuracy:.2%}
    Correct: {self.correct_actions}/{self.total_samples}
  
  Target Accuracy (TA): {self.target_accuracy:.2%}
    Correct: {self.correct_targets}/{self.total_samples}
  
  Value Accuracy (VA): {self.value_accuracy:.2%}
    Correct: {self.correct_values}/{self.total_samples}
"""


class ActionEvaluator:
    """
    Evaluator for comparing predicted actions with oracle (ground-truth) actions.
    
    This class prepares the evaluation framework but doesn't run evaluation yet.
    Actual evaluation happens after the SeeAct baseline is operational.
    """
    
    def __init__(self, 
                 bbox_distance_threshold: float = 24.0,
                 text_match_threshold: float = 0.8):
        """
        Args:
            bbox_distance_threshold: Max pixel distance for bbox matching
            text_match_threshold: Min similarity score for text matching
        """
        self.bbox_threshold = bbox_distance_threshold
        self.text_threshold = text_match_threshold
        
        # Storage for results
        self.results: List[Dict] = []
    
    def compare_action_types(self, 
                            predicted: str, 
                            oracle: str) -> bool:
        """
        Compare action types.
        
        Args:
            predicted: Predicted action type
            oracle: Oracle (ground-truth) action type
        
        Returns:
            True if match
        """
        # Normalize
        pred_norm = predicted.lower().strip()
        oracle_norm = oracle.lower().strip()
        
        # Direct match
        if pred_norm == oracle_norm:
            return True
        
        # Handle common variations
        variations = {
            "select": ["click", "select"],
            "hover": ["click", "hover"],
        }
        
        for group in variations.values():
            if pred_norm in group and oracle_norm in group:
                return True
        
        return False
    
    def compare_selectors(self,
                         predicted: Optional[str],
                         oracle: Optional[str]) -> bool:
        """
        Compare CSS selectors.
        
        For now, uses simple string matching.
        Could be enhanced with selector normalization.
        """
        if predicted is None or oracle is None:
            return predicted == oracle
        
        # Normalize selectors
        pred_norm = predicted.strip().lower()
        oracle_norm = oracle.strip().lower()
        
        return pred_norm == oracle_norm
    
    def compare_coords(self,
                      predicted: Tuple[float, float],
                      oracle: Tuple[float, float]) -> bool:
        """
        Compare coordinate pairs.
        
        Returns True if distance <= threshold.
        """
        if predicted is None or oracle is None:
            return False
        
        px, py = predicted
        ox, oy = oracle
        
        # Euclidean distance
        distance = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
        
        return distance <= self.bbox_threshold
    
    def compare_bbox(self,
                    predicted: List[float],
                    oracle: List[float]) -> bool:
        """
        Compare bounding boxes.
        
        Uses centroid distance as the metric.
        """
        if predicted is None or oracle is None:
            return False
        
        if len(predicted) != 4 or len(oracle) != 4:
            return False
        
        # Calculate centroids
        pred_cx = predicted[0] + predicted[2] / 2
        pred_cy = predicted[1] + predicted[3] / 2
        
        oracle_cx = oracle[0] + oracle[2] / 2
        oracle_cy = oracle[1] + oracle[3] / 2
        
        return self.compare_coords((pred_cx, pred_cy), (oracle_cx, oracle_cy))
    
    def compare_targets(self,
                       predicted_target: Optional[Dict],
                       oracle_target: Optional[Dict]) -> bool:
        """
        Compare targets (bbox, selector, or coords).
        
        Returns True if targets match according to their type.
        """
        # Both None
        if predicted_target is None and oracle_target is None:
            return True
        
        # One is None
        if predicted_target is None or oracle_target is None:
            return False
        
        # Try selector match first
        pred_selector = predicted_target.get("selector")
        oracle_selector = oracle_target.get("selector")
        
        if pred_selector and oracle_selector:
            return self.compare_selectors(pred_selector, oracle_selector)
        
        # Try bbox match
        pred_bbox = predicted_target.get("bbox")
        oracle_bbox = oracle_target.get("bbox")
        
        if pred_bbox and oracle_bbox:
            return self.compare_bbox(pred_bbox, oracle_bbox)
        
        # Try coords match
        pred_coords = predicted_target.get("coords")
        oracle_coords = oracle_target.get("coords")
        
        if pred_coords and oracle_coords:
            return self.compare_coords(
                tuple(pred_coords),
                tuple(oracle_coords)
            )
        
        # No matching type found
        return False
    
    def compare_values(self,
                      predicted: Optional[str],
                      oracle: Optional[str]) -> bool:
        """
        Compare values for type actions.
        
        Uses exact string matching (case-sensitive).
        """
        if predicted is None and oracle is None:
            return True
        
        if predicted is None or oracle is None:
            return False
        
        return predicted == oracle
    
    def evaluate_single(self,
                       predicted_action: Dict,
                       oracle_action: Dict) -> Dict:
        """
        Evaluate a single prediction against oracle.
        
        Args:
            predicted_action: Predicted action dict with:
                - action_type
                - target (optional)
                - value (optional)
            oracle_action: Oracle action dict with same structure
        
        Returns:
            {
                "action_match": bool,
                "target_match": bool,
                "value_match": bool,
                "overall_match": bool
            }
        """
        # Compare action types
        action_match = self.compare_action_types(
            predicted_action.get("action_type", ""),
            oracle_action.get("action_type", "")
        )
        
        # Compare targets
        target_match = self.compare_targets(
            predicted_action.get("target"),
            oracle_action.get("target")
        )
        
        # Compare values (only for type actions)
        value_match = True  # Default to True if not applicable
        if oracle_action.get("action_type", "").lower() == "type":
            value_match = self.compare_values(
                predicted_action.get("value"),
                oracle_action.get("value")
            )
        
        # Overall match: all components must match
        overall_match = action_match and target_match and value_match
        
        return {
            "action_match": action_match,
            "target_match": target_match,
            "value_match": value_match,
            "overall_match": overall_match
        }
    
    def add_result(self,
                  predicted_action: Dict,
                  oracle_action: Dict,
                  metadata: Optional[Dict] = None):
        """
        Add a single evaluation result to the accumulator.
        
        Args:
            predicted_action: Predicted action dict
            oracle_action: Oracle action dict
            metadata: Optional metadata about the sample
        """
        eval_result = self.evaluate_single(predicted_action, oracle_action)
        
        result_entry = {
            "predicted": predicted_action,
            "oracle": oracle_action,
            "evaluation": eval_result,
            "metadata": metadata or {}
        }
        
        self.results.append(result_entry)
    
    def compute_metrics(self) -> EvaluationMetrics:
        """
        Compute aggregate metrics from accumulated results.
        
        Returns:
            EvaluationMetrics object
        """
        if not self.results:
            return EvaluationMetrics()
        
        total = len(self.results)
        correct_actions = sum(1 for r in self.results if r["evaluation"]["action_match"])
        correct_targets = sum(1 for r in self.results if r["evaluation"]["target_match"])
        correct_values = sum(1 for r in self.results if r["evaluation"]["value_match"])
        
        metrics = EvaluationMetrics(
            total_samples=total,
            correct_actions=correct_actions,
            correct_targets=correct_targets,
            correct_values=correct_values,
            action_accuracy=correct_actions / total if total > 0 else 0.0,
            target_accuracy=correct_targets / total if total > 0 else 0.0,
            value_accuracy=correct_values / total if total > 0 else 0.0
        )
        
        return metrics
    
    def get_failure_cases(self, 
                         failure_type: str = "action") -> List[Dict]:
        """
        Get samples where prediction failed.
        
        Args:
            failure_type: "action", "target", "value", or "overall"
        
        Returns:
            List of result dicts where the specified type failed
        """
        field_map = {
            "action": "action_match",
            "target": "target_match",
            "value": "value_match",
            "overall": "overall_match"
        }
        
        field = field_map.get(failure_type, "overall_match")
        
        return [r for r in self.results if not r["evaluation"][field]]
    
    def reset(self):
        """Clear all accumulated results."""
        self.results = []
