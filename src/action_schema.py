"""
Action Schema Definition and Validation

This module defines the strict output contract for SeeAct-style action prediction.
The schema ensures controlled, structured output that prevents hallucination.
"""

from typing import Dict, Optional, Union, List, Literal
from dataclasses import dataclass, asdict
import json


# Allowed action types (SeeAct-style)
ActionType = Literal["click", "type", "scroll", "noop"]


@dataclass
class BBoxTarget:
    """Bounding box target coordinates (normalized 0-1 or pixel coords)"""
    x: float
    y: float
    width: float
    height: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SelectorTarget:
    """CSS selector target"""
    selector: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CoordsTarget:
    """Coordinate-based target (x, y)"""
    x: float
    y: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ActionPrediction:
    """
    The canonical action prediction format.
    
    This is the OUTPUT CONTRACT for the SeeAct-style predictor.
    All model outputs must be parsed into this format.
    
    Fields:
        action_type: one of "click", "type", "scroll", "noop"
        target: dictionary with either:
                - {"bbox": [x, y, w, h]} for bounding box
                - {"selector": "css selector"} for CSS selector
                - {"coords": [x, y]} for coordinate click
                - null for noop
        value: string value for type actions, null otherwise
        confidence: float between 0.0 and 1.0
    """
    action_type: ActionType
    target: Optional[Union[Dict, None]]
    value: Optional[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "value": self.value,
            "confidence": self.confidence
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ActionPrediction":
        """Create ActionPrediction from dictionary"""
        return cls(
            action_type=data.get("action_type", "noop"),
            target=data.get("target"),
            value=data.get("value"),
            confidence=data.get("confidence", 0.0)
        )


class ActionSchema:
    """
    Schema validation and conversion utilities for action predictions.
    """
    
    VALID_ACTION_TYPES = ["click", "type", "scroll", "noop"]
    
    @staticmethod
    def validate_action_type(action_type: str) -> bool:
        """Check if action type is valid"""
        return action_type in ActionSchema.VALID_ACTION_TYPES
    
    @staticmethod
    def validate_target(target: Optional[Dict], action_type: str) -> bool:
        """
        Validate target format.
        
        Rules:
        - noop: target must be None or null
        - click/type: target must have bbox, selector, or coords
        - scroll: target can be None (scroll page) or have direction
        """
        if action_type == "noop":
            return target is None or target == {}
        
        if target is None:
            # scroll can have no target (scroll entire page)
            if action_type == "scroll":
                return True
            return False
        
        # Check for valid target types
        has_bbox = "bbox" in target
        has_selector = "selector" in target
        has_coords = "coords" in target
        
        return has_bbox or has_selector or has_coords
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Check if confidence is in valid range [0.0, 1.0]"""
        return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    
    @staticmethod
    def validate_prediction(pred: Union[Dict, ActionPrediction]) -> tuple[bool, Optional[str]]:
        """
        Validate a complete action prediction.
        
        Returns:
            (is_valid, error_message)
        """
        if isinstance(pred, ActionPrediction):
            pred = pred.to_dict()
        
        # Check required fields
        if "action_type" not in pred:
            return False, "Missing required field: action_type"
        
        action_type = pred["action_type"]
        
        # Validate action type
        if not ActionSchema.validate_action_type(action_type):
            return False, f"Invalid action_type: {action_type}. Must be one of {ActionSchema.VALID_ACTION_TYPES}"
        
        # Validate target
        target = pred.get("target")
        if not ActionSchema.validate_target(target, action_type):
            return False, f"Invalid target for action_type={action_type}"
        
        # Validate confidence
        confidence = pred.get("confidence", 0.0)
        if not ActionSchema.validate_confidence(confidence):
            return False, f"Invalid confidence: {confidence}. Must be float in [0.0, 1.0]"
        
        # Validate value for type actions
        if action_type == "type":
            value = pred.get("value")
            if value is None or value == "":
                return False, "type action requires non-empty value field"
        
        return True, None
    
    @staticmethod
    def normalize_target(target: Optional[Dict]) -> Optional[Dict]:
        """
        Normalize target format to ensure consistency.
        
        Examples:
            {"bbox": "10,20,100,50"} -> {"bbox": [10, 20, 100, 50]}
            {"coords": "100,200"} -> {"coords": [100, 200]}
        """
        if target is None:
            return None
        
        result = {}
        
        # Normalize bbox
        if "bbox" in target:
            bbox = target["bbox"]
            if isinstance(bbox, str):
                # Parse string format
                try:
                    coords = [float(x.strip()) for x in bbox.split(",")]
                    result["bbox"] = coords
                except Exception:
                    result["bbox"] = bbox  # Keep as-is if parsing fails
            elif isinstance(bbox, (list, tuple)):
                result["bbox"] = list(bbox)
            else:
                result["bbox"] = bbox
        
        # Normalize coords
        if "coords" in target:
            coords = target["coords"]
            if isinstance(coords, str):
                try:
                    xy = [float(x.strip()) for x in coords.split(",")]
                    result["coords"] = xy
                except Exception:
                    result["coords"] = coords
            elif isinstance(coords, (list, tuple)):
                result["coords"] = list(coords)
            else:
                result["coords"] = coords
        
        # Keep selector as-is
        if "selector" in target:
            result["selector"] = target["selector"]
        
        return result if result else None


def create_action_prediction(
    action_type: str,
    target: Optional[Dict] = None,
    value: Optional[str] = None,
    confidence: float = 1.0
) -> ActionPrediction:
    """
    Factory function to create a validated ActionPrediction.
    
    Raises ValueError if validation fails.
    """
    pred = ActionPrediction(
        action_type=action_type,
        target=target,
        value=value,
        confidence=confidence
    )
    
    is_valid, error = ActionSchema.validate_prediction(pred)
    if not is_valid:
        raise ValueError(f"Invalid action prediction: {error}")
    
    return pred
