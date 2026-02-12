"""
Action Decoder

Converts raw model output to validated ActionPrediction objects.
Handles:
1. JSON repair and parsing
2. Schema validation
3. Coordinate normalization
4. Target formatting
"""

from typing import Dict, Optional, Tuple

try:
    from .action_schema import ActionPrediction, ActionSchema, create_action_prediction
    from .json_repair import ActionJSONRepair
except ImportError:
    from action_schema import ActionPrediction, ActionSchema, create_action_prediction
    from json_repair import ActionJSONRepair


class ActionDecoder:
    """
    Decode and normalize model outputs into ActionPrediction objects.
    
    This is the critical layer that converts raw text from the model
    into structured, validated action predictions.
    """
    
    def __init__(self, 
                 image_width: Optional[int] = None,
                 image_height: Optional[int] = None,
                 strict_validation: bool = True):
        """
        Args:
            image_width: Original image width for coordinate normalization
            image_height: Original image height for coordinate normalization
            strict_validation: If True, raises exception on validation failure
        """
        self.image_width = image_width
        self.image_height = image_height
        self.strict_validation = strict_validation
    
    def set_image_dimensions(self, width: int, height: int):
        """Update image dimensions for coordinate normalization."""
        self.image_width = width
        self.image_height = height
    
    def normalize_bbox_coords(self, bbox: list, 
                             from_normalized: bool = False) -> list:
        """
        Normalize bbox coordinates.
        
        Args:
            bbox: [x, y, width, height] or [x1, y1, x2, y2]
            from_normalized: If True, bbox is in [0-1] range and needs scaling
        
        Returns:
            [x, y, width, height] in pixel coordinates
        """
        if not self.image_width or not self.image_height:
            # Can't normalize without image dimensions
            return bbox
        
        if len(bbox) != 4:
            return bbox
        
        x, y, w, h = bbox
        
        if from_normalized:
            # Scale from [0-1] to pixel coordinates
            x = x * self.image_width
            y = y * self.image_height
            w = w * self.image_width
            h = h * self.image_height
        
        return [x, y, w, h]
    
    def normalize_coords(self, coords: list,
                        from_normalized: bool = False) -> list:
        """
        Normalize coordinate pair.
        
        Args:
            coords: [x, y]
            from_normalized: If True, coords are in [0-1] range
        
        Returns:
            [x, y] in pixel coordinates
        """
        if not self.image_width or not self.image_height:
            return coords
        
        if len(coords) != 2:
            return coords
        
        x, y = coords
        
        if from_normalized:
            x = x * self.image_width
            y = y * self.image_height
        
        return [x, y]
    
    def normalize_target(self, target: Optional[Dict]) -> Optional[Dict]:
        """
        Normalize target coordinates if present.
        
        Handles:
        - bbox: normalize to pixel coordinates
        - coords: normalize to pixel coordinates
        - selector: pass through as-is
        """
        if target is None:
            return None
        
        normalized = target.copy()
        
        # Check if coordinates are normalized (values between 0 and 1)
        def is_normalized(values):
            return all(0 <= v <= 1 for v in values if isinstance(v, (int, float)))
        
        # Normalize bbox
        if "bbox" in normalized:
            bbox = normalized["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                from_norm = is_normalized(bbox)
                normalized["bbox"] = self.normalize_bbox_coords(list(bbox), from_norm)
        
        # Normalize coords
        if "coords" in normalized:
            coords = normalized["coords"]
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                from_norm = is_normalized(coords)
                normalized["coords"] = self.normalize_coords(list(coords), from_norm)
        
        return normalized
    
    def decode(self, raw_text: str) -> Tuple[Optional[ActionPrediction], Optional[str]]:
        """
        Decode raw model output into ActionPrediction.
        
        Args:
            raw_text: Raw text output from the model
        
        Returns:
            (ActionPrediction, error_message)
            If successful: (prediction, None)
            If failed: (None, error_message)
        """
        # Step 1: Repair and parse JSON
        parsed, error = ActionJSONRepair.repair_action_json(raw_text)
        
        if parsed is None:
            if self.strict_validation:
                return None, f"JSON parsing failed: {error}"
            else:
                # Return a default "noop" action
                return self._create_noop_action(), f"JSON parsing failed (using noop): {error}"
        
        # Step 2: Normalize target coordinates
        if "target" in parsed and parsed["target"] is not None:
            parsed["target"] = ActionSchema.normalize_target(parsed["target"])
            parsed["target"] = self.normalize_target(parsed["target"])
        
        # Step 3: Validate schema
        is_valid, validation_error = ActionSchema.validate_prediction(parsed)
        
        if not is_valid:
            if self.strict_validation:
                return None, f"Schema validation failed: {validation_error}"
            else:
                # Try to fix common issues
                parsed = self._fix_common_validation_errors(parsed)
                is_valid, validation_error = ActionSchema.validate_prediction(parsed)
                
                if not is_valid:
                    return None, f"Schema validation failed: {validation_error}"
        
        # Step 4: Create ActionPrediction object
        try:
            prediction = ActionPrediction.from_dict(parsed)
            return prediction, None
        except Exception as e:
            return None, f"Failed to create ActionPrediction: {str(e)}"
    
    def _create_noop_action(self) -> ActionPrediction:
        """Create a default noop action."""
        return ActionPrediction(
            action_type="noop",
            target=None,
            value=None,
            confidence=0.0
        )
    
    def _fix_common_validation_errors(self, parsed: Dict) -> Dict:
        """
        Attempt to fix common validation errors.
        
        Common fixes:
        - Convert invalid action types to "noop"
        - Add missing required fields
        - Clamp confidence to [0, 1]
        """
        result = parsed.copy()
        
        # Fix action_type
        if "action_type" not in result or result["action_type"] not in ActionSchema.VALID_ACTION_TYPES:
            # Try to infer from other fields
            if "action" in result:
                result["action_type"] = result["action"]
            else:
                result["action_type"] = "noop"
        
        # Clamp confidence
        if "confidence" in result:
            conf = result["confidence"]
            if isinstance(conf, (int, float)):
                result["confidence"] = max(0.0, min(1.0, float(conf)))
            else:
                result["confidence"] = 0.5
        
        # Ensure value is string or None
        if "value" in result and result["value"] is not None:
            if not isinstance(result["value"], str):
                result["value"] = str(result["value"])
        
        return result
    
    def decode_with_metadata(self, raw_text: str) -> Dict:
        """
        Decode and return both prediction and metadata.
        
        Returns:
            {
                "prediction": ActionPrediction or None,
                "raw_text": str,
                "parsed_json": Dict or None,
                "error": str or None,
                "success": bool
            }
        """
        prediction, error = self.decode(raw_text)
        
        # Try to get parsed JSON even if validation failed
        parsed, _ = ActionJSONRepair.repair_action_json(raw_text)
        
        return {
            "prediction": prediction,
            "raw_text": raw_text,
            "parsed_json": parsed,
            "error": error,
            "success": prediction is not None
        }


def decode_action(raw_text: str,
                 image_width: Optional[int] = None,
                 image_height: Optional[int] = None,
                 strict: bool = True) -> Tuple[Optional[ActionPrediction], Optional[str]]:
    """
    Convenience function to decode action from raw text.
    
    Args:
        raw_text: Raw model output
        image_width: Image width for coordinate normalization
        image_height: Image height for coordinate normalization
        strict: If True, raises exception on validation failure
    
    Returns:
        (ActionPrediction, error_message)
    """
    decoder = ActionDecoder(
        image_width=image_width,
        image_height=image_height,
        strict_validation=strict
    )
    return decoder.decode(raw_text)
