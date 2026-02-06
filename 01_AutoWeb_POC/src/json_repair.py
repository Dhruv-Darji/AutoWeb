"""
JSON Repair Layer

Models frequently output malformed JSON. This module provides utilities to:
1. Clean and repair common JSON formatting issues
2. Extract JSON from text with surrounding content
3. Retry parsing with different strategies
"""

import json
import re
from typing import Optional, Dict, Any, Tuple


class JSONRepair:
    """
    Repair malformed JSON outputs from language models.
    
    Common issues addressed:
    - Missing/extra commas
    - Unescaped quotes
    - Trailing commas
    - Missing brackets/braces
    - Text before/after JSON
    - Single quotes instead of double quotes
    """
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract JSON object from text that may contain other content.
        
        Looks for patterns like:
        - {...}
        - [{...}]
        
        Returns the extracted JSON string or None if not found.
        """
        if not text or not isinstance(text, str):
            return None
        
        # Try to find JSON object
        # Look for outermost { ... }
        brace_start = text.find('{')
        if brace_start == -1:
            return None
        
        # Count braces to find matching closing brace
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    return text[brace_start:i+1]
        
        # No matching brace found, try simpler extraction
        # Look for {...} pattern using regex
        match = re.search(r'\{[^}]*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None
    
    @staticmethod
    def fix_quotes(text: str) -> str:
        """
        Replace single quotes with double quotes for JSON compatibility.
        Be careful not to replace quotes inside strings.
        """
        # Simple approach: replace single quotes with double quotes
        # This is a heuristic and may not work for all cases
        
        # First, protect escaped quotes
        text = text.replace("\\'", "<<<ESCAPED_SINGLE>>>")
        text = text.replace('\\"', "<<<ESCAPED_DOUBLE>>>")
        
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        
        # Restore escaped quotes
        text = text.replace("<<<ESCAPED_SINGLE>>>", "\\'")
        text = text.replace("<<<ESCAPED_DOUBLE>>>", '\\"')
        
        return text
    
    @staticmethod
    def remove_trailing_commas(text: str) -> str:
        """Remove trailing commas before closing brackets/braces."""
        # Remove comma before }
        text = re.sub(r',\s*}', '}', text)
        # Remove comma before ]
        text = re.sub(r',\s*]', ']', text)
        return text
    
    @staticmethod
    def fix_common_issues(text: str) -> str:
        """Apply common fixes to malformed JSON."""
        if not text:
            return text
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove markdown code block markers
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Fix trailing commas
        text = JSONRepair.remove_trailing_commas(text)
        
        # Fix missing commas between fields (common error)
        # This is risky and may break valid JSON, so use carefully
        # text = re.sub(r'"\s*\n\s*"', '",\n"', text)
        
        return text
    
    @staticmethod
    def repair_and_parse(text: str, max_attempts: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Attempt to repair and parse JSON from text.
        
        Returns:
            (parsed_dict, error_message)
            If successful: (dict, None)
            If failed: (None, error_message)
        """
        if not text:
            return None, "Empty input text"
        
        # Strategy 1: Try parsing as-is
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            last_error = str(e)
        
        # Strategy 2: Extract JSON and try again
        extracted = JSONRepair.extract_json_from_text(text)
        if extracted:
            try:
                return json.loads(extracted), None
            except json.JSONDecodeError as e:
                last_error = str(e)
        
        # Strategy 3: Apply fixes and try again
        if extracted:
            text_to_fix = extracted
        else:
            text_to_fix = text
        
        fixed = JSONRepair.fix_common_issues(text_to_fix)
        try:
            return json.loads(fixed), None
        except json.JSONDecodeError as e:
            last_error = str(e)
        
        # Strategy 4: Try fixing quotes
        fixed_quotes = JSONRepair.fix_quotes(fixed)
        try:
            return json.loads(fixed_quotes), None
        except json.JSONDecodeError as e:
            last_error = str(e)
        
        # Strategy 5: Try aggressive extraction with regex
        # Look for key-value pairs
        try:
            # Try to build a minimal valid JSON from detected patterns
            action_match = re.search(r'"action(?:_type)?"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
            confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text, re.IGNORECASE)
            value_match = re.search(r'"value"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
            
            if action_match:
                # Build minimal valid JSON
                result = {
                    "action_type": action_match.group(1),
                    "target": None,
                    "value": value_match.group(1) if value_match else None,
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.5
                }
                return result, None
        except Exception as e:
            last_error = f"Aggressive extraction failed: {str(e)}"
        
        return None, f"Failed to parse JSON after {max_attempts} attempts. Last error: {last_error}"
    
    @staticmethod
    def validate_and_repair(
        text: str,
        required_keys: Optional[list] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Parse JSON and validate it has required keys.
        
        Args:
            text: Input text containing JSON
            required_keys: List of required keys (e.g., ['action_type', 'target'])
        
        Returns:
            (parsed_dict, error_message)
        """
        parsed, error = JSONRepair.repair_and_parse(text)
        
        if parsed is None:
            return None, error
        
        # Validate required keys
        if required_keys:
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                return parsed, f"Missing required keys: {missing}"
        
        return parsed, None


class ActionJSONRepair(JSONRepair):
    """
    Specialized JSON repair for action predictions.
    Adds domain-specific fixes and defaults.
    """
    
    REQUIRED_KEYS = ["action_type"]
    OPTIONAL_KEYS = ["target", "value", "confidence"]
    
    @staticmethod
    def add_defaults(parsed: Dict) -> Dict:
        """Add default values for missing optional fields."""
        result = parsed.copy()
        
        # Add defaults
        if "target" not in result:
            result["target"] = None
        
        if "value" not in result:
            result["value"] = None
        
        if "confidence" not in result:
            result["confidence"] = 0.5
        
        # Normalize action_type
        if "action" in result and "action_type" not in result:
            result["action_type"] = result["action"]
        
        return result
    
    @staticmethod
    def repair_action_json(text: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Repair and validate action prediction JSON.
        
        Returns:
            (parsed_dict_with_defaults, error_message)
        """
        parsed, error = JSONRepair.validate_and_repair(
            text,
            required_keys=["action_type"]  # Only action_type is truly required
        )
        
        if parsed is None:
            return None, error
        
        # Add defaults for optional fields
        result = ActionJSONRepair.add_defaults(parsed)
        
        return result, None
