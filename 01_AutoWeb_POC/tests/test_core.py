"""
Lightweight tests for SeeAct components that don't require external dependencies.
Tests core logic without PIL, torch, transformers.
"""

import sys
from pathlib import Path
import json

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from action_schema import ActionPrediction, ActionSchema, create_action_prediction
from json_repair import JSONRepair, ActionJSONRepair
from action_decoder import ActionDecoder


def test_action_schema():
    """Test action schema validation."""
    print("\n=== Testing Action Schema ===")
    
    # Valid action
    action = create_action_prediction(
        action_type="click",
        target={"selector": "#login-button"},
        value=None,
        confidence=0.95
    )
    assert action.action_type == "click"
    print("✓ Valid click action created")
    
    # Valid type action
    action = create_action_prediction(
        action_type="type",
        target={"selector": "#email-input"},
        value="test@example.com",
        confidence=0.9
    )
    assert action.value == "test@example.com"
    print("✓ Valid type action created")
    
    # Test to_json method
    json_str = action.to_json()
    parsed = json.loads(json_str)
    assert parsed["action_type"] == "type"
    assert parsed["value"] == "test@example.com"
    print("✓ JSON serialization works")
    
    # Invalid action type should raise error
    try:
        create_action_prediction(
            action_type="invalid_action",
            target=None,
            value=None,
            confidence=0.5
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Invalid action type rejected")
    
    # Test validation
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": {"coords": [100, 200]},
        "value": None,
        "confidence": 0.8
    })
    assert valid, f"Should be valid: {error}"
    print("✓ Schema validation works")
    
    # Test invalid confidence
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": {"coords": [100, 200]},
        "value": None,
        "confidence": 1.5  # Invalid
    })
    assert not valid
    print("✓ Invalid confidence rejected")


def test_json_repair():
    """Test JSON repair functionality."""
    print("\n=== Testing JSON Repair ===")
    
    # Test 1: Valid JSON
    valid_json = '{"action_type": "click", "target": null, "value": null, "confidence": 0.9}'
    parsed, error = JSONRepair.repair_and_parse(valid_json)
    assert parsed is not None
    assert parsed["action_type"] == "click"
    print("✓ Valid JSON parsed correctly")
    
    # Test 2: JSON with markdown
    markdown_json = '''```json
    {
        "action_type": "type",
        "target": {"selector": "#input"},
        "value": "test",
        "confidence": 0.8
    }
    ```'''
    parsed, error = JSONRepair.repair_and_parse(markdown_json)
    assert parsed is not None
    assert parsed["action_type"] == "type"
    print("✓ Markdown-wrapped JSON repaired")
    
    # Test 3: JSON with trailing comma
    trailing_comma = '{"action_type": "click", "target": null,}'
    parsed, error = JSONRepair.repair_and_parse(trailing_comma)
    assert parsed is not None
    print("✓ Trailing comma fixed")
    
    # Test 4: JSON embedded in text
    embedded = 'Here is the action: {"action_type": "scroll", "target": null, "value": null, "confidence": 0.7} done.'
    parsed, error = JSONRepair.repair_and_parse(embedded)
    assert parsed is not None
    assert parsed["action_type"] == "scroll"
    print("✓ Embedded JSON extracted")
    
    # Test 5: Action JSON repair with defaults
    incomplete = '{"action_type": "click"}'
    parsed, error = ActionJSONRepair.repair_action_json(incomplete)
    assert parsed is not None
    assert parsed["target"] is None  # Default added
    assert parsed["confidence"] == 0.5  # Default added
    print("✓ Incomplete action JSON repaired with defaults")
    
    # Test 6: Empty input
    parsed, error = JSONRepair.repair_and_parse("")
    assert parsed is None
    assert error is not None
    print("✓ Empty input handled correctly")


def test_action_decoder():
    """Test action decoder."""
    print("\n=== Testing Action Decoder ===")
    
    decoder = ActionDecoder(image_width=1280, image_height=720)
    
    # Test 1: Valid JSON
    raw = '{"action_type": "click", "target": {"coords": [640, 360]}, "value": null, "confidence": 0.9}'
    prediction, error = decoder.decode(raw)
    assert prediction is not None
    assert prediction.action_type == "click"
    assert error is None
    print("✓ Valid action decoded")
    
    # Test 2: Normalized coordinates (0-1 range)
    raw = '{"action_type": "click", "target": {"coords": [0.5, 0.5]}, "value": null, "confidence": 0.9}'
    prediction, error = decoder.decode(raw)
    assert prediction is not None
    # Coordinates should be scaled
    target_coords = prediction.target["coords"]
    assert target_coords[0] == 640  # 0.5 * 1280
    assert target_coords[1] == 360  # 0.5 * 720
    print("✓ Normalized coordinates scaled correctly")
    
    # Test 3: Malformed JSON (lenient mode)
    decoder_lenient = ActionDecoder(image_width=1280, image_height=720, strict_validation=False)
    raw = 'action is click on button'  # Not JSON
    prediction, error = decoder_lenient.decode(raw)
    assert prediction is not None  # Should return noop
    assert prediction.action_type == "noop"
    print("✓ Malformed JSON handled gracefully (noop)")
    
    # Test 4: Strict mode with invalid JSON
    decoder_strict = ActionDecoder(image_width=1280, image_height=720, strict_validation=True)
    raw = 'invalid json here'
    prediction, error = decoder_strict.decode(raw)
    assert prediction is None
    assert error is not None
    print("✓ Strict mode rejects invalid JSON")


def test_coordinate_normalization():
    """Test coordinate normalization."""
    print("\n=== Testing Coordinate Normalization ===")
    
    decoder = ActionDecoder(image_width=1920, image_height=1080)
    
    # Test bbox normalization
    target = {"bbox": [0.5, 0.5, 0.1, 0.1]}  # Normalized
    normalized = decoder.normalize_target(target)
    
    expected_bbox = [960.0, 540.0, 192.0, 108.0]  # Scaled to pixel coords
    assert normalized["bbox"] == expected_bbox
    print(f"✓ BBox normalized: {target['bbox']} -> {normalized['bbox']}")
    
    # Test coords normalization
    target = {"coords": [0.25, 0.75]}  # Normalized
    normalized = decoder.normalize_target(target)
    
    expected_coords = [480.0, 810.0]  # Scaled to pixel coords
    assert normalized["coords"] == expected_coords
    print(f"✓ Coords normalized: {target['coords']} -> {normalized['coords']}")
    
    # Test pixel coords (should not be modified)
    target = {"coords": [100, 200]}  # Already in pixels
    normalized = decoder.normalize_target(target)
    assert normalized["coords"] == [100, 200]
    print("✓ Pixel coords unchanged")


def test_target_schema_variations():
    """Test different target format variations."""
    print("\n=== Testing Target Schema Variations ===")
    
    # Test selector target
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": {"selector": "#button"},
        "value": None,
        "confidence": 0.9
    })
    assert valid
    print("✓ Selector target valid")
    
    # Test coords target
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": {"coords": [100, 200]},
        "value": None,
        "confidence": 0.9
    })
    assert valid
    print("✓ Coords target valid")
    
    # Test bbox target
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": {"bbox": [10, 20, 100, 50]},
        "value": None,
        "confidence": 0.9
    })
    assert valid
    print("✓ BBox target valid")
    
    # Test null target for noop
    valid, error = ActionSchema.validate_prediction({
        "action_type": "noop",
        "target": None,
        "value": None,
        "confidence": 1.0
    })
    assert valid
    print("✓ Null target valid for noop")
    
    # Test invalid: missing target for click
    valid, error = ActionSchema.validate_prediction({
        "action_type": "click",
        "target": None,
        "value": None,
        "confidence": 0.9
    })
    assert not valid
    print("✓ Invalid: missing target for click rejected")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("Running SeeAct Pipeline Unit Tests (Lightweight)")
    print("=" * 80)
    
    tests = [
        test_action_schema,
        test_json_repair,
        test_action_decoder,
        test_coordinate_normalization,
        test_target_schema_variations
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test failed: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test error: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    print()
    
    if failed == 0:
        print("✅ All tests passed! Core components are working correctly.")
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download Qwen2-VL-2B model")
        print("  3. Run the complete pipeline with: python src/seeact_pipeline.py")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
