# SeeAct Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SeeAct Single-Step Pipeline                      │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Screenshot (PIL.Image) + Instruction (str)
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. IMAGE PREPROCESSOR (image_preprocessor.py)                      │
│    • Resize with aspect ratio (1280x720)                           │
│    • NO OCR (vision-only)                                           │
│    • NO DOM parsing (intentionally limited)                         │
│    • Store scale_info for coordinate transformation                │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Outputs: processed_image, scale_info
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. PROMPT ENGINE (prompt_engine.py)                                │
│    • Strict JSON-only template                                     │
│    • No explanations allowed                                       │
│    • Clear schema specification:                                   │
│      {action_type, target, value, confidence}                      │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Outputs: prompt_text + image
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. MODEL INTERFACE (model_interface.py)                            │
│    • Qwen2-VL-2B-Instruct (local)                                  │
│    • Vision-language model                                         │
│    • Inference with temperature control                            │
│    • Latency tracking                                              │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Outputs: raw_text, latency
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. JSON REPAIR LAYER (json_repair.py)                              │
│    • Extract JSON from surrounding text                            │
│    • Fix markdown wrappers (```json ... ```)                       │
│    • Remove trailing commas                                        │
│    • Add missing default fields                                    │
│    • Handle malformed brackets                                     │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Outputs: parsed_json or error
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. ACTION DECODER (action_decoder.py)                              │
│    • Parse JSON to dict                                            │
│    • Normalize coordinates:                                        │
│      [0-1] range → pixel coordinates                               │
│    • Validate against schema                                       │
│    • Create ActionPrediction object                                │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Outputs: ActionPrediction
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. SCHEMA VALIDATION (action_schema.py)                            │
│    • Check action_type in [click, type, scroll, noop]             │
│    • Validate target format (selector/coords/bbox/null)            │
│    • Ensure confidence in [0.0, 1.0]                               │
│    • Verify value present for type actions                         │
└─────────────────────────────────────────────────────────────────────┘
   │
   │
   ▼
OUTPUT: ActionPrediction
{
  "action_type": "click|type|scroll|noop",
  "target": {"selector": "..."} OR {"coords": [x,y]} OR {"bbox": [x,y,w,h]},
  "value": "text" OR null,
  "confidence": 0.0-1.0
}


┌─────────────────────────────────────────────────────────────────────┐
│                       Optional: Evaluation                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DATASET ADAPTER (mind2Web_Loader.py)                               │
│    • Load Mind2Web samples                                         │
│    • Extract oracle_action (ground truth)                          │
│    • Format: {image, instruction, oracle_action}                   │
└─────────────────────────────────────────────────────────────────────┘
   │
   │ Provides oracle for comparison
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EVALUATOR (evaluator.py)                                           │
│    • Compare predicted vs oracle                                   │
│    • Compute Action Accuracy (AA)                                  │
│    • Compute Target Accuracy (TA)                                  │
│    • Compute Value Accuracy (VA)                                   │
│    • Track failure cases                                           │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                         Component Details
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ ACTION SCHEMA                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ • ActionPrediction dataclass                                        │
│ • ActionSchema validator                                            │
│ • create_action_prediction() factory                                │
│                                                                      │
│ Ensures:                                                            │
│   ✓ action_type is valid                                           │
│   ✓ target format matches action                                   │
│   ✓ confidence in [0, 1]                                           │
│   ✓ value present for type actions                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ JSON REPAIR                                                          │
├─────────────────────────────────────────────────────────────────────┤
│ • extract_json_from_text()                                          │
│ • fix_quotes()                                                      │
│ • remove_trailing_commas()                                          │
│ • fix_common_issues()                                               │
│ • repair_and_parse() - multi-strategy                               │
│                                                                      │
│ Handles:                                                            │
│   ✓ Markdown wrappers                                              │
│   ✓ Embedded in text                                               │
│   ✓ Trailing commas                                                │
│   ✓ Missing brackets                                               │
│   ✓ Invalid escape sequences                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ACTION DECODER                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ • decode() - main entry point                                       │
│ • normalize_target() - coordinate scaling                           │
│ • normalize_bbox_coords()                                           │
│ • normalize_coords()                                                │
│                                                                      │
│ Features:                                                           │
│   ✓ Automatic coordinate scaling                                   │
│   ✓ Detects [0-1] range vs pixels                                  │
│   ✓ Strict/lenient modes                                           │
│   ✓ Graceful fallback to noop                                      │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                          Data Flow Example
═══════════════════════════════════════════════════════════════════════

Input Image (1920x1080)
     +
"Click the login button"
     │
     ▼
Preprocessed (1280x720, scale=0.667)
     +
"You are a UI action predictor..."
     │
     ▼
Model generates:
"```json\n{\"action_type\": \"click\", ...}\n```"
     │
     ▼
JSON Repair extracts:
{"action_type": "click", "target": {"coords": [0.5, 0.5]}}
     │
     ▼
Decoder normalizes:
{"action_type": "click", "target": {"coords": [640, 360]}}
     │
     ▼
Validation passes ✓
     │
     ▼
ActionPrediction(
  action_type="click",
  target={"coords": [640, 360]},
  value=None,
  confidence=0.95
)


═══════════════════════════════════════════════════════════════════════
                        Error Handling Flow
═══════════════════════════════════════════════════════════════════════

Model Output: "I think you should click the button"
     │
     ▼
JSON Repair: Cannot extract valid JSON
     │
     ▼
Decoder (lenient mode): Create noop action
     │
     ▼
Output: ActionPrediction(action_type="noop", confidence=0.0)

                    OR (strict mode)

     │
     ▼
Decoder (strict mode): Return error
     │
     ▼
Output: None, "JSON parsing failed: ..."


═══════════════════════════════════════════════════════════════════════
                         Testing Strategy
═══════════════════════════════════════════════════════════════════════

test_core.py (No dependencies)
├── test_action_schema()
│   ├── Valid actions ✓
│   ├── Invalid actions ✗
│   └── Schema validation ✓
├── test_json_repair()
│   ├── Valid JSON ✓
│   ├── Markdown wrapped ✓
│   ├── Trailing commas ✓
│   ├── Embedded text ✓
│   └── Missing fields ✓
├── test_action_decoder()
│   ├── Valid decoding ✓
│   ├── Coordinate normalization ✓
│   └── Error handling ✓
├── test_coordinate_normalization()
│   ├── BBox scaling ✓
│   ├── Coords scaling ✓
│   └── Pixel preservation ✓
└── test_target_schema_variations()
    ├── Selector ✓
    ├── Coords ✓
    ├── BBox ✓
    └── Null ✓

mock_demo.py (Mock model outputs)
├── Click with selector ✓
├── Type with value ✓
├── Scroll action ✓
├── Normalized coordinates ✓
├── Bounding box ✓
├── Noop action ✓
├── Malformed JSON ✓
└── Embedded JSON ✓


═══════════════════════════════════════════════════════════════════════
                       Success Criteria ✅
═══════════════════════════════════════════════════════════════════════

✓ Input: Screenshot + Instruction
✓ Minimal Preprocessing (SeeAct-style)
✓ Strict Prompt (JSON-only)
✓ Local Inference (Qwen2-VL-2B ready)
✓ JSON Repair (8+ scenarios)
✓ Output: Valid JSON
✓ Schema Validation
✓ Coordinate Normalization
✓ Error Handling
✓ Tests Passing (5/5)
✓ Mock Demo Successful (8/8)

STOPPING CRITERION: ✅ MET
"I can give a screenshot + instruction and my system outputs 
 a valid JSON action prediction locally."
```
