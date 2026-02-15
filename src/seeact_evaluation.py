"""
SeeAct Evaluation Module

Implements the evaluation metrics described in the SeeAct paper for offline
evaluation on the Multimodal-Mind2Web benchmark.

Metrics:
    Step-level:
        - Element Accuracy  (Ele. Acc)  : predicted element matches ground-truth element
        - Operation F1      (Op. F1)    : predicted action type matches oracle operation
        - Value Accuracy    (Val. Acc)  : for TYPE actions, predicted value matches oracle
        - Step Success Rate (Step SR)   : element + operation + value all correct

    Task-level:
        - Offline0 (Strict)    : task succeeds only if ALL steps succeed
        - Offline1 (Tolerance) : task succeeds if at most 1 step fails

Usage:
    evaluator = SeeActEvaluator(output_dir="eval_results")

    # inside the pipeline loop per step:
    evaluator.record_step(
        annotation_id=...,
        action_uid=...,
        ground_truth=action,        # dict from Mind2Web loader
        predicted_plan=output_plan,  # raw string from action generation
        grounding_result=result,     # dict from action grounding
    )

    # after all tasks:
    evaluator.compute_metrics()
    evaluator.save_results()
    evaluator.print_summary()
"""

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepEvalResult:
    """Evaluation result for a single action step."""

    # Identifiers
    annotation_id: str = ""
    action_uid: str = ""
    step_index: int = 0

    # Ground truth
    gt_operation: str = ""              # CLICK / TYPE / SELECT
    gt_value: str = ""                  # typed value (for TYPE ops)
    gt_element_repr: str = ""           # target_action_reprs from dataset
    gt_pos_backend_ids: List[str] = field(default_factory=list)  # all positive candidate backend_node_ids

    # Predictions
    pred_action_type: str = ""          # parsed from generated plan text
    pred_value: str = ""                # parsed typed value (if any)
    pred_plan_text: str = ""            # raw action generation output
    pred_element: Optional[Dict] = None # selected_element dict from grounding
    pred_backend_id: str = ""           # backend_node_id of selected element
    grounding_success: bool = False

    # Match flags
    operation_match: bool = False
    element_match: bool = False
    value_match: bool = False           # defaults True for non-TYPE ops
    step_success: bool = False          # all three match

    # Timing
    latency: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class TaskEvalResult:
    """Evaluation result for a full task (all steps)."""

    annotation_id: str = ""
    instruction: str = ""
    website: str = ""
    domain: str = ""
    total_steps: int = 0
    steps: List[StepEvalResult] = field(default_factory=list)

    # Task-level metrics (computed after all steps recorded)
    element_accuracy: float = 0.0
    operation_f1: float = 0.0
    value_accuracy: float = 0.0
    step_success_rate: float = 0.0
    offline0: bool = False              # strict: all steps correct
    offline1: bool = False              # tolerance: at most 1 step wrong

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all evaluated tasks."""

    total_tasks: int = 0
    total_steps: int = 0

    # Step-level averages
    element_accuracy: float = 0.0
    operation_f1: float = 0.0
    value_accuracy: float = 0.0
    step_success_rate: float = 0.0

    # Task-level rates
    offline0_rate: float = 0.0          # fraction of tasks passing strict
    offline1_rate: float = 0.0          # fraction of tasks passing tolerance

    # Counts
    correct_elements: int = 0
    correct_operations: int = 0
    correct_values: int = 0
    successful_steps: int = 0
    offline0_tasks: int = 0
    offline1_tasks: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SeeActEvaluator:
    """
    SeeAct offline evaluation on Multimodal-Mind2Web.

    Collects per-step predictions and ground truth, then computes
    Element Accuracy, Operation F1, Value Accuracy, Step SR, and
    Task SR (Offline0/Offline1).
    """

    def __init__(self, output_dir: str = "eval_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Internal storage keyed by annotation_id
        self._task_steps: Dict[str, List[StepEvalResult]] = defaultdict(list)
        self._task_meta: Dict[str, Dict] = {}

        # Computed after evaluate()
        self.task_results: Dict[str, TaskEvalResult] = {}
        self.aggregate: Optional[AggregateMetrics] = None

    # ------------------------------------------------------------------
    # Ground-truth extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_gt_operation(action: Dict) -> str:
        """Extract operation string from ground-truth action dict."""
        op = action.get("operation") or action.get("oracle_action", {})
        if isinstance(op, dict):
            return (op.get("op") or op.get("action_type") or "").upper().strip()
        if isinstance(op, str):
            try:
                parsed = json.loads(op)
                return (parsed.get("op") or "").upper().strip()
            except Exception:
                return op.upper().strip()
        return ""

    @staticmethod
    def _extract_gt_value(action: Dict) -> str:
        """Extract typed value from ground-truth action dict."""
        op = action.get("operation") or action.get("oracle_action", {})
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except Exception:
                return ""
        if isinstance(op, dict):
            return str(op.get("value") or op.get("action_input") or "").strip()
        return ""

    @staticmethod
    def _extract_gt_pos_backend_ids(action: Dict) -> List[str]:
        """Collect all backend_node_ids from pos_candidates."""
        candidates = action.get("candidates") or action.get("pos_candidates") or []
        ids = []
        for c in candidates:
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except Exception:
                    continue
            if isinstance(c, dict):
                bid = c.get("backend_node_id")
                if bid is not None:
                    ids.append(str(bid))
                # also check nested attributes
                attrs = c.get("attributes")
                if isinstance(attrs, dict):
                    bid2 = attrs.get("backend_node_id")
                    if bid2 is not None and str(bid2) not in ids:
                        ids.append(str(bid2))
        return ids

    @staticmethod
    def _extract_gt_element_repr(action: Dict) -> str:
        """Get a human-readable representation of the target element."""
        # Prefer target_action_reprs from dataset
        reprs = action.get("target_action_reprs") or action.get("action_reprs") or ""
        if isinstance(reprs, list):
            return " | ".join(str(r) for r in reprs)
        return str(reprs)

    # ------------------------------------------------------------------
    # Prediction parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action_type(plan_text: str) -> str:
        """Parse action type from the generated plan text.

        Handles formats like:
            'click: Click on X'
            'ACTION_TYPE: click'
            'ACTION_TYPE=click'
            'Action Type: click'
            'input: Enter text...'
        """
        if not plan_text:
            return ""

        text = plan_text.strip()

        # Pattern 1: ACTION_TYPE = <type> or ACTION_TYPE: <type> or ACTION_TYPE=<type>
        m = re.search(
            r"ACTION[_ ]?TYPE\s*[:=]\s*(click|input|type|select|scroll|finish|hover)",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).upper().strip()

        # Pattern 2: "Action Type: <type>" (with space)
        m = re.search(r"Action\s+Type\s*:\s*(click|input|type|select|scroll|finish|hover)", text, re.IGNORECASE)
        if m:
            return m.group(1).upper().strip()

        # Pattern 3: "<type>: <detail>" at start of line
        m = re.match(r"(click|input|type|select|scroll|finish|hover)\s*:", text, re.IGNORECASE)
        if m:
            return m.group(1).upper().strip()

        # Pattern 4: FINISH as standalone
        if re.match(r"\s*FINISH\s*$", text, re.IGNORECASE):
            return "FINISH"

        return ""

    @staticmethod
    def _parse_typed_value(plan_text: str) -> str:
        """Try to extract the typed value from an input/type plan."""
        if not plan_text:
            return ""
        # Look for TYPE: "value" or input: "value" or TYPE text into field and TYPE: "value"
        m = re.search(r'(?:input|type)\s*[:\-]\s*.*?"([^"]+)"', plan_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: after "TYPE:" take everything after the colon
        m = re.search(r'(?:input|type)\s*:\s*(.+)', plan_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_pred_backend_id(grounding_result: Optional[Dict]) -> str:
        """Get backend_node_id from the grounding result's selected element."""
        if not grounding_result or not isinstance(grounding_result, dict):
            return ""
        el = grounding_result.get("selected_element")
        if not el or not isinstance(el, dict):
            return ""
        # Direct field
        bid = el.get("backend_node_id")
        if bid is not None:
            return str(bid)
        # Inside attributes dict
        attrs = el.get("attributes")
        if isinstance(attrs, dict):
            bid = attrs.get("backend_node_id")
            if bid is not None:
                return str(bid)
        return ""

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    @staticmethod
    def _match_operation(pred_type: str, gt_type: str) -> bool:
        """Compare predicted action type against ground truth operation.

        Handles common synonyms:
            INPUT == TYPE
            SELECT == CLICK (SeeAct treats select as click)
        """
        p = pred_type.upper().strip()
        g = gt_type.upper().strip()

        if p == g:
            return True

        # Synonym groups
        synonyms = [
            {"CLICK", "SELECT", "HOVER"},
            {"INPUT", "TYPE"},
        ]
        for group in synonyms:
            if p in group and g in group:
                return True

        return False

    @staticmethod
    def _match_element(pred_backend_id: str, gt_pos_backend_ids: List[str]) -> bool:
        """Check if predicted element is among the positive candidates.

        Element accuracy in SeeAct: the predicted element's backend_node_id
        must appear in the set of ground-truth positive candidates.
        """
        if not pred_backend_id or not gt_pos_backend_ids:
            return False
        return pred_backend_id in gt_pos_backend_ids

    @staticmethod
    def _match_value(pred_value: str, gt_value: str, gt_op: str) -> bool:
        """Compare typed values.  Returns True for non-TYPE operations."""
        if gt_op.upper() not in ("TYPE", "INPUT"):
            return True  # value not applicable
        if not gt_value:
            return True  # no ground truth value to compare
        return pred_value.strip().lower() == gt_value.strip().lower()

    # ------------------------------------------------------------------
    # Public API — recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        annotation_id: str,
        action_uid: str,
        ground_truth: Dict,
        predicted_plan: str,
        grounding_result: Optional[Dict],
        step_index: int = -1,
        latency: float = 0.0,
    ) -> StepEvalResult:
        """Record a single predicted step and evaluate it against ground truth.

        Args:
            annotation_id:   task identifier
            action_uid:      action step identifier
            ground_truth:    dict from Mind2Web loader (contains operation, pos_candidates, etc.)
            predicted_plan:  raw text from action generation
            grounding_result: dict returned by SeeActActionGrounding.process()
            step_index:      0-based step position within the task
            latency:         total step latency in seconds

        Returns:
            StepEvalResult with all match flags computed.
        """
        # Determine step index automatically if not provided
        if step_index < 0:
            step_index = len(self._task_steps[annotation_id])

        # --- Extract ground truth ---
        gt_op = self._extract_gt_operation(ground_truth)
        gt_value = self._extract_gt_value(ground_truth)
        gt_pos_ids = self._extract_gt_pos_backend_ids(ground_truth)
        gt_repr = self._extract_gt_element_repr(ground_truth)

        # --- Extract predictions ---
        pred_type = self._parse_action_type(predicted_plan)
        pred_value = self._parse_typed_value(predicted_plan) if pred_type in ("TYPE", "INPUT") else ""
        pred_backend_id = self._extract_pred_backend_id(grounding_result)
        grounding_ok = (grounding_result or {}).get("success", False)

        # --- Compute matches ---
        op_match = self._match_operation(pred_type, gt_op)
        el_match = self._match_element(pred_backend_id, gt_pos_ids)
        val_match = self._match_value(pred_value, gt_value, gt_op)
        step_ok = op_match and el_match and val_match

        result = StepEvalResult(
            annotation_id=annotation_id,
            action_uid=action_uid,
            step_index=step_index,
            gt_operation=gt_op,
            gt_value=gt_value,
            gt_element_repr=gt_repr,
            gt_pos_backend_ids=gt_pos_ids,
            pred_action_type=pred_type,
            pred_value=pred_value,
            pred_plan_text=predicted_plan,
            pred_element=(grounding_result or {}).get("selected_element"),
            pred_backend_id=pred_backend_id,
            grounding_success=grounding_ok,
            operation_match=op_match,
            element_match=el_match,
            value_match=val_match,
            step_success=step_ok,
            latency=latency,
        )

        # Store
        self._task_steps[annotation_id].append(result)

        # Store task meta on first step
        if annotation_id not in self._task_meta:
            self._task_meta[annotation_id] = {
                "instruction": ground_truth.get("instruction", ""),
                "website": (ground_truth.get("metadata") or {}).get("website", ""),
                "domain": (ground_truth.get("metadata") or {}).get("domain", ""),
            }

        # Print step-level verdict
        verdict = "✓" if step_ok else "✗"
        print(f"    [{verdict}] Step {step_index}: op={op_match} ele={el_match} val={val_match}  "
              f"(pred={pred_type or '?'} gt={gt_op or '?'}, bid={pred_backend_id or 'none'})")

        return result

    # ------------------------------------------------------------------
    # Public API — compute & aggregate
    # ------------------------------------------------------------------

    def compute_metrics(self) -> AggregateMetrics:
        """Compute step-level and task-level metrics across all recorded steps.

        Populates self.task_results and self.aggregate.
        Returns the AggregateMetrics object.
        """
        all_steps: List[StepEvalResult] = []

        for ann_id, steps in self._task_steps.items():
            meta = self._task_meta.get(ann_id, {})
            n = len(steps)

            n_el = sum(1 for s in steps if s.element_match)
            n_op = sum(1 for s in steps if s.operation_match)
            n_val = sum(1 for s in steps if s.value_match)
            n_sr = sum(1 for s in steps if s.step_success)
            n_fail = n - n_sr

            task = TaskEvalResult(
                annotation_id=ann_id,
                instruction=meta.get("instruction", ""),
                website=meta.get("website", ""),
                domain=meta.get("domain", ""),
                total_steps=n,
                steps=steps,
                element_accuracy=n_el / n if n else 0.0,
                operation_f1=n_op / n if n else 0.0,
                value_accuracy=n_val / n if n else 0.0,
                step_success_rate=n_sr / n if n else 0.0,
                offline0=(n_fail == 0),
                offline1=(n_fail <= 1),
            )
            self.task_results[ann_id] = task
            all_steps.extend(steps)

        # Aggregate
        total_steps = len(all_steps)
        total_tasks = len(self.task_results)

        correct_el = sum(1 for s in all_steps if s.element_match)
        correct_op = sum(1 for s in all_steps if s.operation_match)
        correct_val = sum(1 for s in all_steps if s.value_match)
        correct_sr = sum(1 for s in all_steps if s.step_success)

        off0 = sum(1 for t in self.task_results.values() if t.offline0)
        off1 = sum(1 for t in self.task_results.values() if t.offline1)

        self.aggregate = AggregateMetrics(
            total_tasks=total_tasks,
            total_steps=total_steps,
            element_accuracy=correct_el / total_steps if total_steps else 0.0,
            operation_f1=correct_op / total_steps if total_steps else 0.0,
            value_accuracy=correct_val / total_steps if total_steps else 0.0,
            step_success_rate=correct_sr / total_steps if total_steps else 0.0,
            offline0_rate=off0 / total_tasks if total_tasks else 0.0,
            offline1_rate=off1 / total_tasks if total_tasks else 0.0,
            correct_elements=correct_el,
            correct_operations=correct_op,
            correct_values=correct_val,
            successful_steps=correct_sr,
            offline0_tasks=off0,
            offline1_tasks=off1,
        )

        return self.aggregate

    # ------------------------------------------------------------------
    # Public API — output
    # ------------------------------------------------------------------

    def save_results(self, tag: str = "") -> Dict[str, str]:
        """Save step-wise and task-wise evaluation results to JSON files.

        Files written:
            <output_dir>/step_results[_<tag>].json
            <output_dir>/task_results[_<tag>].json
            <output_dir>/aggregate_metrics[_<tag>].json

        Returns dict mapping result type to file path.
        """
        if self.aggregate is None:
            self.compute_metrics()

        suffix = f"_{tag}" if tag else ""
        paths = {}

        # --- Step-level results ---
        step_path = os.path.join(self.output_dir, f"step_results{suffix}.json")
        step_data = []
        for ann_id, steps in self._task_steps.items():
            for s in steps:
                d = s.to_dict()
                # Remove large fields that can't serialize easily
                d.pop("pred_element", None)
                step_data.append(d)

        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False, default=str)
        paths["steps"] = step_path
        print(f"  📄 Step results saved to {step_path}")

        # --- Task-level results ---
        task_path = os.path.join(self.output_dir, f"task_results{suffix}.json")
        task_data = []
        for t in self.task_results.values():
            td = {
                "annotation_id": t.annotation_id,
                "instruction": t.instruction,
                "website": t.website,
                "domain": t.domain,
                "total_steps": t.total_steps,
                "element_accuracy": round(t.element_accuracy, 4),
                "operation_f1": round(t.operation_f1, 4),
                "value_accuracy": round(t.value_accuracy, 4),
                "step_success_rate": round(t.step_success_rate, 4),
                "offline0": t.offline0,
                "offline1": t.offline1,
                "steps_summary": [
                    {
                        "action_uid": s.action_uid,
                        "step_index": s.step_index,
                        "gt_operation": s.gt_operation,
                        "pred_action_type": s.pred_action_type,
                        "operation_match": s.operation_match,
                        "element_match": s.element_match,
                        "value_match": s.value_match,
                        "step_success": s.step_success,
                    }
                    for s in t.steps
                ],
            }
            task_data.append(td)

        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False, default=str)
        paths["tasks"] = task_path
        print(f"  📄 Task results saved to {task_path}")

        # --- Aggregate metrics ---
        agg_path = os.path.join(self.output_dir, f"aggregate_metrics{suffix}.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(self.aggregate.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        paths["aggregate"] = agg_path
        print(f"  📄 Aggregate metrics saved to {agg_path}")

        return paths

    def print_summary(self):
        """Print a formatted evaluation summary to stdout."""
        if self.aggregate is None:
            self.compute_metrics()

        a = self.aggregate
        print("\n" + "=" * 70)
        print("  SeeAct Evaluation Summary")
        print("=" * 70)

        print(f"\n  Tasks evaluated  : {a.total_tasks}")
        print(f"  Total steps      : {a.total_steps}")

        print(f"\n  ── Step-Level Metrics ────────────────────────────")
        print(f"  Element Accuracy : {a.element_accuracy:>7.2%}  ({a.correct_elements}/{a.total_steps})")
        print(f"  Operation F1     : {a.operation_f1:>7.2%}  ({a.correct_operations}/{a.total_steps})")
        print(f"  Value Accuracy   : {a.value_accuracy:>7.2%}  ({a.correct_values}/{a.total_steps})")
        print(f"  Step Success Rate: {a.step_success_rate:>7.2%}  ({a.successful_steps}/{a.total_steps})")

        print(f"\n  ── Task-Level Metrics ────────────────────────────")
        print(f"  Offline0 (strict)   : {a.offline0_rate:>7.2%}  ({a.offline0_tasks}/{a.total_tasks})")
        print(f"  Offline1 (tolerance): {a.offline1_rate:>7.2%}  ({a.offline1_tasks}/{a.total_tasks})")

        # Per-task breakdown
        if self.task_results:
            print(f"\n  ── Per-Task Breakdown ────────────────────────────")
            for t in self.task_results.values():
                status0 = "✓" if t.offline0 else "✗"
                status1 = "✓" if t.offline1 else "✗"
                print(f"  [{status0}/{status1}] {t.annotation_id[:20]:20s}  "
                      f"steps={t.total_steps}  ele={t.element_accuracy:.0%}  "
                      f"op={t.operation_f1:.0%}  sr={t.step_success_rate:.0%}")

        print("\n" + "=" * 70)

    def reset(self):
        """Clear all accumulated results."""
        self._task_steps.clear()
        self._task_meta.clear()
        self.task_results.clear()
        self.aggregate = None
