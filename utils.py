import numpy as np
from collections import defaultdict
import itertools

# ------------------------
# Printing evaluation results
# ------------------------

def print_eval_result(metrics: dict, stage="val", is_improved=False):
    """
    Print evaluation results (accuracy, F1-macro).
    
    Args:
        metrics: dict with keys 'accuracy' and 'f1_macro'
        stage: string label (e.g., "val", "test")
        is_improved: mark with '*' if results improved
    """
    star = " *" if is_improved else ""
    print(f"[{stage.upper():4}] Acc: {metrics['accuracy']:.4f} | "
          f"F1-macro: {metrics['f1_macro']:.4f}{star}")


def print_eval_result_esci(metrics: dict, stage="val", is_improved=False):
    """
    Print evaluation results including per-class accuracy for ESCI labels.
    
    Args:
        metrics: dict with 'accuracy', 'f1_macro', and optionally 'per_class_accuracy'
        stage: string label (e.g., "val", "test")
        is_improved: mark with '*' if results improved
    """
    star = " *" if is_improved else ""
    print(f"[{stage.upper():4}] Acc: {metrics['accuracy']:.4f} | "
          f"F1-macro: {metrics['f1_macro']:.4f}{star}")

    # Print per-class accuracy if available
    if "per_class_accuracy" in metrics:
        id2label = {0: "E", 1: "S", 2: "C", 3: "I"}
        per_class_acc_str = [
            f"{id2label[cls_id]}: {acc:.4f}" 
            for cls_id, acc in metrics["per_class_accuracy"].items()
        ]
        print("        " + " | ".join(per_class_acc_str))