import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
from pathlib import Path
import json

# ------------------------
# Function for loads
# ------------------------

def load_lines(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_pid2text(p: Path):
    """TSV: pid \\t text  -> dict[pid]=text"""
    pid2text = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                pid, text = parts
                pid2text[pid] = text
    return pid2text

def load_classes_int(p: Path):
    class_dict = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            label_int, label_str = line.rstrip("\n").split("\t")
            class_dict[int(label_int)] = label_str
    return class_dict

def load_classes_str(p: Path):
    class_dict = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            label_int, label_str = line.rstrip("\n").split("\t")
            class_dict[label_str] = int(label_int)
    return class_dict

def load_keywords(p: Path):
    keywords = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            key, items = line.rstrip("\n").split(":")
            item_list = [item for item in items.split(",")]
            keywords[key] = item_list
    return keywords

def load_class_graph(p: Path):
    edges = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            p, c = map(int, line.rstrip("\n").split("\t"))
            edges.append((p, c))
    return edges

def load_json(path):
    """Load JSON file into Python object."""
    with open(path) as f:
        return json.load(f)
    
# ------------------------
# Visualization
# ------------------------

def plot_results(results_dict, split="valid", metric='Accuracy'):
    """
    Plot metric values over epochs for multiple models.
    
    Args:
        results_dict: dict of dicts. 
            Example: results_dict["valid"]["mlp_partial"] = [0.8, 0.82, ...]
        split: "valid" or "test"
        metric: name of the metric to display
    """
    assert split in results_dict, f"{split} not in results_dict"

    plt.figure(figsize=(8, 5))
    for label, acc_list in results_dict[split].items():
        plt.plot(acc_list, label=label)

    plt.title(f"{split.capitalize()} {metric} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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