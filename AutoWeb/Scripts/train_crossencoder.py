"""
train_crossencoder.py

Loads a Mind2Web parquet shard, constructs positive/negative training
pairs, fine-tunes a CrossEncoder (e.g. DeBERTa-based) and stores the trained
model in the shared models directory.  Produces training/evaluation plots
and a simple confusion matrix on the dev set.

Usage: edit constants below and run with the Python environment (ml-env).
"""
from __future__ import annotations
import os
import random
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd

# add paths so we can import the SeeAct-main utilities
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
seeact_root = r"d:\Mtech\Sem 3\Codes\SeeAct-main\src"
autoweb_root = os.path.abspath(os.path.join(workspace_root))
# include both repos and ensure AutoWeb package root is visible
sys.path.insert(0, seeact_root)
sys.path.insert(0, autotweb_root := autoweb_root)
# add parent of AutoWeb so that "import AutoWeb" works
sys.path.insert(0, os.path.abspath(os.path.join(autoweb_root, "..")))
sys.path.insert(0, os.path.join(workspace_root, "src"))

print("sys.path entries:", sys.path[:6])

from sentence_transformers import InputExample, evaluation
from torch.utils.data import DataLoader

# import ranking helper from SeeAct-main
try:
    from demo_utils.ranking_model import CrossEncoder
except Exception as e:
    print("Failed to import demo_utils from SeeAct-main; sys.path=", sys.path)
    raise

from mind2Web_Loader import Mind2WebDataset

# ------------------ USER CONFIGURATION ------------------
PARQUET_PATH = r"D:\\Environments\\Datasets\\multimodal-mind2web\\data\\train-00000-of-00027-4d11798d7219186d.parquet"
MODEL_OUTPUT_DIR = r"D:\\Environments\\Models\\cross-encoder-seeact"
PLOTS_DIR = r"D:\\Mtech\\Sem 3\\Codes\\AutoWeb\\Scripts\\training_plots"
TRAIN_FRACTION = 0.8            # train/dev split
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
SEED = 42
# --------------------------------------------------------

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)


def build_examples(parquet_path: str) -> list[InputExample]:
    loader = Mind2WebDataset(root_dir=os.path.dirname(os.path.dirname(parquet_path)))
    df = loader.df
    # we assume `target_action_reprs` contains the text representation of the
    # ground‑truth element for grounding
    df = df[df["target_action_reprs"].notna()]

    examples: list[InputExample] = []
    elements = df["target_action_reprs"].astype(str).tolist()

    for _, row in df.iterrows():
        plan = str(row.get("confirmed_task", ""))
        element = str(row.get("target_action_reprs", ""))
        examples.append(InputExample(texts=[plan, element], label=1.0))
        # add one random negative example
        neg = element
        while neg == element:
            neg = random.choice(elements)
        examples.append(InputExample(texts=[plan, neg], label=0.0))
    return examples


def split_examples(examples: list[InputExample], fraction: float = 0.8):
    random.shuffle(examples)
    cut = int(len(examples) * fraction)
    return examples[:cut], examples[cut:]


def plot_scores(scores: list[float], out_path: str):
    plt.figure()
    plt.plot(scores, marker="o")
    plt.title("Evaluator score per evaluation step")
    plt.xlabel("evaluation step")
    plt.ylabel("score")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion(gt_labels: list[float], pred_labels: list[float], out_path: str):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(gt_labels, pred_labels)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("predicted")
    plt.ylabel("ground truth")
    plt.savefig(out_path)
    plt.close()


def main():
    # build training/validation examples
    all_examples = build_examples(PARQUET_PATH)
    train_ex, dev_ex = split_examples(all_examples, TRAIN_FRACTION)
    print(f"Training examples: {len(train_ex)}, dev examples: {len(dev_ex)}")

    train_dataloader = DataLoader(train_ex, shuffle=True, batch_size=BATCH_SIZE)
    # skip builtin evaluator because CrossEncoder does not support encode()
    dev_evaluator = None  # we will evaluate manually after training

    # initialize model (starting from pretrained cross-encoder)
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
    # ensure compatibility with newer sentence-transformers evaluator
    model.similarity_fn_name = "cosine"

    eval_scores: list[float] = []

    def callback(score, epoch, steps):
        eval_scores.append(score)

    # instead of running all epochs at once, iterate so we can evaluate after
    # each epoch and record accuracy/error for train and dev sets.
    history = {
        "epoch": [],
        "train_acc": [],
        "train_err": [],
        "dev_acc": [],
        "dev_err": [],
    }

    def compute_accuracy(model, examples):
        correct = 0
        for ex in examples:
            score = model.predict([ex.texts])[0]
            pred = 1.0 if score >= 0.5 else 0.0
            if pred == ex.label:
                correct += 1
        return correct / len(examples) if examples else 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\n=== Starting epoch {epoch+1}/{EPOCHS} ===")
        model.fit(
            train_dataloader,
            evaluator=dev_evaluator,
            epochs=1,
            evaluation_steps=0,
            optimizer_params={"lr": LR},
            output_path=MODEL_OUTPUT_DIR,
            save_best_model=True,
            callback=callback,
            show_progress_bar=True,
        )

        # evaluate accuracy on both sets
        train_acc = compute_accuracy(model, train_ex)
        dev_acc = compute_accuracy(model, dev_ex)
        history["epoch"].append(epoch + 1)
        history["train_acc"].append(train_acc)
        history["train_err"].append(1 - train_acc)
        history["dev_acc"].append(dev_acc)
        history["dev_err"].append(1 - dev_acc)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, dev_acc={dev_acc:.4f}")
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")

    # ensure final model saved
    print("Saving final model to", MODEL_OUTPUT_DIR)
    model.save(MODEL_OUTPUT_DIR)

    # plot accuracy/error graphs
    def plot_history(hist: dict, out_prefix: str):
        epochs = hist["epoch"]
        plt.figure()
        plt.plot(epochs, hist["train_acc"], label="train acc", marker="o")
        plt.plot(epochs, hist["dev_acc"], label="dev acc", marker="o")
        plt.title("Accuracy per epoch")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_prefix + "_acc.png")
        plt.close()

        plt.figure()
        plt.plot(epochs, hist["train_err"], label="train err", marker="o")
        plt.plot(epochs, hist["dev_err"], label="dev err", marker="o")
        plt.title("Error per epoch")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_prefix + "_err.png")
        plt.close()

    plot_history(history, os.path.join(PLOTS_DIR, "epoch_history"))

    # confusion matrix using the final model
    trained = model
    trained.eval()
    gt_labels = []
    pred_labels = []
    for ex in dev_ex:
        score = trained.predict([ex.texts])[0]
        pred = 1.0 if score >= 0.5 else 0.0
        pred_labels.append(pred)
        gt_labels.append(ex.label)
    plot_confusion(gt_labels, pred_labels, os.path.join(PLOTS_DIR, "confusion.png"))

    print("Training complete. Model saved to", MODEL_OUTPUT_DIR)
    print("Set environment variable CROSSENCODER_MODEL_PATH to this folder before running grounding.")


if __name__ == "__main__":
    main()
