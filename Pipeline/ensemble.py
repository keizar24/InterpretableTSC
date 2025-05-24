import os
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Pipeline import Pipeline
from PrototypeBasedModel import (
    PrototypeResNet,
    PrototypeCNN,
    PrototypeLSTM,
    PrototypeMLP,
    PrototypeFCN,
)


def majority_vote(pred_list: Iterable[np.ndarray]) -> np.ndarray:
    """Return element-wise majority vote across prediction arrays."""
    stack = np.stack(pred_list, axis=0)
    votes = []
    for i in range(stack.shape[1]):
        counts = np.bincount(stack[:, i].astype(int))
        votes.append(np.argmax(counts))
    return np.array(votes)


def ensemble_pipeline(
    model_class,
    file_path: str,
    n_vars: int,
    num_classes: int,
    selection_types: Iterable[str],
    distance_metrics: Iterable[str],
    result_dir: str | None = None,
    num_prototypes: int = 8,
    threshold: float = 0.5,
    **train_kwargs,
):
    """Train pipelines for each prototype selection and metric then do majority vote."""
    preds_all = []
    labels_ref = None
    for sel in selection_types:
        for metric in distance_metrics:
            sub_dir = None
            if result_dir is not None:
                sub_dir = os.path.join(result_dir, f"{sel}_{metric}")
            pipe = Pipeline(
                model_class=model_class,
                file_path=file_path,
                n_vars=n_vars,
                num_classes=num_classes,
                result_dir=sub_dir,
                use_prototype=True,
                num_prototypes=num_prototypes,
                prototype_selection_type=sel,
                prototype_distance_metric=metric,
            )
            pipe.train(**train_kwargs)
            preds, labels = pipe.predict(threshold=threshold)
            preds_all.append(preds)
            if labels_ref is None:
                labels_ref = labels
    if labels_ref is None:
        raise ValueError("No predictions were produced.")

    final_preds = majority_vote(preds_all)
    acc = accuracy_score(labels_ref, final_preds)
    f1 = f1_score(labels_ref, final_preds)
    return final_preds, {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ensemble prototype pipeline")
    parser.add_argument("--file", required=True, help="Path to labelled CSV file")
    parser.add_argument("--n-vars", type=int, default=5, help="Number of feature columns")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--result-dir", default="../Result/ensemble")
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument(
        "--selections",
        nargs="+",
        default=["random", "k-means", "gmm"],
        help="Prototype selection methods",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["euclidean", "cosine"],
        help="Prototype distance metrics",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument(
        "--model",
        default="PrototypeResNet",
        choices=[
            "PrototypeResNet",
            "PrototypeCNN",
            "PrototypeLSTM",
            "PrototypeMLP",
            "PrototypeFCN",
        ],
    )
    args = parser.parse_args()

    model_map = {
        "PrototypeResNet": PrototypeResNet,
        "PrototypeCNN": PrototypeCNN,
        "PrototypeLSTM": PrototypeLSTM,
        "PrototypeMLP": PrototypeMLP,
        "PrototypeFCN": PrototypeFCN,
    }
    model_cls = model_map[args.model]

    preds, metrics = ensemble_pipeline(
        model_class=model_cls,
        file_path=args.file,
        n_vars=args.n_vars,
        num_classes=args.num_classes,
        selection_types=args.selections,
        distance_metrics=args.metrics,
        result_dir=args.result_dir,
        num_prototypes=args.num_prototypes,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        normalize=args.normalize,
        balance=args.balance,
    )
    print(metrics)
