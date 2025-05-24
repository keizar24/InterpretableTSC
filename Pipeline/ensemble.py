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
