import copy
import os
from typing import Literal, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample

try:
    from imblearn.over_sampling import SMOTE

    _has_smote = True
except ImportError:
    _has_smote = False
    print("Warning: imbalanced-learn is not installed. SMOTE will not be available.")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Baseline model definitions
from BaselineModel import (
    ResNet_baseline,
    CNN_baseline,
    LSTM_baseline,
    MLP_baseline,
    FCN_baseline
)

# Prototype-based model
from PrototypeBasedModel import (
    PrototypeBasedModel,  # ResNet variant (backward compatibility)
    PrototypeCNN,
    PrototypeFCN,
    PrototypeLSTM,
    PrototypeMLP,
    PrototypeResNet,
    PrototypeModelBase,
    PrototypeFeatureExtractor,
    PrototypeSelector,
)


###############################################################################
# Custom Focal Loss
###############################################################################
class FocalLoss(nn.Module):
    """
    Focal loss: a variant of cross-entropy that focuses more on misclassified examples.
    gamma>1 emphasizes hard-to-classify samples.
    alpha can be used to assign different weights to classes.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C), targets: (B,)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = 1 - p if y=1
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


###############################################################################
# Pipeline
###############################################################################
class Pipeline:
    """
    Main logic:
      - Load CSV and do preprocessing (optional normalization, optional prototype selection)
      - Split into train/valid/test. Only the training set is balanced (over/under/SMOTE).
      - Optionally apply cost-sensitive/improved losses (Weighted CE/Focal Loss)
      - Train with early stopping
      - Evaluate on the test set with a customizable threshold
    """

    def __init__(
            self,
            model_class,
            file_path: str,
            n_vars: int,
            num_classes: int,
            result_dir: str | None = None,
            use_prototype: bool = False,
            num_prototypes: int = 8,
            prototype_selection_type: str = 'random',
            prototype_distance_metric: str = 'euclidean',
    ):
        self.model_class = model_class
        self.file_path = file_path
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.window_size = 600
        self.use_prototype = use_prototype
        self.num_prototypes = num_prototypes
        self.prototype_selection_type = prototype_selection_type
        self.prototype_distance_metric = prototype_distance_metric

        if result_dir is None:
            result_dir = f"../Result/{model_class.__name__}"
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | result_dir={self.result_dir}")

        self._df = None
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = None
        self.best_model = None
        self._opt_metric = "loss"

        self._normalize = True
        self._balance = False
        self._balance_strategy = "over"

        self.X_test = None
        self.y_true = None
        self.confusion_mat = None

        self._prototypes = None
        self._proto_labels = None

        self._load_raw_csv()

    def _load_raw_csv(self):
        df = pd.read_csv(self.file_path)
        self._df = df.copy()

    @staticmethod
    def _reset_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            m.reset_parameters()

    @staticmethod
    def _binary_f1(labels: np.ndarray, preds: np.ndarray) -> float:
        return f1_score(labels, preds, pos_label=1, average="binary")

    def preprocessing(self, normalize: bool = True):
        """
        Basic preprocessing (e.g. normalization and optional prototype extraction).
        """
        self._normalize = normalize
        df = self._df.copy()

        # Select relevant columns (Close, High, Low, Open, Volume, etc.) up to n_vars
        feature_cols = ["Close", "High", "Low", "Open", "Volume"][:self.n_vars]

        # Normalization
        if normalize:
            for col in feature_cols:
                mn, mx = df[col].min(), df[col].max()
                df[col] = (df[col] - mn) / (mx - mn + 1e-12)

        # Build sliding windows of size self.window_size
        X_list, y_list = [], []
        half_w = self.window_size // 2
        total_len = len(df)
        for start_idx in range(total_len - self.window_size + 1):
            end_idx = start_idx + self.window_size
            window_data = df.iloc[start_idx:end_idx][feature_cols].values
            center_label_idx = start_idx + half_w
            label = df.iloc[center_label_idx]["Labels"]
            X_list.append(window_data)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # If using the prototype-based approach, select prototypes first
        if self.use_prototype:
            selector = PrototypeSelector(X, y, window_size=self.window_size)
            protos, proto_labels, rem_data, rem_labels = selector.select_prototypes(
                num_prototypes=self.num_prototypes,
                selection_type=self.prototype_selection_type
            )
            self._prototypes = protos
            self._proto_labels = proto_labels

            # Compute prototype features
            t_data = torch.from_numpy(rem_data)
            t_proto = torch.from_numpy(protos)
            extractor = PrototypeFeatureExtractor(t_data, t_proto)
            extractor.plot_prototype_feature_map(
                metric=self.prototype_distance_metric,
                save_path=os.path.join(self.result_dir, "prototype_feature_map.png")
            )
            extractor.plot_prototype_cycles(
                save_dir=self.result_dir,
                short_window=30,
                long_window=120,
                prefix="prototype_cycle"
            )
            feats = extractor.compute_prototype_features(metric=self.prototype_distance_metric)
            X = feats.numpy()
            y = rem_labels

        self.dataset = (X, y)

    def data_loader(
            self,
            batch_size: int = 32,
            train_ratio: float = 0.7,
            valid_ratio: float = 0.15,
            test_ratio: float = 0.15,
            balance: bool = False,
            balance_strategy: Literal["over", "under", "smote"] = "over",
    ):
        """
        Split into train/valid/test. If balance=True, only the training set is balanced.
        balance_strategy can be "over", "under", or "smote" (if imbalanced-learn is installed).
        """
        self._balance = balance
        self._balance_strategy = balance_strategy

        if self.dataset is None:
            raise ValueError("Call preprocessing() first.")

        X, y = self.dataset
        print(f"Full dataset shape: X={X.shape}, y={y.shape}")
        full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        total = len(full_dataset)
        tr = int(train_ratio * total)
        va = int(valid_ratio * total)
        te = total - tr - va
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [tr, va, te], generator=torch.Generator().manual_seed(42)
        )

        print(f"Split sizes: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

        # Balance training set if needed
        if balance:
            x_train_list, y_train_list = [], []
            for (xx, yy) in train_ds:
                x_train_list.append(xx.numpy())
                y_train_list.append(yy.item())
            x_train_arr = np.array(x_train_list)
            y_train_arr = np.array(y_train_list)

            if balance_strategy == "over" or balance_strategy == "under":
                # Count class 0/1
                c0 = np.sum(y_train_arr == 0)
                c1 = np.sum(y_train_arr == 1)
                if c1 > c0:
                    maj_label, min_label = 1, 0
                else:
                    maj_label, min_label = 0, 1

                x_maj = x_train_arr[y_train_arr == maj_label]
                x_min = x_train_arr[y_train_arr == min_label]

                if balance_strategy == "over":
                    # Oversample minority class
                    x_min_rs = resample(x_min, replace=True, n_samples=len(x_maj), random_state=42)
                    y_min_rs = np.full(len(x_maj), min_label)
                    x_bal = np.concatenate([x_maj, x_min_rs], axis=0)
                    y_bal = np.concatenate([
                        np.full(len(x_maj), maj_label),
                        y_min_rs
                    ], axis=0)
                else:
                    # Undersample majority class
                    x_maj_rs = resample(x_maj, replace=False, n_samples=len(x_min), random_state=42)
                    y_maj_rs = np.full(len(x_min), maj_label)
                    x_bal = np.concatenate([x_maj_rs, x_min], axis=0)
                    y_bal = np.concatenate([y_maj_rs, np.full(len(x_min), min_label)], axis=0)

                shuffle_idx = np.random.RandomState(42).permutation(len(x_bal))
                x_bal = x_bal[shuffle_idx]
                y_bal = y_bal[shuffle_idx]

            elif balance_strategy == "smote":
                # SMOTE requires imbalanced-learn
                if not _has_smote:
                    raise ImportError("imbalanced-learn is not installed. SMOTE is unavailable.")
                # Flatten shape (N, window_size, n_vars) into (N, window_size*n_vars) for SMOTE
                # Then reshape back. This is just a demonstration; real cases may require more nuanced methods.
                N, W, D = x_train_arr.shape
                x_train_2d = x_train_arr.reshape(N, -1)
                sm = SMOTE(random_state=42)
                x_bal_2d, y_bal = sm.fit_resample(x_train_2d, y_train_arr)
                newN = x_bal_2d.shape[0]
                x_bal = x_bal_2d.reshape(newN, W, D)

            print(f"Balanced train set: X={x_bal.shape}, y={y_bal.shape}")
            train_ds = TensorDataset(torch.from_numpy(x_bal).float(), torch.from_numpy(y_bal).long())
        else:
            # Print unbalanced train set info
            x_train_list, y_train_list = [], []
            for (xx, yy) in train_ds:
                x_train_list.append(xx.numpy())
                y_train_list.append(yy.item())
            print(f"Unbalanced train set: X={np.array(x_train_list).shape}, y={np.array(y_train_list).shape}")

        # Validation set
        x_valid_list, y_valid_list = [], []
        for (xx, yy) in valid_ds:
            x_valid_list.append(xx.numpy())
            y_valid_list.append(yy.item())
        print(f"Valid set shape: X={np.array(x_valid_list).shape}, y={np.array(y_valid_list).shape}")

        # Test set
        x_test_list, y_test_list = [], []
        for (xx, yy) in test_ds:
            x_test_list.append(xx.numpy())
            y_test_list.append(yy.item())
        self.X_test = torch.from_numpy(np.array(x_test_list))
        self.y_true = torch.from_numpy(np.array(y_test_list))
        print(f"Test set shape: X={self.X_test.shape}, y={self.y_true.shape}")

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def _eval_val(self, model):
        """
        Perform a simple validation-set evaluation using argmax. Returns accuracy and f1.
        """
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in self.valid_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                out = model(x)
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)
        return acc, f1

    def _train_loop(self, optimizer, criterion, epochs, patience):
        """
        Standard training loop with early stopping on validation loss.
        """
        best_loss = float("inf")
        wait = 0
        self.best_model = copy.deepcopy(self.model)

        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(self.train_loader.dataset)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in self.valid_loader:
                    x, y = x.to(self.device).float(), y.to(self.device)
                    val_loss += criterion(self.model(x), y).item() * x.size(0)
            val_loss /= len(self.valid_loader.dataset)

            print(f"Epoch {ep + 1}/{epochs} | Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model.load_state_dict(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(self.best_model.state_dict())
        return best_loss

    def _optuna_objective(self, trial):
        """
        Objective function for hyperparameter optimization using Optuna.
        You can customize the search ranges and the metric to optimize.
        """
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
        opt = trial.suggest_categorical("opt", ["adam", "sgd"])
        epochs = trial.suggest_int("epochs", 20, 60)

        # For each trial, do the data preprocessing & loading again
        self.preprocessing(normalize=self._normalize)
        self.data_loader(batch_size=32, balance=self._balance, balance_strategy=self._balance_strategy)

        # Re-initialize model
        if issubclass(self.model_class, PrototypeModelBase):
            self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
        else:
            self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)
        self.model.apply(self._reset_weights)

        # Default loss is CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        if opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

        patience = 10
        best_val_loss = float("inf")
        wait = 0

        for ep in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

            # Evaluate on the validation set
            self.model.eval()
            vloss = 0
            with torch.no_grad():
                for x, y in self.valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    vloss += criterion(self.model(x), y).item() * x.size(0)
            vloss /= len(self.valid_loader.dataset)

            acc, f1 = self._eval_val(self.model)
            # Determine which metric to optimize
            if self._opt_metric == "loss":
                metric = vloss
            elif self._opt_metric == "accuracy":
                metric = acc
            else:
                metric = f1

            trial.report(metric, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if vloss < best_val_loss:
                best_val_loss = vloss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return metric

    def train(
            self,
            epochs: int = 50,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            use_hpo: bool = False,
            n_trials: int = 30,
            optimize_metric: str = "loss",
            patience: int = 10,
            normalize: bool = True,
            balance: bool = False,
            balance_strategy: Literal["over", "under", "smote"] = "over",
            cost_sensitive: Literal["weighted_ce", "focal", None] = None,
            focal_alpha: float = 1.0,
            focal_gamma: float = 2.0,
    ):
        """
        Arguments:
          - cost_sensitive:
             * "weighted_ce": Weighted cross-entropy
             * "focal": Focal Loss
             * None: default cross-entropy
        """
        self._opt_metric = optimize_metric.lower()
        self._normalize = normalize
        self._balance = balance
        self._balance_strategy = balance_strategy

        # If use_hpo=True, run Optuna for hyperparameter search
        if use_hpo:
            direction = "minimize" if self._opt_metric == "loss" else "maximize"
            study = optuna.create_study(direction=direction)
            study.optimize(self._optuna_objective, n_trials=n_trials)

            best = study.best_params
            with open(os.path.join(self.result_dir, "optuna_best_params.txt"), "w") as f:
                f.write(f"Best hyperparams:\n{best}\n")
                f.write(f"\nBest {self._opt_metric}: {study.best_value}\n")

            # Train again with the best hyperparams
            lr = best["lr"]
            wd = best["wd"]
            opt_method = best["opt"]
            ep = best["epochs"]

            self.preprocessing(normalize=normalize)
            self.data_loader(batch_size=batch_size, balance=balance, balance_strategy=balance_strategy)

            if issubclass(self.model_class, PrototypeModelBase):
                self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
            else:
                self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)

            # Build the criterion
            criterion = nn.CrossEntropyLoss()
            if cost_sensitive == "weighted_ce":
                # Compute class weights from the final train set
                all_ys = []
                for _, yy in self.train_loader.dataset:
                    all_ys.append(yy.item())
                counts = np.bincount(all_ys)
                weights = [sum(counts) / c for c in counts]
                weights = torch.FloatTensor(weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            elif cost_sensitive == "focal":
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

            if opt_method == "adam":
                optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

            val_loss = self._train_loop(optimizer, criterion, epochs=ep, patience=patience)
            return self.best_model, val_loss

        else:
            # Otherwise, just do a single training run
            self.preprocessing(normalize=normalize)
            self.data_loader(batch_size=batch_size, balance=balance, balance_strategy=balance_strategy)

            if issubclass(self.model_class, PrototypeModelBase):
                self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
            else:
                self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)

            # Build the criterion
            criterion = nn.CrossEntropyLoss()
            if cost_sensitive == "weighted_ce":
                all_ys = []
                for _, yy in self.train_loader.dataset:
                    all_ys.append(yy.item())
                counts = np.bincount(all_ys)
                weights = [sum(counts) / c for c in counts]
                weights = torch.FloatTensor(weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            elif cost_sensitive == "focal":
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            val_loss = self._train_loop(optimizer, criterion, epochs=epochs, patience=patience)
            return self.best_model, val_loss

    def predict(self, threshold: float = 0.5):
        """Return predicted and true labels on the test set."""
        if self.test_loader is None:
            raise ValueError("Call data_loader() before predict().")

        self.model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                logits = self.model(x)
                probs = nn.functional.softmax(logits, dim=1)[:, 1]
                pred = (probs >= threshold).long()
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return np.array(preds), np.array(labels)

    def evaluate(self, threshold: float = 0.5):
        """
        Evaluate on the test set. We compute softmax => take class-1 probability => compare with threshold.
        Default threshold=0.5.
        """
        if self.test_loader is None:
            raise ValueError("Call data_loader() before evaluate().")
        preds, labels = self.predict(threshold=threshold)
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)
        self.confusion_mat = confusion_matrix(labels, preds)
        print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Threshold={threshold}")

        plt.figure(figsize=(4, 3))
        sns.heatmap(self.confusion_mat, annot=True, cmap="Blues", fmt="d",
                    xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(self.result_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        return {"accuracy": acc, "f1": f1}

    def find_best_threshold(self, step: float = 0.01, metric: str = "f1", plot_curve: bool = True) -> float:
        """
        Post-hoc search on the validation set to find the best threshold that maximizes a chosen metric (f1 or accuracy).
        step: increment to sweep threshold [0, 1].
        metric: "f1" or "accuracy"
        plot_curve: if True, save a (threshold vs. metric) plot in result_dir.
        Returns the best threshold.
        """
        if self.valid_loader is None:
            raise ValueError("Please call data_loader() to create train/valid/test splits.")
        if self.model is None:
            raise ValueError("Please call train() to train the model first.")

        self.model.eval()
        all_probs = []
        all_labels = []
        # Collect validation set predictions (as probabilities of class=1)
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        thresholds = np.arange(0.0, 1.0 + 1e-9, step)
        scores = []
        best_threshold = 0.5
        best_score = -1

        for th in thresholds:
            preds = (all_probs >= th).astype(int)
            if metric.lower() == "f1":
                sc = f1_score(all_labels, preds)
            elif metric.lower() == "accuracy":
                sc = accuracy_score(all_labels, preds)
            else:
                raise ValueError("Only 'f1' or 'accuracy' are supported as metrics.")
            scores.append(sc)
            if sc > best_score:
                best_score = sc
                best_threshold = th

        print(f"[find_best_threshold] Best threshold on validation set = {best_threshold:.3f}, {metric}={best_score:.4f}")

        # Optional: plot threshold curve
        if plot_curve:
            plt.figure(figsize=(6, 4))
            plt.plot(thresholds, scores, marker="o")
            plt.xlabel("Threshold")
            plt.ylabel(metric.upper())
            plt.title(f"Threshold vs. {metric.upper()} (val set)")
            plt.grid(True)
            plt.tight_layout()
            curve_path = os.path.join(self.result_dir, f"threshold_{metric}_curve.png")
            plt.savefig(curve_path)
            plt.close()

        return best_threshold


###############################################################################
# Ensemble utilities
###############################################################################
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

###############################################################################
# Example usage (main)
###############################################################################
if __name__ == "__main__":
    CSV_FILE_PATH_1 = "../Dataset/ftse_minute_data_may_labelled.csv"
    n_var_1 = 4
    CSV_FILE_PATH_2 = "../Dataset/ftse_minute_data_daily_labelled.csv"
    n_var_2 = 5

    # Baseline models
    baseline_models = [
        ResNet_baseline.ResNet,
        CNN_baseline.CNN,
        LSTM_baseline.LSTM,
        MLP_baseline.MLP,
        FCN_baseline.FCN
    ]
    # Prototype-based models
    selection_types = ["random", "k-means", "gmm"]
    distance_metrics = ["euclidean", "cosine"]
    prototype_models = [
        PrototypeResNet,
        PrototypeCNN,
        PrototypeLSTM,
        PrototypeMLP,
        PrototypeFCN,
    ]

    for model_class in baseline_models:
        pipeline = Pipeline(
            model_class=model_class,
            file_path=CSV_FILE_PATH_1,
            n_vars=n_var_1,
            num_classes=2,
            result_dir=f"../Result/may/{model_class.__name__}",
            use_prototype=False
        )

        pipeline.train(
            use_hpo=True,
            n_trials=10,
            epochs=10,
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="smote",  # requires imbalanced-learn
            optimize_metric="f1",
            cost_sensitive="weighted_ce",  # use FocalLoss
            focal_alpha=1.0,
            focal_gamma=2.0
        )
        bst_threshold = pipeline.find_best_threshold(step=0.01, metric="f1", plot_curve=False)
        results = pipeline.evaluate(threshold=bst_threshold)
        print(f"{model_class.__name__} results:", results)

    for proto_class in prototype_models:
        _, metrics = ensemble_pipeline(
            model_class=proto_class,
            file_path=CSV_FILE_PATH_1,
            n_vars=n_var_1,
            num_classes=2,
            selection_types=selection_types,
            distance_metrics=distance_metrics,
            result_dir=f"../Result/may/{proto_class.__name__}",
            num_prototypes=10,
            threshold=0.5,
            use_hpo=True,
            n_trials=10,
            epochs=10,
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="smote",
            optimize_metric="f1",
            cost_sensitive="weighted_ce",
        )
        print(f"{proto_class.__name__} ensemble results:", metrics)

    # Daily
    for model_class in baseline_models:
        pipeline = Pipeline(
            model_class=model_class,
            file_path=CSV_FILE_PATH_2,
            n_vars=n_var_2,
            num_classes=2,
            result_dir=f"../Result/{model_class.__name__}",
            use_prototype=False
        )

        # Train example: enable FocalLoss + SMOTE + auto hpo
        pipeline.train(
            use_hpo=True,
            n_trials=10,
            epochs=10,  # just an initial range for Optuna, actual ep is overridden by best param
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="smote",  # requires imbalanced-learn
            optimize_metric="f1",
            cost_sensitive="weighted_ce",  # use FocalLoss
            focal_alpha=1.0,
            focal_gamma=2.0
        )

        bst_threshold = pipeline.find_best_threshold(step=0.01, metric="f1", plot_curve=False)
        results = pipeline.evaluate(threshold=bst_threshold)
        print(f"{model_class.__name__} results:", results)

    # Prototype-based models
    for proto_class in prototype_models:
        _, metrics = ensemble_pipeline(
            model_class=proto_class,
            file_path=CSV_FILE_PATH_2,
            n_vars=n_var_2,
            num_classes=2,
            selection_types=selection_types,
            distance_metrics=distance_metrics,
            result_dir=f"../Result/{proto_class.__name__}",
            num_prototypes=10,
            threshold=0.5,
            use_hpo=True,
            n_trials=10,
            epochs=10,
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="smote",
            optimize_metric="f1",
            cost_sensitive="weighted_ce",
        )
        print(f"{proto_class.__name__} ensemble results:", metrics)
