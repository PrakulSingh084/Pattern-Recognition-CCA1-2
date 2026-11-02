#!/usr/bin/env python3
"""Compare three classifiers on published CASME II HOOF features.

The measurements are pre-extracted by Peng et al.; this script implements and
evaluates the classifiers. It does not claim to extract motion from raw videos.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import warnings
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

# Small folds train much faster without large BLAS thread pools. An explicit
# user setting takes precedence over these defaults.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "HOOF_Features.zip": (
        "https://raw.githubusercontent.com/Trunpm/ACII19-Apex-Time-Network/master/HOOF/Features.zip",
        "7262353f12a035140fef8f8ce0a828ed04688a3de04e122bf7bad760da0fac7a"),
    "Apex_Cropped_images.txt": (
        "https://raw.githubusercontent.com/Trunpm/ACII19-Apex-Time-Network/master/Data/Apex_Cropped_images.txt",
        "d81d2f839ac78b15c148bdf9f786d16cebd6b5e2bc251c3731a7e5dc5c69baf1"),
}
CLASS_NAMES = ["Negative", "Positive", "Surprise"]
MODEL_NAMES = ["5-nearest neighbors", "RBF support vector machine", "Random forest",
               "Multilayer perceptron", "ResNet-18 + linear head"]
COLORS = ["#3668b2", "#18a08c", "#e48a33", "#8649a7", "#c85162"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def obtain_sources(data_dir: Path, download: bool) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, expected_hash) in SOURCES.items():
        path = data_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}. Rerun with --download or follow README.")
            print(f"Downloading {name} from the authors' repository...", flush=True)
            req = Request(url, headers={"User-Agent": "CCA1-educational-experiment"})
            with urlopen(req, timeout=120) as response:
                data = response.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Unexpected checksum for {name}; source may have changed. Got {digest}")
            path.write_bytes(data)
        digest = sha256(path)
        if digest != expected_hash:
            raise ValueError(f"Checksum mismatch for {path}: {digest}")
        paths[name] = path
    return paths


def load_casme2(archive: Path, labels_file: Path):
    """Join the original labels and HOOF features using sample ID, never row order."""
    labels = {}
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        filename, label = line.split()
        if not filename.startswith("casme2-"):
            continue
        sample_id = filename.rsplit("-img", 1)[0]
        if sample_id in labels:
            raise ValueError(f"Duplicate label: {sample_id}")
        labels[sample_id] = int(label)
    prefix = "HOOF_TIM10_8/Features/casme2-"
    with zipfile.ZipFile(archive) as zf:
        members = {}
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".mat"):
                key = Path(name).stem
                if key in members:
                    raise ValueError(f"Duplicate feature: {key}")
                members[key] = name
        if labels.keys() != members.keys():
            raise ValueError(f"Features/labels mismatch: missing={labels.keys()-members.keys()}, "
                             f"unlabeled={members.keys()-labels.keys()}")
        X, y, groups, sample_ids = [], [], [], []
        for key in sorted(labels):
            values = loadmat(io.BytesIO(zf.read(members[key])))["HOOF"]
            if values.shape != (8, 9) or not np.isfinite(values).all():
                raise ValueError(f"Unexpected feature values for {key}: {values.shape}")
            subject = key.split("-")[1]
            X.append(values.ravel())
            y.append(labels[key]); groups.append(subject); sample_ids.append(key)
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if len(y) != 145 or len(set(groups)) != 24 or dict(zip(*np.unique(y, return_counts=True))) != {0:88, 1:32, 2:25}:
        raise ValueError("The expected 145 samples, 24 subjects, or class counts did not match.")
    return X, y, groups, sample_ids


def make_models():
    # Hyperparameters specified before evaluation; scaling is fitted per training fold.
    return {
        MODEL_NAMES[0]: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        MODEL_NAMES[1]: make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced")),
        MODEL_NAMES[2]: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
        MODEL_NAMES[3]: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,),
            random_state=42, max_iter=500)),
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    }


def extract_resnet18_features(X: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """Frozen, ImageNet-pretrained standard ResNet-18 on the same HOOF data.

    The 72 numbers are an 8x9 orientation/time histogram, not facial pixels.
    Three repeated channels make the input compatible with the official
    ImageNet weights' 256-resize/224-crop preprocessing. The transformation is
    deterministic and fitted to neither the training nor the test samples.
    """
    if X.shape[1] != 72 or X.min() < 0 or X.max() > 1:
        raise ValueError("Expected normalized 8x9 HOOF histograms in [0, 1].")
    torch.set_num_threads(4)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    backbone = resnet18(weights=weights, progress=False)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    result=[]
    with torch.inference_mode():
        for start in range(0,len(X),batch_size):
            gray=torch.from_numpy(X[start:start+batch_size].astype(np.float32)).reshape(-1,1,8,9)
            rgb=gray.repeat(1,3,1,1)
            processed=preprocess(rgb)
            result.append(backbone(processed).cpu().numpy())
            print(f"ResNet feature extraction: {min(start+batch_size,len(X))}/{len(X)}",flush=True)
    features=np.concatenate(result,axis=0)
    if features.shape != (len(X),512) or not np.isfinite(features).all():
        raise ValueError(f"Invalid ResNet-18 embeddings: {features.shape}")
    torch.set_num_threads(1)
    return features


def train_linear_head(train_features, train_y, test_features):
    """Train only the 3-way ResNet head; no test labels enter training."""
    torch.manual_seed(42)
    layer=torch.nn.Linear(512,3)
    counts=np.bincount(train_y,minlength=3)
    if np.any(counts==0):raise ValueError("A training fold has a missing class")
    weights=torch.tensor(len(train_y)/(3*counts),dtype=torch.float32)
    loss_function=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.Adam(layer.parameters(),lr=0.001,weight_decay=0.0001)
    x=torch.from_numpy(train_features.astype(np.float32))
    y=torch.from_numpy(train_y.astype(np.int64))
    layer.train()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss=loss_function(layer(x),y)
        loss.backward();optimizer.step()
    layer.eval()
    with torch.inference_mode():
        return layer(torch.from_numpy(test_features.astype(np.float32))).argmax(dim=1).cpu().numpy()


def evaluate(X, y, groups, sample_ids, cnn_features):
    names=list(make_models())+[MODEL_NAMES[4]]
    predictions = {name: np.full(len(y), -1, dtype=int) for name in names}
    coverage = np.zeros(len(y), dtype=int)
    convergence_warnings=[]
    logo = LeaveOneGroupOut()
    folds = 0
    for train, test in logo.split(X, y, groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(set(groups[test])) == 1
        assert set(y[train]) == {0, 1, 2}
        coverage[test] += 1
        for name, model in make_models().items():
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always",ConvergenceWarning)
                model.fit(X[train], y[train])
            if any(issubclass(w.category,ConvergenceWarning) for w in recorded):
                convergence_warnings.append({"model":name,"subject":str(groups[test][0])})
            predictions[name][test] = model.predict(X[test])
        predictions[MODEL_NAMES[4]][test]=train_linear_head(cnn_features[train],y[train],cnn_features[test])
        folds += 1
        print(f"Completed subject fold {folds}/24",flush=True)
    assert folds == len(set(groups)) == 24 and np.all(coverage == 1)
    rows = []
    for name, pred in predictions.items():
        assert np.isin(pred, [0, 1, 2]).all()
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
            "negative_recall": recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[0],
            "positive_recall": recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[1],
            "surprise_recall": recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[2],
        })
    frame = pd.DataFrame({"sample_id": sample_ids, "subject": groups, "true_label": y})
    for name, pred in predictions.items(): frame[name] = pred
    return pd.DataFrame(rows), frame, predictions, convergence_warnings


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce3eb", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)


def make_figures(y, predictions, metrics, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":12,
                         "figure.dpi":140, "savefig.dpi":190})
    fig, ax = plt.subplots(figsize=(8, 4.25), layout="constrained")
    counts = np.bincount(y, minlength=3)
    bars = ax.bar(CLASS_NAMES, counts, color=COLORS, width=.58)
    ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts)*1.18); ax.set_ylabel("Number of clips"); style_axes(ax)
    ax.set_title("CASME II evaluation subset: 145 labeled clips", loc="left", fontweight="bold")
    ax.text(0, -.21, "Negative 88  ·  Positive 32  ·  Surprise 25  |  24 participants",
            transform=ax.transAxes, color="#526070", fontsize=9)
    fig.savefig(output/"01_class_distribution.png", bbox_inches="tight"); plt.close(fig)

    selected = metrics[metrics.model.isin(MODEL_NAMES)].set_index("model").loc[MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(10.8, 5.2), layout="constrained")
    x = np.arange(5); width = .24
    for i, (metric,label,color) in enumerate([("accuracy","Accuracy",COLORS[0]),
            ("balanced_accuracy","Balanced accuracy",COLORS[1]), ("macro_f1","Macro F1",COLORS[2])]):
        bars=ax.bar(x + (i-1)*width, selected[metric]*100, width, label=label,color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x,["5-NN", "RBF SVM", "Random forest", "MLP", "ResNet-18"]); ax.set_ylabel("Score (%)")
    ax.set_ylim(0,105); style_axes(ax)
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.00),fontsize=9)
    ax.set_title("Classifier performance on held-out subjects", loc="left", fontweight="bold", pad=30)
    fig.savefig(output/"02_model_comparison.png",bbox_inches="tight"); plt.close(fig)

    fig, axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for i,(name,ax) in enumerate(zip(MODEL_NAMES[:3],axes)):
        cm=confusion_matrix(y,predictions[name],labels=[0,1,2]); norm=cm/cm.sum(axis=1,keepdims=True)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col,row,f"{cm[row,col]}\n({norm[row,col]*100:.0f}%)",ha="center",va="center",
                        color="white" if norm[row,col]>.52 else "#172635",fontsize=10)
        ax.set_xticks(range(3),CLASS_NAMES,rotation=30,ha="right"); ax.set_yticks(range(3),CLASS_NAMES)
        ax.set_xlabel("Predicted label"); ax.set_title(["5-nearest neighbors","RBF SVM","Random forest"][i],fontweight="bold",fontsize=11)
        if i==0: ax.set_ylabel("True label")
    fig.suptitle("Out-of-fold confusion matrices  |  counts and row percentages",fontweight="bold",fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.78,label="Share of true class")
    fig.savefig(output/"03_confusion_matrices.png",bbox_inches="tight"); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(8.6,4.1),layout="constrained")
    for name,ax,title in zip(MODEL_NAMES[3:],axes,["MLP","Pretrained ResNet-18"]):
        cm=confusion_matrix(y,predictions[name],labels=[0,1,2]); norm=cm/cm.sum(axis=1,keepdims=True)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col,row,f"{cm[row,col]}\n({norm[row,col]*100:.0f}%)",ha="center",va="center",
                        color="white" if norm[row,col]>.52 else "#172635",fontsize=10)
        ax.set_xticks(range(3),CLASS_NAMES,rotation=30,ha="right"); ax.set_yticks(range(3),CLASS_NAMES)
        ax.set_xlabel("Predicted label");ax.set_title(title,fontweight="bold")
    axes[0].set_ylabel("True label")
    fig.suptitle("Neural models: held-out confusion matrices",fontweight="bold",fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.78,label="Share of true class")
    fig.savefig(output/"03b_neural_confusion_matrices.png",bbox_inches="tight");plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4.9),layout="constrained")
    x=np.arange(3)
    for i,(name,color) in enumerate(zip(MODEL_NAMES,COLORS)):
        rec=recall_score(y,predictions[name],labels=[0,1,2],average=None,zero_division=0)*100
        bars=ax.bar(x+(i-2)*.15,rec,.15,label=["5-NN","RBF SVM","Random forest","MLP","ResNet-18"][i],color=color)
        ax.bar_label(bars,fmt="%.0f",padding=2,fontsize=7)
    ax.set_xticks(x,["Negative (n=88)","Positive (n=32)","Surprise (n=25)"])
    ax.set_ylabel("Recall (%)"); ax.set_ylim(0,110); style_axes(ax)
    ax.legend(frameon=False,ncol=5,loc="upper center",bbox_to_anchor=(.5,1),fontsize=8)
    ax.set_title("Recall by class",loc="left",fontweight="bold",pad=28)
    fig.savefig(output/"04_class_recall.png",bbox_inches="tight"); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data",help="Directory for original feature archive and labels")
    p.add_argument("--output-dir",type=Path,default=ROOT/"results")
    p.add_argument("--download",action="store_true",help="Download source features and labels if missing")
    args=p.parse_args()
    paths=obtain_sources(args.data_dir,args.download)
    X,y,groups,ids=load_casme2(paths["HOOF_Features.zip"],paths["Apex_Cropped_images.txt"])
    print(f"Loaded {len(y)} samples, {len(set(groups))} subjects, {X.shape[1]} features.",flush=True)
    cnn_features=extract_resnet18_features(X)
    metrics,predictions_frame,predictions,convergence_warnings=evaluate(X,y,groups,ids,cnn_features)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(args.output_dir/"metrics.csv",index=False,float_format="%.8f")
    predictions_frame.to_csv(args.output_dir/"out_of_fold_predictions.csv",index=False)
    reports={name:classification_report(y,pred,labels=[0,1,2],target_names=CLASS_NAMES,
                     output_dict=True,zero_division=0) for name,pred in predictions.items()}
    metadata={"samples":len(y),"subjects":len(set(groups)),"features_per_sample":X.shape[1],
              "class_counts":dict(zip(CLASS_NAMES,np.bincount(y,minlength=3).tolist())),
              "protocol":"24-fold leave-one-subject-out; pooled out-of-fold metrics",
              "numpy_version":np.__version__,"sklearn_version":sklearn.__version__,
              "torch_version":torch.__version__,
              "cnn_backbone":"torchvision ResNet-18, IMAGENET1K_V1/DEFAULT; frozen; 512-feature output",
              "cnn_input":"8x9 HOOF histogram repeated to three channels; official ImageNet transform",
              "cnn_head":"3-way linear layer trained per fold; Adam lr=0.001, weight_decay=0.0001, 200 full-batch epochs; balanced loss",
              "mlp":"scikit-learn defaults with hidden_layer_sizes=(100,), random_state=42, max_iter=500; training-fold StandardScaler",
              "convergence_warnings":convergence_warnings,
              "source_sha256":{name:sha256(path) for name,path in paths.items()},"classification_reports":reports}
    (args.output_dir/"experiment_details.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    make_figures(y,predictions,metrics,args.output_dir/"figures")
    print(metrics.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"Saved results to {args.output_dir}")

if __name__ == "__main__": main()
