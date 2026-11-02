# CCA 1: Micro-expression recognition

This project compares five classifiers on a real CASME II micro-expression subset. It retains the original three approaches and adds both an MLP neural network and a standard pretrained ResNet-18 CNN.

## The five approaches

| Approach | Input | Configuration |
|---|---|---|
| 5-nearest neighbors | 72 HOOF values | k=5, uniform voting; fold-specific scaling |
| RBF SVM | 72 HOOF values | C=1, gamma=scale, balanced class weights; fold-specific scaling |
| Random forest | 72 HOOF values | 200 trees, balanced class weights, seed 42 |
| MLP neural network | 72 HOOF values | scikit-learn MLPClassifier, one hidden layer of 100 units, Adam, seed 42, max_iter=500; fold-specific scaling |
| ResNet-18 CNN | The same 8x9 HOOF values represented as a three-channel image | torchvision's standard ResNet-18 with `ResNet18_Weights.DEFAULT`; pretrained backbone frozen; a new three-class linear head trained separately for each fold |

A majority-class predictor is included as an additional reference baseline. The 145 clips come from 24 people, and have three dataset labels: negative (88), positive (32), surprise (25). The HOOF motion features were provided by Peng et al.; our code implements the classifiers and evaluation, not raw-video processing.

### What the CNN actually sees

The available data contain **motion descriptors, not face photographs**. For each clip, the 72 values form an 8x9 orientation-by-time histogram. For the CNN, the matrix is repeated into three identical channels and passed through the *unmodified, pretrained weights' standard torchvision transforms*: resize the short edge to 256, center-crop to 224x224, and ImageNet channel normalization. The center crop removes some outer histogram area. A frozen ResNet-18 extracts 512 features; a trainable 3-class linear head is fitted on each fold using class-weighted cross-entropy and Adam (learning rate 0.001, weight decay 0.0001, 200 full-batch epochs). These head-training choices are fixed for the experiment and are not claimed to be defaults. The backbone is standard ResNet-18, not a custom CNN.

This is an **exploratory transfer-learning baseline on feature matrices**. The ImageNet-pretrained backbone was designed for natural images, so domain mismatch and resizing/cropping can discard or distort histogram information. It is not a facial-image CNN, a raw video detector, or a reproduction of MADCNN. For a facial CNN comparable to the user's reference, access to suitably licensed face frames or image sequences and a new evaluation would be needed.

All models use the same 145 clip identities and 24 leave-one-subject-out splits. CNN and MLP receive different transformations and the CNN has ImageNet pretraining, so differences cannot be attributed to classifier architecture alone. Hyperparameters were not tuned. The neural models were added after inspecting the earlier baseline results, so treat the comparison as exploratory. All metrics are calculated from pooled out-of-fold predictions, not the average of per-subject fold scores.

## Files

```text
src/run_experiment.py                    all five methods, evaluation and plotting
requirements.txt                         pinned CPU experiment dependencies
requirements-torch-cpu.txt               compatible PyTorch and torchvision versions
results/metrics.csv                      scores for five methods and the baseline
results/out_of_fold_predictions.csv      each clip's held-out predictions
results/experiment_details.json          data, model configurations and warnings
results/figures/                         charts and confusion matrices
Viva_Notes.md                            short explanations for an oral discussion
```

## Reproduce the results on Linux

Python 3.12 and an internet connection are required. These commands use `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python src/run_experiment.py --download
```

The first run downloads the source HOOF archive and label file into `data/` and checks the original SHA-256 digests. It also downloads the official ResNet-18 weights (about 45 MB) through torchvision's normal model cache. The data and weights are not bundled in the ZIP. Later runs can omit `--download`. On Windows, activate the virtual environment with `.venv\Scripts\activate`; PyTorch installation commands for other platforms are available at https://pytorch.org/get-started/locally/.

Alternatively download `HOOF/Features.zip` and `Data/Apex_Cropped_images.txt` from the [original feature repository](https://github.com/Trunpm/ACII19-Apex-Time-Network), saving them as `data/HOOF_Features.zip` and `data/Apex_Cropped_images.txt`. Then run `python src/run_experiment.py`. The loader checks their expected hashes, sample IDs and dimensions and rejects mismatches.

The repository makes precomputed features available but does not state a separate data license for them. The original CASME II database has a research-access process. Check the applicable terms before redistributing data or publishing results. This run used Python 3.12.14, NumPy 2.3.5, scikit-learn 1.8.0, PyTorch 2.8.0+cpu and torchvision 0.23.0+cpu. Small platform-dependent numerical differences are possible.

## Measured results

| Approach | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| 5-nearest neighbors | 59.31% | 36.47% | 33.96% |
| RBF SVM | 48.97% | 40.79% | 40.23% |
| Random forest | 58.62% | 33.15% | 27.04% |
| MLP neural network | 50.34% | 36.03% | 36.21% |
| Pretrained ResNet-18 + head | 55.17% | 51.04% | 49.52% |
| Majority-class baseline | 60.69% | 33.33% | 25.18% |

On this small evaluation, the ResNet-18 setup has the highest macro F1 and balanced accuracy. The majority baseline has the highest raw accuracy by predicting negative for all samples. These findings do not establish robust facial-emotion recognition or generalization beyond this dataset.

## Sources

1. Yan et al. (2014), CASME II, *PLOS ONE*, https://doi.org/10.1371/journal.pone.0086041.
2. Peng et al. (2019), "A Novel Apex-Time Network for Cross-Dataset Micro-Expression Recognition," https://doi.org/10.1109/ACII.2019.8925525; features: https://github.com/Trunpm/ACII19-Apex-Time-Network.
3. MEGC 2019, Recognition Challenge guideline, https://facial-micro-expressiongc.github.io/MEGC2019/images/MEGC2019%20Recognition%20Challenge.pdf.
4. He et al. (2016), "Deep Residual Learning for Image Recognition," https://doi.org/10.1109/CVPR.2016.90. Torchvision documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.
5. Barrett et al. (2019), "Emotional Expressions Reconsidered," https://doi.org/10.1177/1529100619832930.
