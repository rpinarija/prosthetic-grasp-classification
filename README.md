# Hand-View Grasp Classification for Vision-Guided Prosthetics

Deep-learning models that predict how a person would grasp an object (open palm, medium wrap, power sphere,
parallel extension, or palmar pinch) from the image of a camera mounted on the hand. A prosthetic hand
with a palm camera could use this to choose a grasp automatically.

Graduate final project for DATA 6610 Deep Learning (Fall 2024), revisited and rebuilt in 2026.
Built on the public **HANDS dataset** from Northeastern University (Han et al., 2020).

## Results

5-fold group cross-validation (mean ± std across folds). Accuracy is measured against each image's own human label,
the same metric as the original version.

| Model | v1 (Fall 2024, single split) | **v2 (cross-validated)** | v2: matches majority vote | v2: a grasp some labeler chose |
|---|---|---|---|---|
| EfficientNetB0 | — | **49.1 ± 4.5%** | 60.5 ± 5.9% | 97.2% |
| ResNet50 | 33.3% | **49.0 ± 4.0%** | 57.9 ± 3.3% | 95.9% |
| MobileNetV2 | 44.6% | **48.0 ± 3.7%** | 57.6 ± 1.8% | 94.4% |
| Custom CNN | 33.4% | **40.6 ± 6.8%** | 45.4 ± 9.8% | 91.1% |
| *Majority-class baseline* | | *27.9%* | | |
| *Label-noise ceiling* | | *61.1%* | | |

**How to read this:** the 11 human labelers agree on a grasp only about 61% of the time. No model can exceed that
ceiling against individual labels, so the pretrained models close **60–64% of the gap** between the
majority-class baseline and the best achievable score. The three pretrained models are within one standard deviation
of each other. MobileNetV2 matches the others at a fraction of the size, which matters for an embedded prosthetic.

### Benchmark against human agreement

The fairest benchmark when people disagree: for each image, how often does a predictor match the majority choice of the
*other* labelers? A single human labeler, predicting the other ~10, scores **57.4%**.

| Predictor | Matches majority of other labelers | Macro-F1 (4 grasps) |
|---|---|---|
| EfficientNetB0 | **63.5 ± 5.2%** | 0.56 |
| ResNet50 | 61.0 ± 3.3% | 0.54 |
| MobileNetV2 | 60.7 ± 2.1% | 0.53 |
| *Single human labeler* | *57.4 ± 4.5%* | *0.51* |
| Custom CNN | 48.8 ± 8.5% | 0.38 |
| *Always the most common grasp* | *33.6 ± 2.9%* | *0.12* |

The pretrained models predict the consensus grasp at **roughly human-level agreement**, from the hand-camera image alone
(the humans labeled from an eye-level view). They learned the group's average preference from all 11 labelers, which
naturally predicts the group better than any one individual. So this shows human-level consensus prediction, not
better-than-human judgment.

## The v1 → v2 story

My original model reached 44.6% at best, far below the 89% reported in the original paper. Auditing my own code turned up
three kinds of problems:

- **Bugs:** the pretrained models received the wrong input preprocessing (`/255` instead of each model's ImageNet format),
  so ResNet50 learned close to nothing. The training data was never shuffled, and the learning-rate schedule never triggered.
- **Data problems:** 64 images were blank (failed segmentation in the dataset), and 4 objects were split across groups,
  so some object views leaked between training and test.
- **A measurement problem:** the paper's 89% is likely not comparable. It used sigmoid outputs with binary cross-entropy,
  where Keras scores accuracy over each of the 5 outputs separately, and a model that predicts nothing at all scores 80%.
  The dataset's own train/test split was also drawn over images rather than objects. v2 reports this metric too (`binary_acc` ≈ 0.82).

v2 fixes all of these and adds:

- **Soft labels:** each image is trained on the distribution of all 11 labelers' votes, not one person's choice
- **Object-centered crops** from the dataset's bounding boxes, instead of shrinking the full frame
- **Two-stage transfer learning:** train the new head, then fine-tune the top backbone layers at a 100× lower learning rate
- **Clipped class weights** for the heavy class imbalance (one class has only 8 images)
- **Leakage-safe evaluation:** 5-fold `StratifiedGroupKFold` over 103 physical objects, with an inner validation split
  for early stopping, plus confusion matrices, per-class reports, a majority baseline and the label-noise ceiling

## Repository contents

| File | Description |
|---|---|
| [`data6610_final_project-v1.ipynb`](data6610_final_project-v1.ipynb) | **v1:** the original Fall 2024 submission, unchanged, with a header listing its known issues |
| [`data6610_final_project-v2.ipynb`](data6610_final_project-v2.ipynb) | **v2:** the rebuilt pipeline, results, and discussion |
| `results/` | Per-model, per-fold metrics and test predictions from the v2 run, so the notebook's tables and charts render without retraining |

## Running it

1. **Get the data.** Download the HANDS dataset from the
   [Northeastern University Digital Repository](http://hdl.handle.net/2047/D20294524) and extract it next to the notebooks
   as `Dataset_20181026/`. v2 expects `Dataset_20181026/Labels/Label_Complete.csv` and the training images in
   `Dataset_20181026/TrainingImages/` (the download names this folder `TraningImages`; rename it or edit `IMAGE_DIR`).
2. **Install the dependencies** (developed with Python 3.12 and TensorFlow 2.18):
   ```bash
   pip install tensorflow==2.18 scikit-learn pandas numpy matplotlib seaborn pillow
   ```
3. **Run `data6610_final_project-v2.ipynb` top to bottom.** The first run caches the image crops (~1 minute).
   Folds already saved in `results/` are skipped, so training only runs for missing ones.
   A full run from scratch took about 10 hours on CPU (native-Windows TensorFlow has no GPU support).
   Set `QUICK_RUN = True` for a short pass.

## Data and acknowledgments

The HANDS dataset is not included in this repository. It is available for academic, non-commercial research under
its own terms of use. The research question and dataset come from the original authors; this project is an independent
reimplementation and extension of their work.

> Han, M., Günay, S. Y., Schirner, G., Padır, T., & Erdoğmuş, D. (2020). HANDS: a multimodal dataset for modeling toward
> human grasp intent inference in prosthetic hands. *Intelligent Service Robotics*, 13(1), 179–185.
> https://doi.org/10.1007/s11370-019-00293-8
