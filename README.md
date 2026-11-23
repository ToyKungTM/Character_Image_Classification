# Character_Image_Classification
# Zenless Zone Zero Character Classification with AlexNet

An image classification project that identifies 10 characters from the game Zenless Zone Zero. It uses AlexNet (PyTorch) pretrained on ImageNet and fine-tuned on the character images. The code was written and run on Google Colab.

This is a learning project and is not affiliated with the game's publisher.

## Characters

Bella, Burnice, Ellen, Grace, JaneDoe, King Caesar, Miyabi, Nicole, Wise, Yanagi

## Data and Splits

- About 100 images per character, labeled by folder name
- Split with Scikit-learn using stratified sampling so every class keeps the same proportion

| Set | Share | Purpose |
|---|---|---|
| Train | 70% | Fit the model |
| Validation | 15% | Drives early stopping and learning rate reduction |
| Test | 15% | Final evaluation, used once at the end |

## Method

- Model: AlexNet from torchvision (ImageNet weights), with the last layer replaced to match the number of classes
- Augmentation (training set only): horizontal flip, rotation up to ±15 degrees, brightness / contrast / saturation jitter
- Training: Adam, learning rate 0.0001, batch size 32, up to 25 epochs, 224×224 input
- Overfitting control:
  - Early stopping: stops when validation loss does not improve for 5 epochs, then restores the best weights
  - Learning rate reduction: halves the learning rate when validation loss does not improve for 3 epochs
- Metrics: Accuracy, Precision, Recall, F1 score, and Confusion Matrix

## Results

Measured on the test split, which was drawn from the same pool of images as the training set.

| Experiment | Test images | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| 10 classes | 150 | 0.953 | 0.954 | 0.953 | 0.953 |
| 11 classes (added "Other") | 165 | 0.879 | 0.885 | 0.879 | 0.880 |

The second experiment adds an "Other" class (100 images of characters outside the 10) so the model can reject unknown characters. Overall accuracy dropped, and the Other class itself scored Precision 0.471 and Recall 0.533.

## Known Limitations

- The 10-class model has no "none of the above" answer. When given a character outside the group, it still picks one of the 10 characters, sometimes with high confidence (96.3% in one test).
- The confidence shown is the model's confidence for that image, not its overall accuracy, and it can be overconfident.
- The test split comes from the same pool as the training data, so results on new images from the internet may be worse than the numbers above.
- The test set has only 15 images per class, so per-class numbers fluctuate easily.

## How to Use

1. Prepare an image folder with this structure (images are not included in this repository):

```
dataset/
├── Bella/
├── Burnice/
├── Ellen/
└── ... (folder name = character name)
```

2. Open `zzz_classifier.ipynb` in Google Colab.
3. Enable the GPU: Runtime → Change runtime type → GPU.
4. Set `DATA_DIR` in the first cell to point to your image folder.
5. Run the cells in order. The last cell loads the saved model and lets you upload an image to see the predicted character.

## Libraries

torch, torchvision, scikit-learn, matplotlib, numpy, Pillow (all preinstalled on Google Colab)

## Notes

- The dataset is not included because the images are copyrighted by the game's publisher and the original artists.
- The trained model file (`.pth`) is not included because it exceeds GitHub's file size limit.
