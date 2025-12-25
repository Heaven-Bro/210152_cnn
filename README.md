# Geometric Shapes CNN – 210152

CNN image classification project for triangles, squares, and circles, implemented in PyTorch and trained on a custom geometric shapes dataset plus real-world phone photos. [file:1]

---

## Dataset

- **Main training data**: Hand‑drawn geometric shapes in three classes: **circles, squares, triangles**.  
- Directory structure:

  - `data/train/circles`, `data/train/squares`, `data/train/triangles`  
  - `data/val/circles`, `data/val/squares`, `data/val/triangles`  
  - `data/test/circles`, `data/test/squares`, `data/test/triangles`  

- **Custom phone dataset** (real‑world test): 10 smartphone photos stored in `dataset/` (mix of circles, squares, triangles drawn on paper). [file:1]

All images are converted to grayscale and resized to **64×64** pixels, then normalized with mean 0.5 and std 0.5 using `torchvision.transforms`. [file:1]

---

## Model architecture

Implemented in `CNN` class (`torch.nn.Module`):

- Input: \(1 \times 64 \times 64\) grayscale images.  
- Feature extractor:
  - Conv2d(1, 32, kernel=3, padding=1) + ReLU + MaxPool2d(2)  
  - Conv2d(32, 64, kernel=3, padding=1) + ReLU + MaxPool2d(2)  
  - Conv2d(64, 128, kernel=3, padding=1) + ReLU + MaxPool2d(2)  
- Classifier:
  - Flatten  
  - Linear(128×8×8 → 256) + ReLU  
  - Linear(256 → 3) for the three shape classes. [file:1]

Loss function is **CrossEntropyLoss**, optimizer is **Adam** with learning rate 0.001, trained for **15 epochs** with batch size 64. [file:1]

---

## Training procedure

The Colab notebook:

1. Uses `!git clone https://github.com/Heaven-Bro/210152_cnn.git` to automatically download the repo and data. [file:1]  
2. Loads training/validation/test sets via `torchvision.datasets.ImageFolder` and `DataLoader`. [file:1]  
3. Trains the CNN for 15 epochs, tracking **loss** and **accuracy** on both training and validation sets.  
4. Saves the trained weights to `model/210152.pth` using `torch.save(model.state_dict(), ...)`. [file:1]

---

## Results

- Final **training accuracy** ≈ **84%**, **validation accuracy** ≈ **88%** on the shapes dataset.  
- Training loss decreases from about **1.17 → 0.40**, validation loss from **1.10 → 0.32**, showing good convergence. [file:49][file:50]  
- Confusion matrix shows high correct counts for all three classes, with most mistakes between circles and triangles. [file:52]  
- On the 10 phone images, most triangles and squares are predicted correctly with confidence above **99%**, while a few circle/square drawings are misclassified as triangles. [file:53]

The notebook includes plots of **Loss vs Epochs**, **Accuracy vs Epochs**, the **confusion matrix**, misclassified examples, and the **phone prediction gallery**. [file:49][file:50][file:51][file:52][file:53]

---

## Evaluation and visuals

1. **Confusion Matrix** on the test set  
   - Diagonal entries are high for all three classes.  
   - Most errors are circles misclassified as triangles or squares.

   ![Confusion Matrix](https://github.com/user-attachments/assets/22c1d433-dca7-404e-8ea8-b293299cc1b4)

2. **Misclassified examples**  
   - Shows 3 test images where the model predicted the wrong class, with titles like  
     `True: circles, Pred: squares`.

   ![Misclassified Examples](https://github.com/user-attachments/assets/bd496cfc-7e54-4181-a4e5-46eac5a28757)

3. **Custom Prediction Gallery (phone images)**  
   - Loads all images from `dataset/`.  
   - Applies the same transforms as training (grayscale, resize, normalize).  
   - Runs them through the model in `eval()` mode and uses `torch.softmax` to get probabilities.  
   - Displays a grid of the 10 photos with titles `Pred: <class> (<confidence>%)`. [file:1]

   ![Phone Prediction Gallery](https://github.com/user-attachments/assets/c54ea116-7a98-4633-ae71-1b6c96a227a4)

Example predictions:

- Some triangles and squares are classified correctly with confidence above **99%**.  
- A few circle/square drawings are misclassified as triangles, which is discussed in the error analysis. [file:53]

---

## How to run (Colab)

1. Open the Colab notebook (link provided in submission).  
2. Click **Runtime → Run all**.  
3. The notebook will automatically:

   - Clone this repository.  
   - Load the geometric shapes dataset from `data/`.  
   - Train the CNN (or load `model/210152.pth` if already saved).  
   - Evaluate on the test set and show the confusion matrix and misclassified samples.  
   - Load the 10 phone images from `dataset/` and output the predictions plus the gallery. [file:1]

No manual uploads or path changes are required; everything runs end‑to‑end from the GitHub repo as specified in the assignment. [file:1]
