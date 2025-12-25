text
# Geometric Shapes CNN – 210152

CNN classifier for **circles**, **squares**, and **triangles** using PyTorch, trained in Colab and tested on 10 phone photos of hand‑drawn shapes. [file:1]

## Dataset

- `data/train`, `data/val`, `data/test` with subfolders: `circles/`, `squares/`, `triangles/`.  
- `dataset/` contains 10 smartphone images (3 circles, 4 squares, 3 triangles).  
- All images are grayscale, resized to **64×64**, and normalized with mean 0.5, std 0.5 using `torchvision.transforms`. [file:1]

## Model

- CNN: 3 Conv2d+ReLU+MaxPool blocks → Flatten → Linear(128×8×8→256) → ReLU → Linear(256→3).  
- Loss: CrossEntropyLoss, Optimizer: Adam(lr=0.001), batch size 64, 15 epochs. [file:1]  
- Trained weights saved as `model/210152.pth`. [file:1]

## Training & Evaluation

- Notebook starts with:

!git clone https://github.com/Heaven-Bro/210152_cnn.git
%cd 210152_cnn

text

and loads data via `ImageFolder` + `DataLoader` (no manual uploads). [file:1]  
- Final accuracy ≈ **84% train**, **88% validation**; loss decreases smoothly over epochs (plots included). [file:49][file:50]  
- Confusion matrix and 3 misclassified test images are shown to analyze errors. [file:51][file:52]

## Real‑World Phone Images

- All images from `dataset/` are preprocessed with the same transforms and passed through the saved model in `.eval()` mode with `torch.softmax` to get probabilities. [file:1]  
- Notebook prints predictions like  
`dataset/circle1.jpeg -> triangles (99.89%)`  
and displays a 2×5 gallery with `Pred: <class> (<confidence>%)` captions. [file:53]

## Repo Structure

210152_cnn/
data/train|val|test/...
dataset/...
model/210152.pth
210152_cnn.ipynb
README.md
plots/ (training curves, confusion matrix, gallery)

text

This satisfies the assignment requirements for automated cloning, training, saved model, visuals, and real‑world testing. [file:1]
