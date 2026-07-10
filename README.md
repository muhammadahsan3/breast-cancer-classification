# 🧬 Deep Neural Network for Breast Cancer Classification

A deep learning project that classifies breast cancer tumors as **Benign** or **Malignant** using a custom-built Neural Network in **PyTorch**, trained on the Wisconsin Breast Cancer Diagnostic dataset.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical. This project builds a simple but effective Deep Neural Network that learns from diagnostic measurements and predicts whether a tumor is benign or malignant.

---

## 🗂️ Project Structure

```
├── Deep_Neural_Network_for_Breast_Cancer_Classification.ipynb  # Main notebook
├── requirements.txt                                             # Python dependencies
├── .gitignore                                                   # Files to ignore in Git
└── README.md                                                    # Project documentation
```

---

## 📁 Dataset

- **Source**: [UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Features**: 30 numeric diagnostic measurements (e.g. radius, texture, perimeter, area, smoothness)
- **Target**: `B` = Benign → `0`, `M` = Malignant → `1`
- **Balanced Subset**: 200 Benign + 200 Malignant samples (randomly sampled)
- **Download**: Automatically fetched in the notebook via `ucimlrepo`

---

## 🧠 Model Architecture

```
Input Layer  →  30 features
Hidden Layer →  64 neurons  (ReLU activation)
Output Layer →  2 neurons   (Benign / Malignant)
```

```python
ClassificationNet(
  (fc1): Linear(in_features=30, out_features=64)
  (fc2): Linear(in_features=64, out_features=2)
)
```

---

## ⚙️ Project Workflow

1. **Load Dataset** — Fetched directly from UCI ML Repository using `ucimlrepo`
2. **Explore Data** — Visualize class distribution (imbalanced → balanced)
3. **Balance Dataset** — Sample 200 Benign + 200 Malignant randomly
4. **Preprocess** — Label encode targets, train/test split (80/20), StandardScaler normalization
5. **Convert to Tensors** — Wrap data in PyTorch `TensorDataset` and `DataLoader`
6. **Build Model** — 2-layer fully connected neural network
7. **Train with Adam** — 10 epochs, CrossEntropyLoss, lr=0.001
8. **Train with SGD** — Compare performance using SGD + momentum optimizer
9. **Visualize** — Plot training vs test loss curves for both optimizers

---

## 🔍 Optimizer Comparison

| Optimizer | Learning Rate | Extras |
|-----------|--------------|--------|
| Adam | 0.001 | — |
| SGD | 0.001 | momentum=0.9, weight_decay=0.0001 |

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- ucimlrepo

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Deep_Neural_Network_for_Breast_Cancer_Classification.ipynb
```

> The dataset is downloaded automatically — no manual download needed!

---

## 🚀 Future Improvements

- Add more hidden layers and experiment with dropout regularization
- Try different activation functions (Leaky ReLU, Sigmoid)
- Add accuracy, precision, recall, and F1-score metrics
- Deploy as a simple web app using Streamlit

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
