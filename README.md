# Machine Learning – ANN vs. SVM

---

## 1 Overview

Binary‑classification homework comparing **Artificial Neural Networks (ANN)** and **Support Vector Machines (SVM)**.  
Training loops run for **100 / 1 000 / 10 000** epochs with **early stopping**, so the full 10 000 may not be reached.

> **Tip** – If you see “`./data/dataset.pkl` not found”, open the **project root** (see screenshot _How_the_Folder_Should_Look.png_) in VS Code / your IDE. Running from a different CWD will break the relative paths.

---

## 2 Methods

### 2.1 ANN

- 1, 2 and 3 hidden‑layer architectures
- Optimisers: **SGD** and **Mini‑Batch GD**
- Training/validation loss and decision boundaries plotted
- Best model selected & tested

### 2.2 SVM

- Kernels: **Linear**, **Polynomial** (varying degree), **Gaussian RBF** (varying γ)
- Decision boundaries plotted for all variants
- Best model selected & tested

---

## 3 Requirements

| Library      | Version                                |
| ------------ | -------------------------------------- |
| numpy        | 2.2.1                                  |
| pandas       | – (comes with sklearn)                 |
| matplotlib   | 3.10.0                                 |
| seaborn      | 0.13.2                                 |
| scikit‑learn | 1.6.0                                  |
| keras        | 3.7.0                                  |
| tensorflow   | 2.18.0 (`tensorflow-intel` also works) |

Install:

```bash
pip install -r requirements.txt
```

GPU on macOS → add `tensorflow-metal`.

---

## 4 Repo Layout

```
project_root/
├── data/
│   ├── dataset.pkl
│   └── train_test_validate.pkl
├── results/
│   ├── nn/   # ANN plots + CM
│   └── svm/  # SVM confusion matrices
├── DataHandler.py
├── NeuralNetwork.py
├── NeuralNetworkHandler.py
├── SupportVectorMachine.py
├── main.py
└── requirements.txt
```

---

## 5 Run

```bash
python main.py
```

The script will create splits if absent and write all artefacts to **`results/`**.

---

## 6 Outputs

- `results/nn/decision-boundary/*` – decision boundary PNGs per epoch setting
- `results/nn/loss/*` – loss curves
- `results/nn/confusion-matrix/*` – confusion matrices
- `results/svm/confusion-matrix/` – SVM matrices
- `results/program_output.txt` – full console log (auto‑generated; replicate with `python main.py > program_output.txt`)

---

## 7 Notes

- Dataset auto‑generated & cached under `data/`.
- Code is modular – each algorithm in its own class.
- Output files are appended on new runs.

---

## 8 Author

Tuğcan Topaloğlu
