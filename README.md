# Final Project: Hierarchical Multi-Label Text Classification

**DATA304 – Big Data Analysis**

본 저장소는 **Hierarchical Multi-Label Text Classification (HMTC)** 최종 프로젝트의 전체 코드 및 결과를 포함합니다.
본 프로젝트는 **라벨이 없는 Amazon product review 데이터**와 **주어진 taxonomy 및 class keyword 정보만을 사용한 weakly-supervised / unsupervised setting**에서 수행되었습니다.

---

## 📁 Repository Structure

```
.
├── Amazon_products/              # Provided dataset (train/test, taxonomy, keywords)
├── Dump/                         # Experimental / deprecated files (not used for final result)
├── References/                   # Reference paper (2021.naacl-main.335)
├── Outputs/                      # Final outputs and Kaggle submission files
│
├── 00. LLM Embeddings.ipynb      # LLM-based class/text embedding generation
├── 01. Making_Embeddings.ipynb   # Sentence embedding construction
├── 02. Core_Class_Mining.ipynb   # Core class mining (TaxoClass-style)
├── 03. MLP_Training.ipynb        # MLP classifier training (+ self-training)
├── 04. GAT_Training.ipynb        # GAT-based label encoder training
├── 04. GAT_Training(LLM).ipynb   # GAT training with LLM-enhanced labels
├── 05. GAT_momentum.ipynb        # GAT + momentum / stabilization experiment
│
├── dummy_baseline.ipynb          # Provided random / TF-IDF baseline
├── Final_Project_Report_2024320333.pdf
├── final_project.pdf             # Project description (provided)
└── README.md
```

---

## ⚙️ Environment

* Python ≥ 3.9
* PyTorch
* NumPy
* scikit-learn
* transformers
* tqdm

GPU is **optional**. All notebooks automatically fall back to CPU if CUDA is unavailable.

---

## 🎯 Reproducibility

This repository is organized to **reproduce the final Kaggle result step-by-step**.
All random seeds are explicitly fixed as required by the project guideline.

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## ▶️ How to Reproduce the Final Result

### Step 0. Dataset Preparation

The dataset is already placed under:

```
Amazon_products/
```

No additional data is used.

---

### Step 1. Embedding Construction

1. **(Optional) LLM-based embedding**

   ```
   00. LLM Embeddings.ipynb
   ```
2. Sentence / document embedding generation

   ```
   01. Making_Embeddings.ipynb
   ```

---

### Step 2. Silver Label Generation (Core Class Mining)

```
02. Core_Class_Mining.ipynb
```

This notebook implements:

* Top-down taxonomy traversal
* Core class candidate selection
* Confidence-based core class filtering

---

### Step 3. Model Training

* **MLP-based classifier**

  ```
  03. MLP_Training.ipynb
  ```
* **GAT-based classifier**

  ```
  04. GAT_Training.ipynb
  04. GAT_Training(LLM).ipynb
  ```
* **GAT + Momentum stabilization**

  ```
  05. GAT_momentum.ipynb
  ```

Self-training is explicitly implemented and logged.

---

### Step 4. Kaggle Submission File

The final Kaggle submission file is generated in:

```
outputs/
```

The submission file follows the required format:

```
studentID_final.csv
```

This file corresponds to the **best-performing model reported in the report**.

---

## 📊 Experimental Results

The following variants are implemented and reported:

* MLP only
* MLP + Self-Training
* GAT only
* GAT + Self-Training
* GAT + Momentum
* GAT + Momentum + Self-Training

All results are summarized in:

```
Final_Project_Report_2024320333.pdf
```

---

## 🤖 LLM Usage Disclosure

* LLMs were used **only for auxiliary purposes** (e.g., class representation enhancement).
* Total API calls ≤ **1,000**, complying with the project policy.
* All LLM prompts and outputs were saved and are reproducible.
* No Amazon review text was directly provided to the LLM.

---

## ⚠️ Notes

* No external data beyond the provided dataset was used.
* No pretrained model fine-tuned on Amazon product data was used.
* The API Keys are **NOT** in the git (.env) - only the results are saves as *Amazon_products\class_llm_texts.txt*

---

## 👤 Author

* Student ID: **2024320333**
* GitHub ID: **mjkoo1226**

---

If any issue arises during reproduction, please follow the notebook order strictly as listed above.
