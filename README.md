# SentimentAnalysis - Depression Detection from Text using BERT

Classifying tweets as depressed and not depressed.
https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets

This project uses a fine-tuned BERT model to classify text data as depressed or not depressed, targeting early detection of mental health risks. Built using PyTorch and HuggingFace Transformers, it includes a full training-validation pipeline with RMSE-based regression loss and classification reporting.

Given a labeled dataset of social media or conversational texts, the goal is to classify each sample into:

- Depressed (1)

- Not Depressed (0)

This binary classification task supports healthcare AI systems in mental health screening.



## Dataset
The dataset is assumed to be a CSV file with the following columns:

- text: the raw user-generated sentence

- label: 1 for depressed, 0 for not depressed

From the full dataset, a subset of 5000 samples is randomly selected. A Stratified 5-Fold split is used to ensure balanced representation of both classes.

## Technologies Used

| Area           | Tool/Library                                 |
| -------------- | -------------------------------------------- |
| Language Model | `bert-base-uncased` (HuggingFace)            |
| Deep Learning  | PyTorch (`torch`, `torch.nn`, AMP)           |
| Tokenization   | HuggingFace `transformers` tokenizer         |
| Evaluation     | Scikit-learn (`classification_report`, RMSE) |
| Visualization  | `matplotlib`                                 |
| Data Handling  | `pandas`, `numpy`                            |


## 🚀 Pipeline Overview

### 1. Preprocessing & Tokenization
- Text data is tokenized using `BertTokenizer.encode_plus`.
- Fixed max sequence length: `max_sens = 16`.
- Prepares:
  - Input IDs
  - Attention masks
  - Token type IDs

### 2. Custom PyTorch Dataset & Dataloader
- `BERTDataSet` class wraps the encoded text and target labels.
- PyTorch `DataLoader` is used for efficient batching and shuffling.

### 3. Model Initialization
- Uses `BertForSequenceClassification` from HuggingFace with `num_labels=1`.
- Single output neuron for binary classification (depressed vs. not depressed).
- Output interpreted using a 0.5 threshold.

### 4. Training Loop
- Mixed-precision training enabled via `torch.cuda.amp` for speed and memory efficiency.
- Optimizer: `AdamW` with `weight_decay=1e-2`.
- Learning rate scheduler: `get_linear_schedule_with_warmup` for smooth warmup.
- Loss Function: Root Mean Squared Error (RMSE) using `nn.MSELoss` (supports regression-like training).
- Best model checkpoint is saved based on lowest validation RMSE.

### 5. Evaluation
- Predictions are binarized using a threshold of 0.5.
- `classification_report` from scikit-learn provides:
  - Precision
  - Recall
  - F1-score
  - Support per class
- Generates plots:
  - **Loss vs Epoch**
  - **Score vs Epoch**


