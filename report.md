# Project Report

## 1. Overview
In this project, our goal is to build an effective model for the **UA Locations Extractions** competition. The task is to recognize location entities—cities, regions, countries, and address fragments—in Ukrainian, Russian, and mixed-code texts.

Since the competition metric is the **F1-score**, we focus on achieving the best balance between precision (avoiding false positives) and recall (capturing as many true locations as possible). The challenge lies in handling the complex morphology and variability of RU/UA languages, but our aim is to design a model that performs reliably across all text types.


## 2. Data Insights

### 2.1 Dataset Description

The test dataset consists of messages from a Kyiv news Telegram chat, containing a mixture of Ukrainian, Russian, and transliterated text. After analyzing the language distribution, we found that the messages are still mostly Ukrainian. Because of this, we decided to train our model on a Ukrainian-focused NER dataset.

<img width="720" src="https://github.com/user-attachments/assets/0d269eeb-d0df-4802-8da5-4ba85871d188"/>

For training, we use the **uk_geo_dataset**, a Ukrainian NER corpus from Corpora Ukrainian. It contains approximately **1M text samples** with annotated **location** and **organization** entities, which makes it suitable for our task.

We also had a **small subsample of the test dataset (26 rows)** to use as an internal validation set. This allows us to evaluate the model on data that closely matches the competition format.

It is important to highlight that the test dataset is **not of very high quality**. After a detailed manual review, we found many cases where locations are **missing labels** or annotated inconsistently. This introduces noise and impacts the reliability of evaluation scores.

---

### 2.2 Exploratory Data Analysis (EDA)

#### Test & Validation Datasets

**Assumptions made for the validation dataset extend to the test dataset**, as both follow a very similar distribution.

- Around **half of the samples** do not contain location entities at all, and most of the rest contain **only one**.

<img width="720" src="https://github.com/user-attachments/assets/bb88eafc-e591-4d50-a773-b0dff647598b"/>

- Sentence length distribution in the validation & test dataset:

<img width="720" src="https://github.com/user-attachments/assets/fd0b1fdf-ef11-49b3-b29a-c78a3617b58c"/>

#### Train Dataset

<table>
  <tr>
    <td><img width="400" src="https://github.com/user-attachments/assets/23be7187-d5d0-4d83-a3ff-5354cf2e846d"/></td>
    <td><img width="400" src="https://github.com/user-attachments/assets/59e86652-b900-4e0a-9682-d1b4f13a888f"/></td>
  </tr>
</table>

### Preprocessing Steps
For training and validation datasets, we converted all not `LOC` labels into O

## 4. Modeling Approach

### 4.1 Baseline Model

**Tokenization**  
- Word-level tokenization.  
- Punctuation is split into separate tokens using a simple regex tokenizer.  
- This helps the model better capture boundaries and improves handling of short Telegram-style messages.

**Embeddings**  
- We use **FastText word embeddings** for Ukrainian/Russian.  
- All **unknown (OOV) tokens** are mapped to a **trainable embedding vector**, allowing the model to learn representations for rare or misspelled words.

**Model Architecture**  
- Encoder: **Bidirectional LSTM (BiLSTM)** to capture left and right contextual information.  
- Output Layer: **Softmax classifier** applied on top of each token representation.  
- Loss: Cross-entropy over token classes (LOC vs O).

**Reasoning**  
- Word-level FastText embeddings handle RU/UA morphology well due to subword information.  
- BiLSTM + softmax provides a simple, interpretable baseline that is easy to train and compares well to classical NER baselines.
## 4.2 Training Setup
**Loss Function**  
- We use `CrossEntropyLoss` with `ignore_index = -100` to ensure that padding tokens do not contribute to the loss.
**Optimizer**  
- Optimizer: **Adam**  
- Learning rate: **1e-3**
**Model Dimensions**  
- BiLSTM hidden size: **128**
**Training Procedure**  
- The model is trained for **15 epochs**.  
- Based on validation metrics, the **best performance was achieved at epoch 9**.  
- Training was stable, and no overfitting was observed before epoch 9.
**Kaggle Submission Performance**  
- Final F1-score on the public leaderboard: **0.50**  

### 4.3 LSTM + FastText + CRF Model
**Tokenization**
- Word-level tokenization with punctuation split as separate tokens.
- Stanza tokenizer used for Ukrainian text; emojis and noise removed.

**Embeddings**
- Pretrained FastText word vectors (300-dim).
- Unknown tokens replaced with a **trainable UNK embedding**.

**Architecture**
- **BiLSTM encoder**: hidden size 128, 2 layers, dropout 0.3.
- **Linear layer** to project LSTM outputs to label logits.
- **CRF decoding layer** to enforce valid label transitions (improves sequence consistency over softmax).

**Loss & Optimization**
- Loss: negative log-likelihood from CRF, with padding masked via `ignore_index = -100`.
- Optimizer: Adam, learning rate `1e-3`.

**Training**
- Trained for **10 epochs**, best validation performance at **epoch 6**.
**Performance**
- Kaggle public leaderboard F1-score: **0.50**.
- As the final submission should consist only of extracted locations, CRF improves the correctness of label orders, but it doesn`t really influence the final submission.

### other models
#### CRF + hidden size = 256 
kaggle - 0.53

### Model Comparison (Validation F1 vs Kaggle F1)

| Model                             | Validation F1 | Kaggle Public F1 |
|----------------------------------|----------------|------------------|
| **Baseline**                     | **0.50**       | **0.49**         |
| **LSTM (no CRF), hidden = 128**  | **0.52**       | **0.50**         |
| **LSTM + CRF, hidden = 256**     | **0.71**       | **0.53**         |



## Ensemble model




## 8. Conclusions



