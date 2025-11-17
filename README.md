# Tokenization & Punctuation Strategy for UA/RU NER  
### (FastText Only • Trainable Embedding Only • Combined Embeddings)

This document describes how to preprocess Ukrainian/Russian text for Named Entity Recognition (NER) in three embedding scenarios:

1. **FastText embeddings only**  
2. **Trainable embeddings only**  
3. **Combined FastText + Trainable embeddings**  

UA/RU morphology, punctuation, and mixed-code text require careful handling.  
Follow the steps below for consistent and high-performing NER models.

---

# Scenario 1 — Using FastText Embeddings Only

FastText produces **strong vectors for real words** but **meaningless vectors for punctuation**.  
To use FastText correctly:

---

## ✔ Step 1: Strip punctuation **inside words**

Examples:

- `Києві,` → `Києві`
- `Львів!!!` → `Львів`
- `(Київ)` → `Київ`
- `м.Астана` → `Астана` *(or split further if needed)*

```python
import re

PUNCT_STRIP = re.compile(r"^[^\wА-Яа-яІіЇїЄєҐґ]+|[^\wА-Яа-яІіЇїЄєҐґ]+$", re.UNICODE)

def clean_word(tok: str):
    tok = PUNCT_STRIP.sub("", tok)
    return tok if tok else None
```

---

## ✔ Step 2: Keep punctuation tokens in the sequence  
Do **not** remove tokens like `,`, `!`, `:`, `?` — they help NER detect boundaries.

---

## ✔ Step 3: Map punctuation tokens to a **trainable `ft_unk` vector**  
FastText embeddings for punctuation are garbage, so override them:

```python
def get_fasttext_vec(token, ft, ft_unk):
    cleaned = clean_word(token)
    if cleaned:
        return torch.tensor(ft.get_word_vector(cleaned))
    else:
        return ft_unk  # trainable vector for all punctuation / unknowns
```

---

## ✔ Summary (FastText only)

- Words → cleaned → real FastText vectors  
- Punctuation → always `ft_unk`  
- Keep punctuation tokens  
- Do NOT include punctuation in vocab  

---

# 🟩 Scenario 2 — Using Trainable Embeddings Only

Here you rely on:

```python
nn.Embedding(vocab_size, embed_dim)
```

No external pretrained embeddings.

---

## ✔ Step 1: Strip punctuation inside words

Same as in Scenario 1.

---

## ✔ Step 2: Do NOT put punctuation tokens into your vocabulary  
Vocab should include:

- real words  
- digits  
- `<pad>`  
- `<unk>`

Punctuation should NOT be assigned separate embeddings.

---

## ✔ Step 3: Map all punctuation tokens → `<unk>`  
Implementation:

```python
if token in vocab:
    token_id = vocab[token]
else:
    token_id = vocab["<unk>"]  # all punctuation lands here
```

---

## ✔ Summary (Trainable only)

- Words → cleaned → vocab lookup  
- Punctuation → `<unk>`  
- Keep punctuation in sequence  
- Vocab stays clean and compact  

---

# 🟧 Scenario 3 — Combined FastText + Trainable Embeddings (Recommended)

This is the **best classical NER architecture**:

```
embedding = concat(trainable_word_embedding, fasttext_embedding)
```

Each token becomes:

```
[trainable_emb_dim] + [ft_dim]
```

---

## ✔ Step 1: Strip punctuation inside words

Identical to scenarios 1 and 2.

---

## ✔ Step 2: Trainable embedding side  
- Vocabulary **excludes punctuation**  
- Punctuation → `<unk>`  

---

## ✔ Step 3: FastText embedding side  
- Words → cleaned → FT vector  
- Punctuation → `ft_unk`  
- Empty tokens → `ft_unk`  

---

## ✔ Step 4: Concatenate embeddings

```python
combined_vec = torch.cat([emb_trainable, emb_fasttext], dim=-1)
```
