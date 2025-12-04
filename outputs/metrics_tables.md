# Evaluation Results

## Overall Metrics

| Model | Accuracy | Precision | Recall | F1 (Weighted) | F1 (Macro) | Kappa |
|-------|----------|-----------|--------|---------------|------------|-------|
| VADER | 0.133 | 0.968 | 0.133 | 0.194 | 0.092 | 0.013 |
| TextBlob | 0.380 | 0.968 | 0.380 | 0.520 | 0.208 | 0.035 |
| DistilBERT | 0.047 | 0.969 | 0.047 | 0.030 | 0.064 | 0.021 |
| GPT | 0.947 | 0.955 | 0.947 | 0.950 | 0.620 | 0.312 |


## Per-Class Metrics

| Model | Class | Precision | Recall | F1 | Support |
|-------|-------|-----------|--------|----|---------|
| VADER | Positive | 0.041 | 1.000 | 0.078 | 4 |
| VADER | Negative | 0.000 | 0.000 | 0.000 | 1 |
| VADER | Neutral | 1.000 | 0.110 | 0.199 | 145 |
| TextBlob | Positive | 0.047 | 1.000 | 0.090 | 4 |
| TextBlob | Negative | 0.000 | 0.000 | 0.000 | 1 |
| TextBlob | Neutral | 1.000 | 0.366 | 0.535 | 145 |
| DistilBERT | Positive | 0.078 | 1.000 | 0.145 | 4 |
| DistilBERT | Negative | 0.010 | 1.000 | 0.020 | 1 |
| DistilBERT | Neutral | 1.000 | 0.014 | 0.027 | 145 |
| GPT | Positive | 0.200 | 0.250 | 0.222 | 4 |
| GPT | Negative | 0.500 | 1.000 | 0.667 | 1 |
| GPT | Neutral | 0.979 | 0.966 | 0.972 | 145 |


## Model Agreement

| Model Pair | Agreement Rate | Cohen's Kappa |
|------------|----------------|---------------|
| VADER vs TextBlob | 56.0% | 0.232 |
| VADER vs DistilBERT | 52.0% | 0.227 |
| VADER vs GPT | 14.7% | 0.023 |
| TextBlob vs DistilBERT | 30.0% | 0.068 |
| TextBlob vs GPT | 33.3% | -0.036 |
| DistilBERT vs GPT | 4.7% | 0.014 |


## Confusion Matrices

Confusion Matrix - VADER:
                  Predicted
              Pos    Neg    Neu
Actual ------------------------------
  Pos  |     4     0     0
  Neg  |     1     0     0
  Neu  |    93    36    16

Confusion Matrix - TextBlob:
                  Predicted
              Pos    Neg    Neu
Actual ------------------------------
  Pos  |     4     0     0
  Neg  |     1     0     0
  Neu  |    80    12    53

Confusion Matrix - DistilBERT:
                  Predicted
              Pos    Neg    Neu
Actual ------------------------------
  Pos  |     4     0     0
  Neg  |     0     1     0
  Neu  |    47    96     2

Confusion Matrix - GPT:
                  Predicted
              Pos    Neg    Neu
Actual ------------------------------
  Pos  |     1     0     3
  Neg  |     0     1     0
  Neu  |     4     1   140
