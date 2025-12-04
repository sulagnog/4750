# Reddit Sentiment Analysis System

A comparative analysis of sentiment analysis tools for Reddit startup discussions.

## Overview

This project implements and compares multiple sentiment analysis approaches on Reddit comments from startup-related subreddits:

1. **VADER** - Lexicon-based sentiment analyzer optimized for social media
2. **TextBlob** - Simple lexicon-based sentiment analysis
3. **DistilBERT** - Transformer-based model fine-tuned on sentiment data
4. **Azure OpenAI GPT** - Large language model with prompt-based classification

## Project Structure

```
Comp-4750/
├── src/
│   ├── data_collection.py    # Reddit JSON scraper
│   ├── preprocessing.py      # Text cleaning and filtering
│   ├── sentiment_models.py   # All sentiment analysis models
│   ├── evaluation.py         # Metrics and evaluation
│   └── visualizations.py     # Charts and figures
├── data/
│   ├── raw_comments.csv      # Scraped Reddit data
│   ├── processed_comments.csv# Cleaned data
│   ├── labeled_comments.csv  # Gold-labeled data
│   └── results.csv           # All model predictions
├── outputs/
│   ├── evaluation_results.json
│   ├── metrics_tables.md     # Paper-ready tables
│   └── *.png                 # Visualizations
├── main.py                   # Main pipeline script
├── requirements.txt          # Dependencies
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('vader_lexicon')
```

### 3. Configure Azure OpenAI (Optional)

Create a `.env` file with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Usage

### Run Full Pipeline

```bash
# Full pipeline with auto-labeling (uses GPT for gold labels)
python main.py --auto-label

# Full pipeline without GPT (faster, no API costs)
python main.py --skip-gpt --auto-label

# With custom sample size
python main.py --auto-label --sample 100
```

### Run Individual Components

```bash
# Data collection only
python -m src.data_collection

# Preprocessing only
python -m src.preprocessing

# Test sentiment models
python -m src.sentiment_models

# Test evaluation
python -m src.evaluation
```

## Output Files

After running the pipeline, you'll have:

- **`outputs/metrics_tables.md`** - Copy-paste ready tables for your paper
- **`outputs/all_metrics_comparison.png`** - Model comparison chart
- **`outputs/confusion_matrix_*.png`** - Confusion matrices
- **`outputs/error_analysis.csv`** - Examples where models disagree

## Research Paper Sections

This system generates data for the following paper sections:

### Methodology
- Data collection: Reddit JSON API scraping
- Preprocessing: URL removal, markdown cleaning, length filtering
- Models: VADER, TextBlob, DistilBERT, GPT

### Results
- Overall accuracy, precision, recall, F1 per model
- Per-class metrics (Positive/Negative/Neutral)
- Confusion matrices
- Inter-model agreement statistics

### Discussion
- Error analysis with specific examples
- Model strengths/weaknesses on Reddit language

## Target Subreddits

- r/startups
- r/Startup_Ideas  
- r/Entrepreneur
- r/smallbusiness
- r/SaaS

## Citation

If using this code for academic work, please cite the relevant tools:
- VADER: Hutto & Gilbert (2014)
- TextBlob: Loria (2018)
- DistilBERT: Sanh et al. (2019)
- Transformers: Wolf et al. (2020)

# 4750
