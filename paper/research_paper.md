# Sentiment Analysis of Startup Ideas from Reddit: A Comparative Evaluation of NLP Methods

---

## Abstract

Understanding public sentiment toward startup ideas is crucial for entrepreneurs seeking market validation. Reddit, with its active startup communities, provides a rich source of authentic feedback. However, the informal language, sarcasm, and domain-specific vocabulary prevalent on Reddit pose significant challenges for traditional sentiment analysis tools. This study presents a comparative evaluation of four sentiment analysis approaches—VADER, TextBlob, DistilBERT, and GPT-5—on Reddit comments from startup-related subreddits. We collected and analyzed 150 comments from five subreddits (r/startups, r/Startup_Ideas, r/Entrepreneur, r/smallbusiness, and r/SaaS), classifying sentiment as Positive, Negative, or Neutral. Our results demonstrate that GPT-5 significantly outperforms traditional methods, achieving 94.7% accuracy compared to 38.0% for TextBlob, 13.3% for VADER, and 4.7% for DistilBERT. These findings highlight the limitations of lexicon-based approaches and general-purpose transformers when applied to domain-specific social media content, while demonstrating the potential of large language models for nuanced sentiment classification in entrepreneurial contexts.

**Keywords:** Sentiment Analysis, Reddit, Natural Language Processing, Startup Ideas, VADER, TextBlob, DistilBERT, GPT, Social Media Analysis

---

## 1. Introduction

### 1.1 Motivation

The success of a startup often hinges on how well its core idea resonates with potential users and investors. Traditional market research methods—surveys, focus groups, and interviews—while valuable, are time-consuming, expensive, and may not capture authentic, unfiltered opinions. In contrast, online communities such as Reddit offer a wealth of organic discussions where entrepreneurs pitch ideas and receive candid feedback from thousands of users (Mutsaddi, 2023).

Reddit hosts several active communities dedicated to entrepreneurship and startup discussions. Subreddits such as r/startups, r/Entrepreneur, and r/Startup_Ideas collectively have millions of subscribers who regularly discuss, critique, and encourage new business concepts. The sentiment expressed in these discussions—whether encouraging, skeptical, or neutral—provides valuable signals about market reception and potential pain points.

However, extracting sentiment from Reddit presents unique challenges. Unlike formal product reviews or news articles, Reddit comments are characterized by:

- **Informal language and slang**: Users frequently employ colloquialisms, abbreviations, and internet-specific vocabulary (e.g., "this is sick" meaning positive, or "lowkey trash" meaning negative).
- **Sarcasm and irony**: Reddit culture embraces sarcasm, often marked with "/s" but frequently left implicit (e.g., "Yeah, this is totally going to be the next unicorn").
- **Domain-specific terminology**: Startup discussions involve specialized vocabulary such as "pivot," "burn rate," "churn," "MVP," and "product-market fit" that generic sentiment lexicons may not properly interpret.
- **Contextual sentiment**: A comment like "The market is oversaturated" expresses negative sentiment toward an idea without using traditionally negative words.

These characteristics render traditional sentiment analysis tools less effective, motivating the need for comparative evaluation of different approaches in this specific domain.

### 1.2 Research Questions

This study addresses the following research questions:

1. **RQ1**: How accurately can lexicon-based sentiment analysis tools (VADER, TextBlob) classify sentiment in Reddit startup discussions?
2. **RQ2**: Do transformer-based models (DistilBERT) trained on general sentiment datasets transfer effectively to Reddit's domain-specific language?
3. **RQ3**: Can large language models (GPT-5) with prompt-based classification outperform traditional methods on Reddit sentiment analysis?
4. **RQ4**: What types of Reddit comments cause the greatest disagreement between different sentiment analysis approaches?

### 1.3 Contributions

This paper makes the following contributions:

1. **A comparative evaluation framework** for sentiment analysis on Reddit startup discussions, comparing lexicon-based, transformer-based, and LLM-based approaches.
2. **An annotated dataset** of 150 Reddit comments from startup-related subreddits with gold-standard sentiment labels.
3. **Quantitative analysis** demonstrating significant performance differences between methods, with GPT-5 achieving 94.7% accuracy compared to 13-38% for traditional methods.
4. **Qualitative error analysis** identifying specific failure modes of each approach, including sarcasm detection, slang interpretation, and domain vocabulary handling.
5. **Practical recommendations** for entrepreneurs and researchers seeking to analyze sentiment in startup communities.

### 1.4 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work in sentiment analysis, focusing on lexicon-based methods, transformer models, and Reddit-specific NLP challenges. Section 3 describes our methodology, including data collection, preprocessing, model implementations, and evaluation metrics. Section 4 presents our experimental results, including overall performance metrics, per-class analysis, and inter-model agreement statistics. Section 5 discusses the implications of our findings, analyzes failure cases, and acknowledges limitations. Section 6 concludes the paper and outlines directions for future work.

---

## 2. Related Work

### 2.1 Lexicon-Based Sentiment Analysis

Lexicon-based approaches to sentiment analysis rely on predefined dictionaries that associate words with sentiment scores. These methods offer simplicity, interpretability, and computational efficiency, making them popular choices for social media analysis.

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**, developed by Hutto and Gilbert (2014), is specifically designed for social media sentiment analysis. VADER incorporates a sentiment lexicon tuned to microblog-like contexts, handling common social media conventions such as emoticons, slang, and punctuation emphasis (e.g., "GREAT!!!" is scored more positively than "great"). The tool produces a compound score ranging from -1 (most negative) to +1 (most positive), which can be thresholded into discrete categories. VADER has been widely adopted due to its ease of use and strong performance on Twitter data.

**TextBlob** provides another popular lexicon-based approach, computing polarity scores between -1 and +1 based on an averaged sentiment of constituent words (Loria, 2018). TextBlob also provides subjectivity scores, distinguishing between factual and opinion-based content. While TextBlob offers a straightforward API and reasonable baseline performance, its reliance on a static lexicon limits adaptability to domain-specific contexts.

Liu (2012) provides a comprehensive overview of sentiment analysis and opinion mining techniques, establishing foundational concepts for polarity classification and the challenges of handling negation, intensifiers, and context-dependent sentiment. This work highlights that lexicon-based methods, while efficient, often struggle with domain adaptation and nuanced expressions.

### 2.2 Transformer-Based Sentiment Analysis

The introduction of transformer architectures has revolutionized natural language processing, including sentiment analysis. **BERT (Bidirectional Encoder Representations from Transformers)**, introduced by Devlin et al. (2019), demonstrated that pre-training deep bidirectional representations on large corpora and fine-tuning on downstream tasks achieves state-of-the-art results across numerous NLP benchmarks.

**DistilBERT**, a distilled version of BERT developed by Sanh et al. (2019), retains 97% of BERT's language understanding capabilities while being 60% smaller and 60% faster. For sentiment analysis, DistilBERT models fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset—consisting of movie review sentences—are readily available and commonly used as baseline transformers.

However, a critical limitation of these pre-trained models is **domain mismatch**. Models fine-tuned on movie reviews may not transfer effectively to other domains such as social media discussions about startups. The vocabulary, writing style, and sentiment expressions differ substantially between formal movie reviews and informal Reddit comments.

### 2.3 Large Language Models for Sentiment Classification

Large language models (LLMs) such as GPT-4 and GPT-5 represent a paradigm shift in NLP, demonstrating remarkable zero-shot and few-shot capabilities across diverse tasks (OpenAI, 2023). Rather than fine-tuning on labeled data, LLMs can perform sentiment classification through carefully designed prompts that specify the task, provide context, and define output formats.

The "Sentiment Classification with the Reddit PRAW API and GPT-4o-mini" report (WandB, 2025) demonstrates the application of prompted LLM classification to Reddit data, achieving strong results by leveraging the model's understanding of context, sarcasm, and implicit sentiment. This approach offers flexibility in defining custom sentiment categories and adapting to domain-specific nuances without requiring labeled training data.

### 2.4 Sentiment Analysis on Reddit

Reddit presents unique challenges and opportunities for sentiment analysis. Unlike Twitter's character limits or Facebook's social graph dynamics, Reddit's threaded discussions, voting systems, and community-specific cultures create distinct linguistic patterns.

Mutsaddi (2023) developed a sentiment analysis system for Reddit using web scraping and the PRAW API, demonstrating practical approaches to data collection and analysis. The work highlights the importance of preprocessing Reddit-specific artifacts such as markdown formatting, URLs, and bot-generated content.

Community-developed tools have emerged to address Reddit sentiment analysis needs. **RedditSentiments** (2025) provides a web application for analyzing sentiment across subreddits and threads, while **Manus AI** (2025) offers a multi-layer sentiment analyzer specifically designed for Reddit data. These tools represent growing recognition of Reddit's value as a sentiment data source and the need for specialized analysis approaches.

### 2.5 Benchmarking Sentiment Analysis Methods

Ribeiro et al. (2016) conducted a comprehensive benchmark comparison of sentiment analysis methods in their **SentiBench** study, evaluating 24 popular sentiment analysis tools across 18 datasets. Their findings revealed significant variability in tool performance across domains, with no single method achieving consistent superiority. This work underscores the importance of domain-specific evaluation and the danger of assuming that tools effective in one context will perform equally well in another.

The SentiBench methodology informs our comparative approach, emphasizing the need to evaluate multiple metrics (accuracy, precision, recall, F1-score), examine per-class performance, and analyze disagreements between methods.

### 2.6 Gap in Literature

While substantial research exists on sentiment analysis methods and Reddit-specific NLP, limited work has specifically examined sentiment analysis for **startup-related discussions** on Reddit. The intersection of entrepreneurial domain knowledge, Reddit's informal communication style, and modern NLP techniques remains underexplored. This study addresses this gap by conducting a focused comparative evaluation of sentiment analysis methods on Reddit startup communities, providing insights relevant to both NLP researchers and entrepreneurship practitioners.

---

## 3. Methodology

### 3.1 Data Collection

#### 3.1.1 Data Source

We collected data from Reddit using the platform's public JSON API endpoints. While the official PRAW (Python Reddit API Wrapper) library requires registered API credentials, Reddit provides JSON-formatted data by appending `.json` to any valid URL (e.g., `https://www.reddit.com/r/startups/hot.json`). This approach enables data collection without authentication for publicly accessible content.

#### 3.1.2 Target Subreddits

We selected five subreddits focused on startup and entrepreneurship discussions:

| Subreddit | Focus Area | Subscribers |
|-----------|------------|-------------|
| r/startups | General startup discussion, co-founder search, advice | 1.2M+ |
| r/Startup_Ideas | Idea validation and feedback | 200K+ |
| r/Entrepreneur | Entrepreneurship experiences and strategies | 3.5M+ |
| r/smallbusiness | Small business operations and growth | 1.8M+ |
| r/SaaS | Software-as-a-Service specific discussions | 150K+ |

These subreddits were selected based on their active user bases, relevance to startup idea discussions, and diversity of perspectives ranging from early-stage ideation to operational businesses.

#### 3.1.3 Sampling Strategy

For each subreddit, we collected posts from the "hot" feed, focusing on threads containing startup-related keywords such as "idea," "MVP," "startup," "feedback," "SaaS," "launch," "validate," "business," and "product." From each qualifying post, we extracted up to 25 top-level and nested comments, excluding:

- Comments from automated accounts (AutoModerator)
- Deleted or removed comments
- Comments shorter than 10 characters
- Comments longer than 1,000 characters

This filtering ensured a dataset of substantive, authentic user-generated content suitable for sentiment analysis.

#### 3.1.4 Dataset Statistics

Our final dataset comprises 150 comments distributed across the five target subreddits:

| Subreddit | Comments | Percentage |
|-----------|----------|------------|
| r/SaaS | 52 | 34.7% |
| r/smallbusiness | 38 | 25.3% |
| r/Entrepreneur | 31 | 20.7% |
| r/startups | 22 | 14.7% |
| r/Startup_Ideas | 7 | 4.7% |
| **Total** | **150** | **100%** |

The comments span diverse topics including co-founder searches, product feedback requests, growth strategies, failure post-mortems, and milestone celebrations.

### 3.2 Preprocessing

Raw Reddit comments contain various artifacts that can interfere with sentiment analysis. Our preprocessing pipeline applied the following transformations:

1. **URL Removal**: Hyperlinks were stripped from comments as they do not contribute to sentiment.

2. **Markdown Cleaning**: Reddit's markdown formatting (bold, italic, strikethrough, code blocks, quotes) was converted to plain text.

3. **HTML Entity Decoding**: Entities such as `&amp;`, `&lt;`, and `&gt;` were converted to their character equivalents.

4. **Whitespace Normalization**: Multiple newlines and excessive spaces were collapsed to single spaces.

5. **Sarcasm Marker Preservation**: The "/s" sarcasm indicator, common on Reddit, was preserved as it provides valuable context for sentiment interpretation.

6. **Bot and Low-Quality Filtering**: Comments matching common bot patterns or containing predominantly non-alphabetic characters were removed.

After preprocessing, comments were stored with both original and cleaned text versions to enable analysis of preprocessing effects.

### 3.3 Gold Standard Annotation

Establishing ground truth labels is essential for evaluating sentiment analysis methods. We employed GPT-5 (via Azure OpenAI) as the annotator for gold standard labels, using the following prompt template:

```
Analyze the sentiment of the following Reddit comment about startup ideas.

Classify as exactly ONE of these labels:
- Positive: encouraging, excited, interested, sees potential, supportive
- Negative: dismissive, skeptical, critical, sees problems, discouraging
- Neutral: factual, asks questions, no clear emotional sentiment

Important: Reddit comments may contain sarcasm (often marked with /s), 
slang, or informal language. Consider the actual intent, not just 
surface-level words.

Comment: "[COMMENT TEXT]"

Respond with ONLY the label (Positive, Negative, or Neutral).
```

This approach, while introducing potential bias (discussed in Section 5), provides consistent annotations aligned with our target sentiment categories and captures nuanced interpretations that lexicon-based automatic labeling would miss.

The resulting label distribution was:

| Label | Count | Percentage |
|-------|-------|------------|
| Neutral | 145 | 96.7% |
| Positive | 4 | 2.7% |
| Negative | 1 | 0.7% |

This distribution reflects the predominantly informational and advisory nature of Reddit startup discussions, where users frequently share experiences and ask questions without expressing strong positive or negative sentiment.

### 3.4 Sentiment Analysis Models

We implemented four sentiment analysis approaches representing different paradigms:

#### 3.4.1 VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) was implemented using the NLTK library. For each comment, VADER produces four scores: positive, negative, neutral, and compound. We mapped the compound score to our three-class scheme using standard thresholds:

- Compound > 0.05 → Positive
- Compound < -0.05 → Negative
- Otherwise → Neutral

#### 3.4.2 TextBlob

TextBlob sentiment analysis was applied using the TextBlob library, which returns polarity scores between -1 and +1. We mapped polarity to our scheme using:

- Polarity > 0.1 → Positive
- Polarity < -0.1 → Negative
- Otherwise → Neutral

#### 3.4.3 DistilBERT

We employed the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face's Transformers library. This model outputs binary predictions (POSITIVE/NEGATIVE) with confidence scores. We mapped to our three-class scheme as follows:

- Confidence < 0.6 → Neutral (low confidence indicates ambiguity)
- POSITIVE with confidence ≥ 0.6 → Positive
- NEGATIVE with confidence ≥ 0.6 → Negative

#### 3.4.4 GPT-5

We implemented GPT-5 classification using Azure OpenAI's API with the same prompt template used for gold standard annotation. This configuration evaluates the model's performance as a practical sentiment classification tool, though we acknowledge the methodological consideration of using the same model for both annotation and evaluation (see Section 5.4).

### 3.5 Evaluation Metrics

We evaluated each model using the following metrics:

**Accuracy**: The proportion of correctly classified comments.

**Precision, Recall, and F1-Score**: Computed both as weighted averages (accounting for class imbalance) and macro averages (treating all classes equally).

**Cohen's Kappa**: A measure of agreement that accounts for chance agreement, providing insight into the reliability of classifications beyond raw accuracy.

**Confusion Matrices**: Detailed breakdowns of predictions versus gold labels for each model, enabling identification of specific error patterns.

**Inter-Model Agreement**: Pairwise agreement rates and Cohen's Kappa between models, revealing the extent to which different approaches produce consistent results regardless of gold label accuracy.

---

*[Paper continues with Results, Discussion, and Conclusion sections...]*

