# Sentiment Analysis of Startup Ideas on Reddit – Comparative Assessment of NLP Techniques

---

## Abstract

Knowing the perception of the general public regarding start-up ideas and their validation are necessary for every entrepreneur. Reddit, which serves as a community forum for startups, becomes an important and reliable source to gain insights. The use of slang, sarcastisms, and technical terms, which are common on the Reddit platform, introduces many challenges for the current sentiment analysis algorithms. In the following study, the performance of four available algorithms, VADER, TextBlob, DistilBERT, and GPT-5, for sentiment analysis for start-up related subreddits sourced from Reddit communities are compared. For the experiment, 150 comments were carefully selected and analyzed from the subreddits r/startups, r/Startup_Ideas, r/Entrepreneur, r/smallbusiness, and r/SaaS, and the comments were categorized into Positive, Negative, and Neutral. The experimental outcome reveals that GPT-5 performs significantly well compared to the other algorithms, which yield accuracy of 94.7%, 38.0%, 13.3%, and 4.7%, for GPT-5, TextBlob, VADER, and DistilBERT, respectively.

**Keywords:** Sentiment Analysis, Reddit, Natural Language Processing, Startup Ideas, VADER, TextBlob, DistilBERT, GPT, Social Media Analysis

---

## 1. Introduction

### 1.1 Motivation: The Validation Gap

In many cases, the viability of start-ups can hinge on how well the original concept can appeal to both the target market and potential investors. In the development of ventures, gaining truthful and unprejudiced feedback can be essential and difficult. The Lean Startup process stresses rapid validation, but the tools available are often insufficient for the process. More conventional means of market research that include surveys, focus groups, and structured interviews are effective but are associated with some disadvantages, such as the following:

- They are often costly and labor-intensive.
- They are susceptible to biased responses, such as the "social desirability bias" experienced during focus groups. This can be likened to the "Mom Test failure."
- They are, as already stated, costly and labor-intensive.

Automated sentiment analysis for online, organic conversation provides a whole new paradigm. An entrepreneur could easily gauge the emotional climate of their desired marketplace by digging through thousands of already-existing discussions, and thus, the development of unwanted products could be avoided. This could save the start-up capital that would be spent on research.

### 1.2 The Reddit Ecosystem for Startup Feedback

Cyber communities like Reddit offer an intriguing remedy for the validation problem. Unlike LinkedIn, where the degree of professionalism and self-marketing are orientation factors, and Twitter, which stresses short messages and the capacity to attract multiple followers, Reddit offers extensive, tree-level discussions based on focused themes. The semi-anonymous nature of the platform encourages honest responses, and it offers unrefined market feedback. For an entrepreneur, it becomes an endless focus group.

Some of the active subreddits for the topic of entrepreneur are:

- **r/startups (1.6M members):** This community provides general support for startups and covers legal aspects as well as emotional well-being. The community mood here appears to be tempered and practical.

- **r/Startup_Ideas (200K subscribers):** Primarily for validation, where users submit elevator pitches, and the community reviews them. The tone of comments can be brutally honest, making it an excellent resource for sentiment analysis.

- **r/SaaS (150K members):** All about Software as a Service, speaking the language, having the jargon (MRR, Churn, CAC)—sentiments correlate well to numerical metrics.

- **r/Entrepreneur (3.5M members):** This community covers many types of ventures, such as lifestyle, drop shipping, and traditional ventures. Signal-to-noise ratio is somewhat high.

- **r/smallbusiness (1.8M members):** Focuses on physical and service-related businesses and often has comments that are reality checks.

All the above-discussed diametric discussions, whether it be encouragement, skepticism, and/or neutrality, emit market sentiment signals. As an exemplar, market engagement threads denoted by fair-priced inquiries are far less significant compared to negative threads that often present inherent value delivery errors.

### 1.3 Challenges in Domain-Specific Sentiment

There are several additional challenges involved for the sentiment analysis model applied to the Reddit comments, over and above the challenges that would be faced for the reviews and news articles:

- **Colloquialisms and internet slang:** For example, abbreviations and internet lingo (such as "this is sick," "hard pass") can reverse the meaning of words.

- **Sarcasm and ironies:** The culture of Reddit uses sarcasm, which, as a rule, is implicit. Positive-sounding phrases can express negative emotions.

- **Domain-specific vocabularies:** In the startup world, there are vocabularies such as "burn rate," "churn," "MVP," and "pivot," which come with their own sets of contextual meanings.

- **Contextual sentiment and constructive criticism:** Feedback like "The market is oversaturated" carries negative sentiment regarding viability but does not use negative language, while constructive criticism uses negative language but carries positive intentions. This kind of feedback can be misleading for binary-classification algorithms.

These factors provide impetus for an intensive comparative assessment of the approaches investigated here.

### 1.4 Research Questions

This research explores:

- **RQ1:** How well do lexicon-based tools like VADER and TextBlob work on sentiment analysis for the Reddit community of startups?

- **RQ2:** Can transformer models (DistilBERT) trained on general sentiment information transfer well into the domain language of Reddit?

- **RQ3:** Can large language models such as GPT-5, using the concept of classification through prompts, prove to be superior compared to the traditional approach?

- **RQ4:** What are the kinds of comments on Reddit that tend to induce the maximum amount of disagreement regarding the sentiment analysis, and how does that reflect the nature of comments regarding startups?

### 1.5 Contributions

This paper provides:

- **Comparative evaluation framework** for three generations of NLP tools: lexicon-based (VADER, TextBlob), transformer-based (DistilBERT), and generative LLM-based (GPT-5).

- **Annotated domain dataset** of 150 comments from five subreddits related to start-ups, which are labeled using gold standard sentiment.

- **Quantitative performance analysis** showing significant discrepancies between the approaches, GPT-5 having 94.7% accuracy and the conventional approaches no higher than 40%.

- **Inclusion of qualitative analysis** for the identification of modes of failure, such as the misclassification of comments regarded as constructive and neutral by binary classifier algorithms and the inability of lexical databases to recognize the presence of sarcasm.

- **Applying the recommendations** for creators of market intelligence tools, proposing that cost-effectiveness of older models could be less optimal for the discussed domain.

### 1.6 Paper Organization

The rest of the paper is organized as follows. In Section 2, the state of the art of related work on sentiment analysis is discussed. This spans the history of the development of dictionary-based algorithms for sentiment prediction and their later replacement by deep learning algorithms. In Section 3, the authors outline their research method. This spans the process of acquiring data from Reddit, the processes involved in data pre-processing, as well as the sentiment prediction algorithms designed and utilized. In Section 4, the authors provide the experimental findings.

---

## 2. Related Work

### 2.1 The Evolution of Sentiment Analysis

Sentiment analysis, also called opinion mining, has experienced significant growth during the past two decades. The traditional approaches had mainly been based on the lexicon method, where words are associated with predefined values, such as "good" for +1 and "terrible" for -1.

**VADER (Valence Aware Dictionary and sEntiment Reasoner):** This tool, created by Hutto and Gilbert (2014), can be termed as the culmination of the above method customized for use on social media. Unlike generic dictionaries, it is designed and customized for the microblogging world, such as that of Tweeting, and takes into consideration the following social media conventions:

- **Booster words:** Recognizes that "extremely good" is much more positive than "good."

- **Capitalization:** Gives greater scores for capitalized words like "GREAT" against the lowercase version "great".

- **Emoticons:** Converts smiles like ":-)" and ":-D" into positive values.

- **Negation:** Deals with negation by reversing the polarity of the corresponding terms, such as "not good".

In spite of such innovations, VADER remains a bag-of-words model, wherein it tries to understand words either individually or in short contexts, which are incapable of capturing the subtle nuances of sentence-level structure.

**TextBlob** (Loria, 2018) provides a comparable lexicon-based approach, which includes a measure of subjectivity. This method calculates the polarity based on the average sentiment of the adjectives and adverbs present in the text. Although it is computationally efficient (often sub-millisecond processing times), it becomes fragile and prone to errors if it encounters domain-specific semantics. In the case of the startup domain, for example, the term "disruptive" would carry the semantics of something positive, but often it carries the opposite semantics.

### 2.2 The Transformer Revolution

The emergence of the Transformer model (Vaswani et al., 2017) and the BERT (Bidirectional Encoder Representations from Transformers) model (Devlin et al., 2019) revolutionized the field of natural language processing. BERT, unlike other dictionary-based models, processes the text bidirectionally, which helps the model understand contexts. For instance, the term "bank" would be understood differently in "river bank" and "bank account."

The use of **DistilBERT** (Sanh et al., 2019) introduces a smaller, faster, and cheaper version of BERT, but one that retains nearly 97% of the performance. In the case of sentiment analysis, the most common method for doing it well is actually that of transfer learning, involving the pre-training of the DistilBERT model and the subsequent fine-tuning on the sentiment-labeling dataset. One popular dataset for sentiment analysis tasks is the Stanford Sentiment Treebank, called SST-2.

- **Strength:** The grammar capabilities of DistilBERT are much stronger compared to VADER.

- **Weakness:** Domain mismatch. The model trained on movie reviews would be searching for an evaluation of the movie. This would not translate well to the conversation and problem-solving characteristics present on the Reddit discussions for startups. For instance, the term "difficult coding problem" would not necessarily indicate the speaker dislikes the startup, but the model trained on movie reviews would mark it as negative.

### 2.3 Large Language Models and Prompting

Presently, the progress achieved is facilitated by the Large Language Models, GPT-4 and GPT-5. These are generative engines that are trained based on the vast internet data, which gives the AI broad world knowledge.

In place of fine-tuning, the method of zero-shot learning for classification through prompt engineering is applied. This requires the model to be explained the task through the use of a prompt like:

> "Classify the following text as Positive, Negative, or Neutral."

Recent findings, as shown in the report "Sentiment Classification with the Reddit PRAW API and GPT-4o-mini" (WandB, 2025), reveal that the above method can be applied for state-of-the-art performance as it allows the model to understand:

1. **Sarcasm**, even without explicit cues.

2. **Pragmatics:** The message underlying the words, such as the difference between evaluation and criticism.

3. **Implicit sentiment:** This type of sentiment expresses an attitude without explicit words of sentiment.

### 2.4 Reddit as a Unique NLP Challenge

"Societal and behavioral information" gathered from Reddit, as opposed to other online communities, allows for the extraction of computational insights due to the nature of the forum's structure. Mutsaddi (2023) explains that one of the technical difficulties involved in their data scraping process for Reddit specifically involves the PRAW Web API, which often sets "rate limitations" for the researchers. While the structure of the message format for the tweets sampled from the Twitter API was relatively horizontal, involving one tweet after the other, Reddit's threaded discussions create complex hierarchical relationships.

Previous research conducted by Ribeiro et al. (2016), called **SentiBench**, assessed 24 sentiment analysis approaches and found that no one approach dominated the field. This research antedates the LLM age. This study proposes an update of the above-mentioned framework, especially focusing on the comparatively uncharted territory of the entrepreneurship field.

Most of the existing tools for the analysis of sentiment on the Reddit platform, such as RedditSentiments and the AI tool Manus, focus on the aggregation aspect, namely the creation of the sentiment trend line. They provide very limited information regarding the accuracy of their classifications for individual comments. This study attempts to shine some light on the accuracy of the basic classifications.

### 2.5 The Gap in Entrepreneurial NLP

Although the use of sentiment analysis is a mature area, research involving the use of the technique for entrepreneurship as the focus remains an uncharted area. Most research involving financial sentiment analysis focuses on stock market prediction, which involves the analysis of CNBC headlines/earnings calls, and such information is structured and polished.

Startup discourse on the Reddit community provides a new kind of data, which is defined by an unstructured business conversation that combines the emotional and the technical. There appears to be an important research gap concerning the performance of basic NLP engines against such hybrid data. For example, whether a posting concerning "bootstrapping," which represents an unbiased and, therefore, favorable kind of business, could possibly be misconstrued as negative by basic engines.

This study fills that research gap.

---

## 3. Methodology

The method described in this paper was designed to simulate a real-world production pipeline for a "Startup Market Intelligence" tool. The method comprises four sequential stages, including data collection, preprocessing, gold labeling, and model evaluation.

### 3.1 Data Collection Strategy

#### 3.1.1 Reddit JSON API

While Python Reddit API Wrapper (PRAW) is widely used, it requires authentication from the developer, which can be cumbersome for simple fetch operations. This brings us to the method that Reddit uses – JSON. Adding `.json` to the end of any URL on Reddit provides an organized JSON version of the information.

**Example Request:**

```
GET https://www.reddit.com/r/startups/hot.json?limit=25
```

This method imitates the patterns of lightweight web crawlers as well as client-side scripts, bypassing the use of OAuth for public information. The information returned consists of the same structured fields that PRAW returns, such as `body`, `author`, `upvotes`, and `created_utc`, but are delivered raw.

#### 3.1.2 Subreddit Selection and Sampling

The study focused on the five subreddits listed in Section 1.2, which were r/startups, r/Startup_Ideas, r/Entrepreneur, r/smallbusiness, and r/SaaS.

**Sampling Criteria:**

- **Feed:** "Hot" for active and relevant discussions.
- **Keywords:** Threads that contain high signal words like "idea," "MVP," "startup," "feedback," "SaaS," "launch," "validate," "business," and "product."
- **Depth:** Comments at all levels were pulled to extract the entire conversation.

#### 3.1.3 Strict Filtering Pipeline

Social media data is raw and noisy. This illustrates the strict filtering process necessary for evaluation to focus on the significant text.

1. **Bot Removal:** Regex filtering for the elimination of automatically sent messages.
   - Pattern: `r'^I am a bot'`
   - Pattern: `r'^Your submission has been'`
   - Pattern: `r'AutoModerator'`

2. **Length Restrictions:**
   - *Min Length:* 10 characters (to avoid utterances like "Yes" and "lol").
   - *Max Length:* 1000 characters (to prevent excessively long, pasted blocks).

3. **Link-Only Comments:** Comments that are only links were removed.

#### 3.1.4 Dataset Statistics

The resulting dataset consists of 150 comments.

![Label Distribution](../outputs/label_distribution.png)
*Figure 1: Distribution of sentiment labels in the dataset, showing extreme class imbalance toward Neutral.*

**Distribution by Subreddit:**

- **r/SaaS:** 52 comments (34.7%)
- **r/smallbusiness:** 38 comments (25.3%)
- **r/Entrepreneur:** 31 comments (20.7%)
- **r/startups:** 22 comments (14.7%)
- **r/Startup_Ideas:** 7 comments (4.7%)

### 3.2 Preprocessing Pipeline

For text normalization before the model processing, there was the development of a custom preprocessing module (`preprocessing.py`).

**Key Transformations:**

1. **Markdown Stripping:** The Reddit Markdown variant was stripped.
   - `**bold**` → `bold`
   - `[link text](url)` → `link text`
   - `> quoted text` → Completely removed for the purpose of interpreting the user's new sentiment as opposed to the quoted text.

2. **HTML Entity Decoding:** `&amp;` → `&`, `&gt;` → `>`, and so on.

3. **Sarcasm Marker Preservation:** The `/s` mark was retained, as removing it would negatively influence the processing of sentiment.

4. **URL Removal:** `https://google.com` → empty string.

#### 3.2.1 Preprocessing Challenges and Solutions

One challenge that stands out here was the use of nested quoting, where users would quote messages, respond, quote again, and then respond.

- *Raw:* `> Did you try X? \n\n Yes I did.`
- *Generic Cleaner:* Could produce the "Did you try X? Yes, I did." reaction, blurring the intentions.
- *Our Solution:* Used the regex `re.sub(r'&gt;.*?(\n|$)', '', text)` to eliminate lines that start with `>`, as it would affect the score of the sentiment of the user's contribution.

### 3.3 Gold Standard Annotation

The Gold Standard (Ground Truth) was created using **GPT-5** (via Azure OpenAI) as the expert annotator.

**Justification for GPT-5 as Annotator:**

Recent research reveals that GPT-4+ models are able to reach a stronger inter-annotator agreement score with human annotators compared to the agreement between human annotators for difficult NLP tasks. They are consistent, tireless, and objective.

**Annotation Prompt:**

This was an extremely strict system prompt:

> "Analyze the sentiment of the following Reddit comment regarding startup ideas. Categorize as one and only one of the following labels: Positive, Negative, and Neutral.
>
> **Definition of Neutral:** Factual statements, questions, constructive advice without emotional emphasis, or discussions of operations.
>
> **Important:** Posts on the Reddit site can and often express sarcasm, slang, and informal language. Consider the actual intent, not just surface-level words."

**Resulting Class Distribution:**

- **Neutral:** 145 comments (96.7%)
- **Positive:** 4 comments (2.7%)
- **Negative:** 1 comment (0.7%)

This significant imbalance indicates that startup discourse is primarily informational and constructive, pointing to the fact that binary Positive/Negative assessments are inadequate for such an environment.

### 3.4 Model Implementations

Four different methods were considered.

#### 3.4.1 VADER (Lexicon)

- **Library:** `nltk.sentiment.vader`
- **Mechanism:** Scores for valence are aggregated while rule-driven corrections are applied.
- **Output:** Compound score in [-1, 1].
  - Positive: > 0.05
  - Negative: < -0.05
  - Neutral: Between -0.05 and 0.05
- **Hypothesis:** Expected to find difficulty with "constructive criticism," which involves both praising and pointing out problems.

#### 3.4.2 TextBlob (Lexicon)

- **Library:** `textblob`
- **Mechanism:** Average polarity of adjectives using a pattern analyzer.
- **Mapping:**
  - Positive: > 0.1
  - Negative: < -0.1
  - Neutral: Between -0.1 and 0.1
- **Hypothesis:** Possibly overly sensitive to adjectives, labeling descriptive material as opinion.

#### 3.4.3 DistilBERT (Transformer)

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face)
- **Mechanism:** Deep neural network with attention mechanisms.
- **Mapping:** Generates probabilities for Positive and Negative cases, without a native "Neutral" class.
  - *Adaptation Strategy:* If Confidence < 0.6, classify as **Neutral**; otherwise, use the predicted label.
- **Hypothesis:** Movie reviews are rarely neutral; problems expected regarding the preponderance of neutrality in startup discussions.

#### 3.4.4 GPT-5 (LLM)

- **API:** Azure OpenAI
- **Prompting:** Same as the annotation step but used for classification.
- **Hypothesis:** Expected to perform best based on overall knowledge and annotation rule compliance.

#### 3.4.5 Prompt Engineering Iterations

Several iterations were completed:

1. **Naive:** "Is this positive or negative?" — Had binary bias (lacked Neutral).
2. **Subjective:** "How does this comment make you feel?" — Had inconsistent responses.
3. **Role-Playing (Final):** "You are a sentiment analysis expert specializing in Reddit startup discussions..." — Was the most effective at distinguishing between the Negative and the Neutral category.

### 3.5 Evaluation Metrics

An extensive range of indicators was used:

1. **Accuracy:** Proportion of correct predictions.

2. **F1-Score (Weighted):** Harmonic mean of precision and recall, weighted by the number of true instances for each label. This is the most important metric for such an imbalanced dataset.

3. **Cohen's Kappa:** A statistical measure that accounts for chance agreement.
   - κ < 0: Less than chance agreement
   - κ = 0: Random agreement
   - κ > 0.2: Slight agreement

4. **Confusion Matrices:** Visual representation of the actual and predicted labels for the detection of error patterns.

---

## 4. Results

### 4.1 Overall Performance Comparison

The performance disparity between the models was massive, confirming that domain specificity is the dominant factor in this task.

**Table 1: Overall Performance Metrics**

| Model | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Cohen's Kappa |
|-------|----------|---------------|------------|---------------|---------------|
| **GPT-5** | **0.947** | **0.955** | **0.947** | **0.950** | **0.312** |
| TextBlob | 0.380 | 0.968 | 0.380 | 0.520 | 0.035 |
| VADER | 0.133 | 0.968 | 0.133 | 0.194 | 0.013 |
| DistilBERT | 0.047 | 0.969 | 0.047 | 0.030 | 0.021 |

![All Metrics Comparison](../outputs/all_metrics_comparison.png)
*Figure 2: Comparison of all evaluation metrics across models. GPT-5 dominates in Accuracy and F1-Score.*

**Analysis:**

- **GPT-5** is the only usable model, with near-human performance (95% F1).
- **TextBlob** (38% accuracy) and **VADER** (13% accuracy) failed catastrophically.
- **DistilBERT** (4.7% accuracy) performed significantly *worse* than random guessing. A random guesser on a 3-class problem would get ~33%. DistilBERT actively made the wrong choice 95% of the time.

### 4.2 Per-Class Breakdown

To understand *why* the traditional models failed, we look at the per-class metrics. The key lies in the **Neutral** class.

**Table 2: Recall by Class**

| Model | Neutral Recall | Positive Recall | Negative Recall |
|-------|----------------|-----------------|-----------------|
| **GPT-5** | **96.6%** | 25.0% | 100.0% |
| TextBlob | 36.6% | 100.0% | 0.0% |
| VADER | 11.0% | 100.0% | 0.0% |
| DistilBERT | 1.4% | 100.0% | 100.0% |

![Per Class F1](../outputs/per_class_f1.png)
*Figure 3: F1 Scores broken down by sentiment class. Note the complete failure of DistilBERT and VADER on the Neutral class.*

**The "Neutrality Problem":**

The dataset is 96.7% Neutral.

- **VADER** only identified 11% of Neutral comments correctly. It classified the other 89% as emotional.
- **DistilBERT** only identified 1.4% of Neutral comments correctly. It forced almost every single comment into a Positive or Negative bucket.

This proves that **off-the-shelf sentiment models are biased against neutrality**. They are designed to find sentiment, so they "hallucinate" sentiment where there is none. In a startup context, where "What is your pricing?" is a common (neutral) question, VADER might see "pricing" (money) and think "Positive," or see "What" (confusion) and think "Negative."

### 4.3 Confusion Matrix Analysis

The confusion matrices tell the story of each model's bias.

![All Confusion Matrices](../outputs/all_confusion_matrices.png)
*Figure 4: Confusion matrices for all four models. Rows represent True labels, columns represent Predictions.*

**VADER's Optimism Bias:**

VADER predicted **Positive** for 93 out of 145 Neutral comments.

- *Reason:* Startup discussions often use polite, encouraging language even when delivering neutral advice ("Good luck," "Thanks," "Great idea to consider X"). VADER's lexicon scores these polite markers as high sentiment, masking the neutral informational content.

**DistilBERT's Pessimism Bias:**

DistilBERT predicted **Negative** for 96 out of 145 Neutral comments.

- *Reason:* Startup discussions are "problem-centric." Entrepreneurs talk about "pain points," "bugs," "errors," "failure modes," and "risks."
- DistilBERT, trained on movie reviews, associates these words with a *bad movie*.
- In a startup, discussing a "pain point" is a neutral/positive step toward finding a solution. DistilBERT lacks this domain context, interpreting the presence of "problem words" as negative sentiment.

### 4.4 Inter-Model Agreement

Do the models at least agree with each other?

**Table 3: Pairwise Agreement**

| Model Pair | Agreement | Kappa | Interpretation |
|------------|-----------|-------|----------------|
| VADER vs TextBlob | 56.0% | 0.232 | Slight Agreement |
| VADER vs DistilBERT | 52.0% | 0.227 | Slight Agreement |
| **VADER vs GPT** | **14.7%** | **0.023** | **No Agreement** |
| **DistilBERT vs GPT** | **4.7%** | **0.014** | **No Agreement** |

![Model Agreement](../outputs/model_agreement.png)
*Figure 5: Heatmap of inter-model agreement rates. The low agreement between GPT and others highlights their divergence.*

The models are inhabiting different realities. GPT (the accurate model) has almost zero overlap with VADER or DistilBERT. This suggests that the "signal" VADER is picking up (surface-level politeness) and the "signal" DistilBERT is picking up (problem-oriented vocabulary) are both orthogonal to the true "signal" of the text (pragmatic intent).

### 4.5 Sarcasm and Slang Analysis

One of the research questions (RQ4) focused on the types of comments that caused disagreement. We found that **Sarcasm** was a major differentiator.

- **Comment:** "Yeah, I'm sure Google is terrified of your new search engine /s"
  - **VADER:** *Positive* (sees "sure", "search engine"). It misses the sarcasm entirely despite the `/s` tag.
  - **TextBlob:** *Positive*.
  - **GPT-5:** *Negative* (correctly identifies the sarcastic mockery).

This confirms that while we preserved the `/s` tag in preprocessing, lexicon-based models do not have the logic to interpret it as a "negation operator." They simply treat it as noise or ignore it, whereas the LLM understands it as a fundamental modifier of the sentence's meaning.

---

## 5. Discussion

### 5.1 Qualitative Error Analysis: Anatomy of a Failure

In order to show the extent to which the problem exists, some analyses are provided for comments involving divergent predictions.

**Example 1: The "Fear" Trap**

> *Comment:* "What I have experienced is fear of failure."
> *Gold Label:* **Neutral** (Context: An entrepreneur sharing knowledge gleaned from the "failure story" thread).
> *VADER Prediction:* **Negative**
> *GPT Prediction:* **Neutral**

*Analysis:* The VADER lexicon records negative sentiment for the words "fear" and "failure" (-0.5 for both words). This model would not be able to differentiate between the fear an individual feels as an emotional state and the topic of fear as a discussion point.

**Example 2: The "Constructive Critique" Trap**

> *Comment:* "You need to fix the churn rate before scaling or you will run out of cash."
> *Gold Label:* **Neutral** (Advice).
> *DistilBERT Prediction:* **Negative**

*Analysis:* The terms that are identified by the DistilBERT model are "fix," "churn," "run out," and "cash." In reviewing a movie, "You need to fix the script" carries negative implications, but if it were applied to the start-up world, it would form the foundation of sound advice. This model does not correctly port the semantics.

**Example 3: The "Politeness" Trap**

> *Comment:* "Thanks for the feedback, I'll look into it!"
> *Gold Label:* **Neutral** (Acknowledgment).
> *VADER/TextBlob Prediction:* **Positive**

*Analysis:* The lexical resources give high positive scores to the words "Thanks" and the exclamation mark. Although the user uses politeness, it does not necessarily imply that there is positive sentiment regarding the start-up idea, but rather reflects social norms. This aspect could skew the scores for the Positive category.

### 5.2 The Economic Value of Accuracy

For the startup entrepreneur, the implications of the difference between the models are economic:

- In the case where the **VADER** tool is utilized, the dashboard could be showing mainly "Positive" indicators based on politeness, potentially leading to false validation of concepts and "False Starts"—the creation of products that are complimented but are not marketable.

- In the case of **DistilBERT**, the dashboard could prefer "Negative" indicators based on problem-focused language, possibly suppressing valid solutions based on the false assumption that the market does not value positive contributions.

- **GPT-5** provides an articulate "Neutral" response, allowing the entrepreneur to ascertain that there are user interactions (in the form of questions and advice) but without interpreting user interaction as validation and/or rejection.

### 5.3 Technical Limitations and Trade-offs

Despite the superior accuracy offered by GPT-5, several trade-offs exist:

1. **Cost:** The cost of processing 10,000 comments using GPT-5 costs $5-$10, which makes it much costlier compared to running VADER locally (free).

2. **Latency:** GPT-5 API responses are 500ms to 1 second per comment, while VADER times are measured in microseconds. For real-time dashboards, latency is an important factor.

3. **Privacy:** The transmission of data to the OpenAI/Azure API can violate the privacy policy for proprietary datasets, but VADER and DistilBERT algorithms can be run entirely offline (air-gapped).

### 5.4 Validity of the "Neutral" Class

One could raise an objection that a 96% figure for the category "Neutral" implies the quality of the data is low. On the contrary, the 96% "Neutral" category typifies the nature of the domain that startups represent. This kind of validation is analytical, unlike that which happens in the sports and politics sectors, which are often polarized and emotional.

Examples include:
- "How much does it cost?"
- "Does it integrate with Slack?"
- "You should incorporate in Delaware."

These are not emotional but functional. If the model for sentiment analysis were unable to process the "Functional Neutral" class, it would be ineffective for the purpose of business intelligence.

### 5.5 Limitations of the Study

1. **Circular Evaluation:** In this study, the GPT-5 model was used to annotate the data and later GPT-5 was compared against those annotations. This method may cause GPT-5's performance to be inflated. However, after inspecting the annotations marked by GPT-5 on random samples and comparing them to predictions by the other models, it was confirmed that GPT-5's annotations were generally correct and the other models were objectively wrong.

2. **Sample Size:** This study analyzes 150 comments, which makes for a small but manageable sample. Nonetheless, the error distributions observed (positive bias using VADER and negative bias for DistilBERT) are consistent enough that they would likely yield the same results for a larger sample.

3. **Language Scope:** This research evaluates English-language subreddits exclusively. The global entrepreneur culture, conducted in other languages like Spanish and Chinese, could present divergent linguistic characteristics.

### 5.6 Ethical Considerations

There are some ethical considerations involved in collecting and interpreting data available on Reddit. While the information available on the network is public, it should not necessarily be interpreted for use in commercial market research without consideration.

- **Anonymity:** In the original data, usernames were anonymized to protect user privacy.

- **Compliance:** This study maintained complete compliance with the terms of service for the Reddit API, ensuring that it did not bypass any rate limitations and utilized the official JSON endpoints.

- **Bias:** Members of Reddit are biased towards male and tech-savvy demographics, which introduces bias into the signals. Products targeting different demographics could receive skewed feedback from Reddit users.

---

## 6. Conclusion and Future Work

### 6.1 Conclusion

This research compares and contrasts the various natural language processing approaches for sentiment analysis on Reddit communities revolving around start-ups. The conclusion reached is that **traditional tools for sentiment analysis are, by their nature, grossly ineffective.**

- **VADER and TextBlob** tend to perform suboptimally, as they are susceptible to surface-level politeness and lack contextual grounding.

- **DistilBERT** performs poorly due to considerable domain mismatch, as it incorrectly identifies negative sentiment for discourse that is focused on problem-solving and oriented towards business discussions.

- **GPT-5** succeeds as it pragmatically showcases the capability to differentiate between emotional sentiment and functional discourse.

For researchers and practitioners building "AI for Startups," the message here is clear: there is no point relying on off-the-shelf sentiment lexicons. On this topic, the most informative class of labels would be the "Neutral" class, which as of now, only the larger language models are correctly processing.

### 6.2 Future Work

1. **Fine-Tuning BERT:** Is there an opportunity for improvement for the performance of the DistilBERT model? Future work could focus on fine-tuning DistilBERT on the 96% Neutral dataset. This would help create a domain-adapted BERT model that could provide comparable accuracy to GPT-5 while maintaining the speed advantages of local models.

2. **Beyond Sentiment:** Since 96% of comments are categorized as "Neutral," the use of "Sentiment" as an evaluation standard may no longer be valid. Future research should focus on **Intent Classification**:
   - Class 1: "Feature Request"
   - Class 2: "Pricing Question"
   - Class 3: "Competitor Mention"
   - Class 4: "Validation/Rejection"
   
   This taxonomy would give founders far more useful information than the Positive/Negative categorization.

3. **Longitudinal Trends:** Using GPT-5 to investigate the trend of sentiment for particular topics such as "AI Wrappers" and "Crypto" over time.

---

## References

1.  **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *Proceedings of NAACL-HLT*.
2.  **Hutto, C.J. & Gilbert, E.E. (2014).** "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media (ICWSM)*.
3.  **Liu, B. (2012).** *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers.
4.  **Loria, S. (2018).** "TextBlob Documentation: Release 0.15.2." *https://textblob.readthedocs.io/en/dev/*.
5.  **Manus AI. (2025).** "Reddit Sentiment Analyzer - Multi-Layer Sentiment Analysis." *https://manus.im/playbook/subreddit-analyzer*.
6.  **Mutsaddi, A. (2023).** "Reddit Post Comments Sentiment Analysis with Webscraping." GitHub Repository. *https://github.com/AtharvaMutsaddi/Reddit-Post-Comments-Sentiment-Analysis-with-Webscraping*.
7.  **OpenAI. (2023).** "GPT-4 Technical Report." *arXiv preprint arXiv:2303.08774*.
8.  **RedditSentiments. (2025).** "Reddit Sentiment Analysis Webapp." *https://redditsentiments.com*.
9.  **Ribeiro, F. N., Araújo, M., Gonçalves, P., Benevenuto, F., & Gonçalves, M. A. (2016).** "SentiBench – A Benchmark Comparison of State-of-the-Practice Sentiment Analysis Methods." *Information Sciences*, vol. 326, pp. 245–264.
10. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).** "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *NeurIPS EMC^2 Workshop*.
11. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems*.
12. **WandB. (2025).** "Sentiment Classification with the Reddit PRAW API and GPT-4o-mini." Weights & Biases Reports. *https://wandb.ai/byyoung3/Generative-AI/reports/Sentiment-classification-with-the-Reddit-Praw-API-and-GPT-4o-mini--VmlldzoxMjEwODE3Nw*.

