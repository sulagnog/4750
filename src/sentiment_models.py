"""
Sentiment Analysis Models Module
Implements multiple sentiment analysis approaches:
1. VADER (lexicon-based)
2. TextBlob (lexicon-based)
3. DistilBERT (transformer-based)
4. Azure OpenAI GPT (LLM-based)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sentiment thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05
TEXTBLOB_POS_THRESHOLD = 0.1
TEXTBLOB_NEG_THRESHOLD = -0.1


class VADERSentiment:
    """VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer."""
    
    def __init__(self):
        import nltk
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        
        print("✅ VADER initialized")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        Returns dict with scores and label.
        """
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound > VADER_POS_THRESHOLD:
            label = 'Positive'
        elif compound < VADER_NEG_THRESHOLD:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'compound': compound,
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu'],
            'label': label
        }
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """Analyze a batch of texts and return labels."""
        return [self.analyze(text)['label'] for text in texts]


class TextBlobSentiment:
    """TextBlob sentiment analyzer."""
    
    def __init__(self):
        from textblob import TextBlob
        self.TextBlob = TextBlob
        print("✅ TextBlob initialized")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        Returns dict with polarity, subjectivity, and label.
        """
        blob = self.TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > TEXTBLOB_POS_THRESHOLD:
            label = 'Positive'
        elif polarity < TEXTBLOB_NEG_THRESHOLD:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """Analyze a batch of texts and return labels."""
        return [self.analyze(text)['label'] for text in texts]


class DistilBERTSentiment:
    """DistilBERT transformer-based sentiment analyzer."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        from transformers import pipeline
        
        print(f"🔄 Loading DistilBERT model: {model_name}...")
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512
        )
        print("✅ DistilBERT initialized")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        Maps POSITIVE/NEGATIVE to our 3-class scheme.
        """
        try:
            result = self.classifier(text[:512])[0]
            hf_label = result['label']
            score = result['score']
            
            # Map to 3-class scheme
            # If confidence is low, mark as Neutral
            if score < 0.6:
                label = 'Neutral'
            elif hf_label == 'POSITIVE':
                label = 'Positive'
            else:
                label = 'Negative'
            
            return {
                'hf_label': hf_label,
                'confidence': score,
                'label': label
            }
        except Exception as e:
            print(f"⚠️ DistilBERT error: {e}")
            return {'hf_label': 'ERROR', 'confidence': 0, 'label': 'Neutral'}
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """Analyze a batch of texts and return labels."""
        labels = []
        for i, text in enumerate(texts):
            if i % 50 == 0 and i > 0:
                print(f"   Processed {i}/{len(texts)} comments...")
            labels.append(self.analyze(text)['label'])
        return labels


class AzureOpenAISentiment:
    """Azure OpenAI GPT-based sentiment analyzer."""
    
    def __init__(self):
        from openai import AzureOpenAI
        
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError(
                "Missing Azure OpenAI credentials. Set environment variables:\n"
                "  AZURE_OPENAI_ENDPOINT\n"
                "  AZURE_OPENAI_API_KEY\n"
                "  AZURE_OPENAI_DEPLOYMENT"
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        print(f"✅ Azure OpenAI initialized (deployment: {self.deployment})")
    
    def _create_prompt(self, text: str) -> str:
        """Create the classification prompt."""
        return f"""Analyze the sentiment of the following Reddit comment about startup ideas.

Classify as exactly ONE of these labels:
- Positive: encouraging, excited, interested, sees potential, supportive
- Negative: dismissive, skeptical, critical, sees problems, discouraging
- Neutral: factual, asks questions, no clear emotional sentiment

Important: Reddit comments may contain sarcasm (often marked with /s), slang, or informal language. Consider the actual intent, not just surface-level words.

Comment: "{text}"

Respond with ONLY the label (Positive, Negative, or Neutral). No explanation."""
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment using Azure OpenAI.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert specializing in Reddit startup discussions."},
                    {"role": "user", "content": self._create_prompt(text)}
                ],
                max_completion_tokens=50
            )
            
            label = response.choices[0].message.content.strip()
            
            # Validate and normalize label
            label_lower = label.lower()
            if 'positive' in label_lower:
                label = 'Positive'
            elif 'negative' in label_lower:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return {
                'raw_response': response.choices[0].message.content,
                'label': label
            }
        
        except Exception as e:
            print(f"⚠️ Azure OpenAI error: {e}")
            return {'raw_response': str(e), 'label': 'Neutral'}
    
    def analyze_batch(self, texts: List[str], delay: float = 0.5) -> List[str]:
        """
        Analyze a batch of texts with rate limiting.
        """
        import time
        labels = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"   GPT processed {i}/{len(texts)} comments...")
            
            result = self.analyze(text)
            labels.append(result['label'])
            
            if delay > 0:
                time.sleep(delay)
        
        return labels


def run_all_models(df: pd.DataFrame, text_column: str = 'text_clean', use_gpt: bool = True) -> pd.DataFrame:
    """
    Run all sentiment models on a DataFrame and add prediction columns.
    
    Args:
        df: DataFrame with text data
        text_column: Name of column containing text to analyze
        use_gpt: Whether to run Azure OpenAI (costs money, takes time)
    
    Returns:
        DataFrame with added prediction columns
    """
    df = df.copy()
    texts = df[text_column].tolist()
    
    print(f"\n🔬 Running sentiment analysis on {len(texts)} comments...")
    print("=" * 50)
    
    # VADER
    print("\n1️⃣ Running VADER...")
    vader = VADERSentiment()
    df['VADER_label'] = vader.analyze_batch(texts)
    
    # TextBlob
    print("\n2️⃣ Running TextBlob...")
    textblob = TextBlobSentiment()
    df['TextBlob_label'] = textblob.analyze_batch(texts)
    
    # DistilBERT
    print("\n3️⃣ Running DistilBERT...")
    distilbert = DistilBERTSentiment()
    df['DistilBERT_label'] = distilbert.analyze_batch(texts)
    
    # Azure OpenAI GPT
    if use_gpt:
        print("\n4️⃣ Running Azure OpenAI GPT...")
        try:
            gpt = AzureOpenAISentiment()
            df['GPT_label'] = gpt.analyze_batch(texts, delay=0.3)
        except Exception as e:
            print(f"⚠️ Skipping GPT due to error: {e}")
            df['GPT_label'] = 'N/A'
    else:
        print("\n4️⃣ Skipping Azure OpenAI GPT (use_gpt=False)")
        df['GPT_label'] = 'N/A'
    
    print("\n✅ All models complete!")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing sentiment models...")
    
    test_texts = [
        "This is a great idea! I love it and would definitely use this product.",
        "Terrible concept. This will never work and is a complete waste of time.",
        "Interesting approach. Have you considered the market size for this?",
        "Yeah right, this is totally going to be the next unicorn /s",
        "The UI looks clean but I'm not sure about the business model.",
    ]
    
    test_df = pd.DataFrame({'text_clean': test_texts})
    
    # Run models (skip GPT for quick test)
    result_df = run_all_models(test_df, use_gpt=False)
    
    print("\n📊 Results:")
    print(result_df[['text_clean', 'VADER_label', 'TextBlob_label', 'DistilBERT_label']].to_string())

