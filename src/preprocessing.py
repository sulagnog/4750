"""
Text Preprocessing Module
Cleans and prepares Reddit comments for sentiment analysis.
"""

import re
import pandas as pd
from typing import Optional
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean a single comment text for sentiment analysis.
    
    Cleaning steps:
    1. Remove URLs
    2. Remove Reddit-specific markdown
    3. Remove excessive whitespace
    4. Preserve important sentiment markers like /s (sarcasm)
    5. Remove special characters but keep basic punctuation
    """
    if not isinstance(text, str):
        return ""
    
    # Store sarcasm marker if present
    has_sarcasm = "/s" in text.lower() or "\\s" in text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](link) -> text
    text = re.sub(r'&gt;.*?(\n|$)', '', text)  # Remove quoted text
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#x200B;', '', text)  # Zero-width space
    
    # Remove Reddit formatting markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
    text = re.sub(r'\^+', '', text)  # Superscript markers
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Add sarcasm marker back if it was present
    if has_sarcasm and "/s" not in text.lower():
        text = text + " /s"
    
    return text


def filter_comments(df: pd.DataFrame, min_length: int = 10, max_length: int = 1000) -> pd.DataFrame:
    """
    Filter comments based on quality criteria.
    
    Removes:
    - Very short comments (< min_length characters)
    - Very long comments (> max_length characters)  
    - Bot/automated comments
    - Non-English or gibberish comments
    """
    original_count = len(df)
    
    # Ensure text column exists
    if 'text' not in df.columns:
        print("⚠️ No 'text' column found in DataFrame")
        return df
    
    # Remove empty/null texts
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 0]
    
    # Filter by length
    df = df[df['text'].str.len() >= min_length]
    df = df[df['text'].str.len() <= max_length]
    
    # Remove common bot patterns
    bot_patterns = [
        r'^I am a bot',
        r'^This is a bot',
        r'beep boop',
        r'^Your post has been',
        r'^Your submission has been',
        r'^Thank you for your submission',
        r'I\'m a bot',
    ]
    
    for pattern in bot_patterns:
        df = df[~df['text'].str.contains(pattern, case=False, regex=True, na=False)]
    
    # Remove comments that are just links or mentions
    df = df[~df['text'].str.match(r'^u/\w+\s*$', na=False)]  # Just a user mention
    df = df[~df['text'].str.match(r'^r/\w+\s*$', na=False)]  # Just a subreddit mention
    
    # Remove comments with too many special characters (likely not real text)
    df = df[df['text'].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)) / max(len(x), 1) > 0.5)]
    
    removed_count = original_count - len(df)
    print(f"🧹 Filtered out {removed_count} comments ({original_count} → {len(df)})")
    
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame of comments.
    """
    print("🔄 Starting preprocessing pipeline...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Clean text
    print("   Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Filter comments
    print("   Filtering comments...")
    df = filter_comments(df, min_length=10, max_length=1000)
    
    # Remove any rows where cleaning resulted in empty text
    df = df[df['text_clean'].str.len() > 0]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"✅ Preprocessing complete. {len(df)} comments ready for analysis.")
    
    return df


def prepare_for_labeling(df: pd.DataFrame, sample_size: int = 200, output_path: str = "data/for_labeling.csv") -> pd.DataFrame:
    """
    Prepare a sample of comments for manual labeling.
    
    Creates a CSV with columns for manual sentiment labeling:
    - comment_id
    - text_clean
    - subreddit
    - post_title
    - gold_label (empty, to be filled manually)
    """
    # Sample comments if we have more than requested
    if len(df) > sample_size:
        # Stratified sampling by subreddit if possible
        if 'subreddit' in df.columns:
            sample_df = df.groupby('subreddit', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // df['subreddit'].nunique() + 1)),
                include_groups=False
            ).head(sample_size)
        else:
            sample_df = df.sample(sample_size)
    else:
        sample_df = df.copy()
    
    # Select columns for labeling
    label_columns = ['comment_id', 'text_clean', 'subreddit', 'post_title']
    available_columns = [col for col in label_columns if col in sample_df.columns]
    
    label_df = sample_df[available_columns].copy()
    
    # Add empty gold_label column
    label_df['gold_label'] = ''
    
    # Add helper columns
    label_df['notes'] = ''
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(output_path, index=False)
    
    print(f"📝 Created labeling file with {len(label_df)} comments: {output_path}")
    print("\n   Labeling instructions:")
    print("   - Open the CSV in Excel/Google Sheets")
    print("   - Fill 'gold_label' column with: Positive, Negative, or Neutral")
    print("   - Positive: encouraging, excited, sees potential")
    print("   - Negative: dismissive, skeptical, critical")
    print("   - Neutral: factual, questions, no clear sentiment")
    
    return label_df


def load_labeled_data(labeled_path: str = "data/labeled_comments.csv") -> pd.DataFrame:
    """
    Load manually labeled comments and validate.
    """
    df = pd.read_csv(labeled_path)
    
    # Validate labels
    valid_labels = ['Positive', 'Negative', 'Neutral', 'positive', 'negative', 'neutral']
    
    if 'gold_label' not in df.columns:
        raise ValueError("Missing 'gold_label' column in labeled data")
    
    # Standardize labels
    df['gold_label'] = df['gold_label'].str.strip().str.capitalize()
    
    # Filter to only labeled rows
    labeled_df = df[df['gold_label'].isin(['Positive', 'Negative', 'Neutral'])]
    
    print(f"📊 Loaded {len(labeled_df)} labeled comments")
    print(f"   Label distribution:")
    print(labeled_df['gold_label'].value_counts().to_string())
    
    return labeled_df


if __name__ == "__main__":
    # Test preprocessing
    print("🧪 Testing preprocessing module...")
    
    # Load raw data if exists
    raw_path = "data/raw_comments.csv"
    if Path(raw_path).exists():
        df = pd.read_csv(raw_path)
        df = preprocess_dataframe(df)
        
        # Save processed data
        df.to_csv("data/processed_comments.csv", index=False)
        
        # Prepare for labeling
        prepare_for_labeling(df, sample_size=200)
    else:
        print(f"⚠️ No raw data found at {raw_path}. Run data_collection.py first.")

