#!/usr/bin/env python3
"""
Reddit Sentiment Analysis - Main Pipeline
==========================================
This script runs the complete sentiment analysis pipeline:
1. Data Collection (Reddit JSON scraping)
2. Preprocessing
3. Sentiment Analysis (VADER, TextBlob, DistilBERT, GPT)
4. Evaluation & Visualization

Usage:
    python main.py                    # Run full pipeline
    python main.py --skip-collection  # Skip data collection, use existing data
    python main.py --skip-gpt         # Skip GPT (saves API costs)
    python main.py --sample 100       # Limit to 100 comments
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_collection import collect_startup_comments, save_raw_data, TARGET_SUBREDDITS
from preprocessing import preprocess_dataframe, prepare_for_labeling
from sentiment_models import run_all_models
from evaluation import evaluate_all_models, save_evaluation_results, find_disagreement_examples, format_metrics_table
from visualizations import generate_all_visualizations


def run_pipeline(args):
    """Run the complete sentiment analysis pipeline."""
    
    print("=" * 60)
    print("🚀 REDDIT SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Collection
    if not args.skip_collection:
        print("\n📥 STEP 1: Collecting Reddit Data")
        print("-" * 40)
        
        comments_df = collect_startup_comments(
            subreddits=TARGET_SUBREDDITS,
            posts_per_subreddit=args.posts_per_sub,
            comments_per_post=args.comments_per_post,
            delay=1.5
        )
        
        if len(comments_df) == 0:
            print("❌ No comments collected. Check internet connection.")
            return
        
        save_raw_data(comments_df, "data/raw_comments.csv")
    else:
        print("\n📥 STEP 1: Loading Existing Data")
        print("-" * 40)
        
        if not Path("data/raw_comments.csv").exists():
            print("❌ No existing data found. Run without --skip-collection first.")
            return
        
        comments_df = pd.read_csv("data/raw_comments.csv")
        print(f"   Loaded {len(comments_df)} comments from data/raw_comments.csv")
    
    # Step 2: Preprocessing
    print("\n🔧 STEP 2: Preprocessing")
    print("-" * 40)
    
    processed_df = preprocess_dataframe(comments_df)
    
    # Sample if requested
    if args.sample and len(processed_df) > args.sample:
        processed_df = processed_df.sample(args.sample, random_state=42)
        print(f"   Sampled {args.sample} comments for analysis")
    
    processed_df.to_csv("data/processed_comments.csv", index=False)
    
    # Step 3: Check for Gold Labels or Auto-Label
    print("\n🏷️  STEP 3: Sentiment Labeling")
    print("-" * 40)
    
    labeled_path = Path("data/labeled_comments.csv")
    
    if labeled_path.exists() and not args.auto_label:
        print("   Found existing labeled data!")
        labeled_df = pd.read_csv(labeled_path)
        
        # Merge with processed data
        if 'text_clean' not in labeled_df.columns:
            labeled_df = labeled_df.merge(
                processed_df[['comment_id', 'text_clean']], 
                on='comment_id', 
                how='left'
            )
    else:
        if args.auto_label:
            print("   Auto-labeling with GPT (this will be used as gold labels)...")
            # Use GPT predictions as gold labels for demonstration
            from sentiment_models import AzureOpenAISentiment
            
            try:
                gpt = AzureOpenAISentiment()
                sample_size = min(args.sample or 150, len(processed_df))
                labeled_df = processed_df.head(sample_size).copy()
                labeled_df['gold_label'] = gpt.analyze_batch(
                    labeled_df['text_clean'].tolist(), 
                    delay=0.3
                )
                labeled_df.to_csv(labeled_path, index=False)
                print(f"   Auto-labeled {len(labeled_df)} comments and saved to {labeled_path}")
            except Exception as e:
                print(f"   ⚠️ Auto-labeling failed: {e}")
                print("   Creating labeling template instead...")
                prepare_for_labeling(processed_df, sample_size=150, output_path="data/for_labeling.csv")
                print("\n   ❗ Please manually label data/for_labeling.csv")
                print("   Then rename it to data/labeled_comments.csv and re-run.")
                return
        else:
            print("   No labeled data found. Creating labeling template...")
            prepare_for_labeling(processed_df, sample_size=150, output_path="data/for_labeling.csv")
            print("\n   ❗ Please manually label data/for_labeling.csv")
            print("   Then rename it to data/labeled_comments.csv and re-run.")
            print("\n   Or run with --auto-label to use GPT as gold labeler (for demonstration)")
            return
    
    # Step 4: Run Sentiment Models
    print("\n🔬 STEP 4: Running Sentiment Models")
    print("-" * 40)
    
    results_df = run_all_models(
        labeled_df, 
        text_column='text_clean',
        use_gpt=not args.skip_gpt
    )
    
    results_df.to_csv("data/results.csv", index=False)
    print(f"   Saved results to data/results.csv")
    
    # Step 5: Evaluation
    print("\n📊 STEP 5: Evaluation")
    print("-" * 40)
    
    model_columns = ['VADER_label', 'TextBlob_label', 'DistilBERT_label']
    if not args.skip_gpt:
        model_columns.append('GPT_label')
    
    eval_results = evaluate_all_models(
        results_df, 
        gold_column='gold_label',
        model_columns=model_columns
    )
    
    save_evaluation_results(eval_results, output_dir="outputs")
    
    # Print summary
    print("\n📋 RESULTS SUMMARY")
    print("=" * 60)
    print(format_metrics_table(eval_results))
    
    # Step 6: Visualizations
    print("\n🎨 STEP 6: Generating Visualizations")
    print("-" * 40)
    
    generate_all_visualizations(results_df, eval_results, output_dir="outputs")
    
    # Step 7: Error Analysis
    print("\n🔍 STEP 7: Error Analysis")
    print("-" * 40)
    
    disagreements = find_disagreement_examples(
        results_df, 
        gold_column='gold_label',
        model_columns=model_columns,
        n_examples=15
    )
    
    if len(disagreements) > 0:
        disagreements.to_csv("outputs/error_analysis.csv", index=False)
        print(f"   Found {len(disagreements)} disagreement examples")
        print(f"   Saved to outputs/error_analysis.csv")
        
        print("\n   Sample disagreements:")
        for i, row in disagreements.head(5).iterrows():
            print(f"\n   Text: {row['text'][:80]}...")
            print(f"   Gold: {row['gold_label']}", end="")
            for col in model_columns:
                model_name = col.replace('_label', '')
                if model_name in row:
                    print(f" | {model_name}: {row[model_name]}", end="")
            print()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("   - data/raw_comments.csv        (scraped data)")
    print("   - data/processed_comments.csv  (cleaned data)")
    print("   - data/results.csv             (all predictions)")
    print("   - outputs/evaluation_results.json")
    print("   - outputs/metrics_tables.md    (copy-paste for paper)")
    print("   - outputs/*.png                (figures for paper)")
    print("   - outputs/error_analysis.csv   (disagreement examples)")
    
    print(f"\n📊 Dataset: {len(results_df)} comments analyzed")
    print(f"🏆 Best model: {max(eval_results['model_metrics'], key=lambda x: x['f1_weighted'])['model']}")
    print(f"   F1 Score: {eval_results['summary']['best_model_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Reddit Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection, use existing data')
    parser.add_argument('--skip-gpt', action='store_true',
                        help='Skip GPT analysis (saves API costs)')
    parser.add_argument('--auto-label', action='store_true',
                        help='Use GPT to auto-generate gold labels (for demonstration)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Limit analysis to N comments')
    parser.add_argument('--posts-per-sub', type=int, default=15,
                        help='Number of posts to fetch per subreddit')
    parser.add_argument('--comments-per-post', type=int, default=25,
                        help='Number of comments per post')
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

