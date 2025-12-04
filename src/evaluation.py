"""
Evaluation Module
Computes metrics, generates confusion matrices, and creates visualizations for the research paper.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)


def compute_metrics(y_true: List[str], y_pred: List[str], model_name: str = "Model") -> Dict:
    """
    Compute comprehensive metrics for a single model.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1 (overall and per-class)
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    labels = ['Positive', 'Negative', 'Neutral']
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'kappa': kappa,
        'per_class': {
            label: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': int(support[i]) if support is not None else 0
            }
            for i, label in enumerate(labels)
        }
    }


def compute_model_agreement(predictions1: List[str], predictions2: List[str], 
                            name1: str = "Model1", name2: str = "Model2") -> Dict:
    """
    Compute agreement between two models' predictions.
    """
    agreement_rate = sum(p1 == p2 for p1, p2 in zip(predictions1, predictions2)) / len(predictions1)
    kappa = cohen_kappa_score(predictions1, predictions2)
    
    return {
        'models': f"{name1} vs {name2}",
        'agreement_rate': agreement_rate,
        'kappa': kappa
    }


def generate_confusion_matrix(y_true: List[str], y_pred: List[str], 
                               labels: List[str] = None) -> np.ndarray:
    """Generate confusion matrix."""
    if labels is None:
        labels = ['Positive', 'Negative', 'Neutral']
    return confusion_matrix(y_true, y_pred, labels=labels)


def evaluate_all_models(df: pd.DataFrame, gold_column: str = 'gold_label',
                        model_columns: List[str] = None) -> Dict:
    """
    Evaluate all models against gold labels.
    
    Args:
        df: DataFrame with predictions
        gold_column: Column name for gold/ground truth labels
        model_columns: List of column names containing model predictions
    
    Returns:
        Dictionary with all evaluation results
    """
    if model_columns is None:
        model_columns = ['VADER_label', 'TextBlob_label', 'DistilBERT_label', 'GPT_label']
    
    # Filter to only include valid gold labels
    valid_labels = ['Positive', 'Negative', 'Neutral']
    df_eval = df[df[gold_column].isin(valid_labels)].copy()
    
    y_true = df_eval[gold_column].tolist()
    
    results = {
        'model_metrics': [],
        'confusion_matrices': {},
        'model_agreements': [],
        'summary': {}
    }
    
    # Evaluate each model
    available_models = [col for col in model_columns if col in df_eval.columns and df_eval[col].iloc[0] != 'N/A']
    
    for model_col in available_models:
        y_pred = df_eval[model_col].tolist()
        model_name = model_col.replace('_label', '')
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, model_name)
        results['model_metrics'].append(metrics)
        
        # Generate confusion matrix
        cm = generate_confusion_matrix(y_true, y_pred)
        results['confusion_matrices'][model_name] = cm.tolist()
    
    # Compute inter-model agreement
    for i, model1 in enumerate(available_models):
        for model2 in available_models[i+1:]:
            pred1 = df_eval[model1].tolist()
            pred2 = df_eval[model2].tolist()
            name1 = model1.replace('_label', '')
            name2 = model2.replace('_label', '')
            
            agreement = compute_model_agreement(pred1, pred2, name1, name2)
            results['model_agreements'].append(agreement)
    
    # Summary statistics
    results['summary'] = {
        'total_samples': len(df_eval),
        'label_distribution': df_eval[gold_column].value_counts().to_dict(),
        'best_model_accuracy': max(m['accuracy'] for m in results['model_metrics']),
        'best_model_f1': max(m['f1_weighted'] for m in results['model_metrics'])
    }
    
    return results


def format_metrics_table(results: Dict) -> str:
    """
    Format metrics as a markdown table for the paper.
    """
    table = "| Model | Accuracy | Precision | Recall | F1 (Weighted) | F1 (Macro) | Kappa |\n"
    table += "|-------|----------|-----------|--------|---------------|------------|-------|\n"
    
    for metrics in results['model_metrics']:
        table += f"| {metrics['model']} | "
        table += f"{metrics['accuracy']:.3f} | "
        table += f"{metrics['precision_weighted']:.3f} | "
        table += f"{metrics['recall_weighted']:.3f} | "
        table += f"{metrics['f1_weighted']:.3f} | "
        table += f"{metrics['f1_macro']:.3f} | "
        table += f"{metrics['kappa']:.3f} |\n"
    
    return table


def format_per_class_table(results: Dict) -> str:
    """
    Format per-class metrics as a markdown table.
    """
    table = "| Model | Class | Precision | Recall | F1 | Support |\n"
    table += "|-------|-------|-----------|--------|----|---------|\n"
    
    for metrics in results['model_metrics']:
        for label, class_metrics in metrics['per_class'].items():
            table += f"| {metrics['model']} | {label} | "
            table += f"{class_metrics['precision']:.3f} | "
            table += f"{class_metrics['recall']:.3f} | "
            table += f"{class_metrics['f1']:.3f} | "
            table += f"{class_metrics['support']} |\n"
    
    return table


def format_agreement_table(results: Dict) -> str:
    """
    Format model agreement as a markdown table.
    """
    table = "| Model Pair | Agreement Rate | Cohen's Kappa |\n"
    table += "|------------|----------------|---------------|\n"
    
    for agreement in results['model_agreements']:
        table += f"| {agreement['models']} | "
        table += f"{agreement['agreement_rate']:.1%} | "
        table += f"{agreement['kappa']:.3f} |\n"
    
    return table


def format_confusion_matrix_text(cm: List[List[int]], model_name: str) -> str:
    """
    Format confusion matrix as text for the paper.
    """
    labels = ['Positive', 'Negative', 'Neutral']
    
    text = f"\nConfusion Matrix - {model_name}:\n"
    text += "                  Predicted\n"
    text += "              Pos    Neg    Neu\n"
    text += "Actual " + "-" * 30 + "\n"
    
    for i, label in enumerate(labels):
        text += f"  {label[:3]:>3}  |  "
        text += "  ".join(f"{cm[i][j]:>4}" for j in range(3))
        text += "\n"
    
    return text


def find_disagreement_examples(df: pd.DataFrame, gold_column: str = 'gold_label',
                                model_columns: List[str] = None, n_examples: int = 10) -> pd.DataFrame:
    """
    Find examples where models disagree with gold labels or each other.
    Useful for error analysis in the paper.
    """
    if model_columns is None:
        model_columns = ['VADER_label', 'TextBlob_label', 'DistilBERT_label', 'GPT_label']
    
    available_models = [col for col in model_columns if col in df.columns and df[col].iloc[0] != 'N/A']
    
    examples = []
    
    for idx, row in df.iterrows():
        gold = row[gold_column]
        predictions = {col.replace('_label', ''): row[col] for col in available_models}
        
        # Count disagreements
        disagreements = sum(1 for pred in predictions.values() if pred != gold)
        unique_predictions = len(set(predictions.values()))
        
        if disagreements > 0 or unique_predictions > 1:
            examples.append({
                'text': row.get('text_clean', row.get('text', ''))[:200],
                'gold_label': gold,
                **predictions,
                'n_disagreements': disagreements,
                'n_unique_predictions': unique_predictions
            })
    
    # Sort by number of disagreements (most interesting first)
    examples_df = pd.DataFrame(examples)
    if len(examples_df) > 0:
        examples_df = examples_df.sort_values('n_disagreements', ascending=False).head(n_examples)
    
    return examples_df


def save_evaluation_results(results: Dict, output_dir: str = "outputs"):
    """
    Save all evaluation results to files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save raw results as JSON
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save formatted tables as text
    with open(f"{output_dir}/metrics_tables.md", 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(format_metrics_table(results))
        f.write("\n\n## Per-Class Metrics\n\n")
        f.write(format_per_class_table(results))
        f.write("\n\n## Model Agreement\n\n")
        f.write(format_agreement_table(results))
        f.write("\n\n## Confusion Matrices\n")
        for model_name, cm in results['confusion_matrices'].items():
            f.write(format_confusion_matrix_text(cm, model_name))
    
    print(f"💾 Results saved to {output_dir}/")


if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing evaluation module...")
    
    # Create sample data
    sample_data = {
        'text_clean': [
            "Great idea!", "Terrible concept", "Interesting approach",
            "Love it!", "Won't work", "Need more info",
            "Amazing!", "Bad idea", "Could be good"
        ],
        'gold_label': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral'],
        'VADER_label': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Positive'],
        'TextBlob_label': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral'],
        'DistilBERT_label': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral'],
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run evaluation
    results = evaluate_all_models(df)
    
    print("\n📊 Overall Metrics:")
    print(format_metrics_table(results))
    
    print("\n🤝 Model Agreement:")
    print(format_agreement_table(results))

