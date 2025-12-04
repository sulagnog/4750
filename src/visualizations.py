"""
Visualization Module
Creates charts and figures for the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path


# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def plot_label_distribution(df: pd.DataFrame, label_column: str = 'gold_label',
                            output_path: str = "outputs/label_distribution.png"):
    """
    Plot the distribution of sentiment labels in the dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    
    counts = df[label_column].value_counts()
    bars = ax.bar(counts.index, counts.values, 
                  color=[colors.get(label, '#95a5a6') for label in counts.index])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Sentiment Label')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Distribution of Gold Standard Labels')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved label distribution plot to {output_path}")


def plot_model_comparison(results: Dict, metric: str = 'f1_weighted',
                          output_path: str = "outputs/model_comparison.png"):
    """
    Plot comparison of model performance metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m['model'] for m in results['model_metrics']]
    values = [m[metric] for m in results['model_metrics']]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved model comparison plot to {output_path}")


def plot_all_metrics_comparison(results: Dict, output_path: str = "outputs/all_metrics_comparison.png"):
    """
    Plot multiple metrics for all models as grouped bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = [m['model'] for m in results['model_metrics']]
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [m[metric] for m in results['model_metrics']]
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Sentiment Analysis Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved all metrics comparison to {output_path}")


def plot_confusion_matrix(cm: List[List[int]], model_name: str,
                          output_path: str = None):
    """
    Plot a single confusion matrix as a heatmap.
    """
    if output_path is None:
        output_path = f"outputs/confusion_matrix_{model_name.lower()}.png"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Positive', 'Negative', 'Neutral']
    cm_array = np.array(cm)
    
    # Create heatmap
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved confusion matrix for {model_name} to {output_path}")


def plot_all_confusion_matrices(results: Dict, output_path: str = "outputs/all_confusion_matrices.png"):
    """
    Plot all confusion matrices in a single figure.
    """
    n_models = len(results['confusion_matrices'])
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    labels = ['Positive', 'Negative', 'Neutral']
    
    for i, (model_name, cm) in enumerate(results['confusion_matrices'].items()):
        ax = axes[i]
        cm_array = np.array(cm)
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(model_name, fontweight='bold')
    
    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Confusion Matrices by Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved all confusion matrices to {output_path}")


def plot_per_class_f1(results: Dict, output_path: str = "outputs/per_class_f1.png"):
    """
    Plot F1 scores per class for each model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [m['model'] for m in results['model_metrics']]
    classes = ['Positive', 'Negative', 'Neutral']
    
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, metrics in enumerate(results['model_metrics']):
        f1_scores = [metrics['per_class'][cls]['f1'] for cls in classes]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=metrics['model'], color=colors[i])
    
    ax.set_xlabel('Sentiment Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved per-class F1 plot to {output_path}")


def plot_model_agreement_heatmap(results: Dict, output_path: str = "outputs/model_agreement.png"):
    """
    Plot model agreement as a heatmap.
    """
    # Build agreement matrix
    models = list(set(
        m.split(' vs ')[0] for m in [a['models'] for a in results['model_agreements']]
    ).union(
        m.split(' vs ')[1] for m in [a['models'] for a in results['model_agreements']]
    ))
    
    n = len(models)
    agreement_matrix = np.eye(n)  # Diagonal is 1 (self-agreement)
    
    for agreement in results['model_agreements']:
        m1, m2 = agreement['models'].split(' vs ')
        i, j = models.index(m1), models.index(m2)
        agreement_matrix[i, j] = agreement['agreement_rate']
        agreement_matrix[j, i] = agreement['agreement_rate']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=models, yticklabels=models, ax=ax,
                vmin=0, vmax=1, annot_kws={'size': 12})
    
    ax.set_title('Inter-Model Agreement Rate', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved model agreement heatmap to {output_path}")


def generate_all_visualizations(df: pd.DataFrame, results: Dict, output_dir: str = "outputs"):
    """
    Generate all visualizations for the paper.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n🎨 Generating visualizations...")
    print("=" * 50)
    
    # Label distribution
    if 'gold_label' in df.columns:
        plot_label_distribution(df, output_path=f"{output_dir}/label_distribution.png")
    
    # Model comparison charts
    plot_model_comparison(results, metric='f1_weighted', 
                         output_path=f"{output_dir}/model_comparison_f1.png")
    plot_model_comparison(results, metric='accuracy',
                         output_path=f"{output_dir}/model_comparison_accuracy.png")
    plot_all_metrics_comparison(results, output_path=f"{output_dir}/all_metrics_comparison.png")
    
    # Per-class F1
    plot_per_class_f1(results, output_path=f"{output_dir}/per_class_f1.png")
    
    # Confusion matrices
    for model_name, cm in results['confusion_matrices'].items():
        plot_confusion_matrix(cm, model_name, 
                             output_path=f"{output_dir}/confusion_matrix_{model_name.lower()}.png")
    plot_all_confusion_matrices(results, output_path=f"{output_dir}/all_confusion_matrices.png")
    
    # Model agreement
    if results['model_agreements']:
        plot_model_agreement_heatmap(results, output_path=f"{output_dir}/model_agreement.png")
    
    print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing visualization module...")
    
    # Create sample results
    sample_results = {
        'model_metrics': [
            {'model': 'VADER', 'accuracy': 0.72, 'precision_weighted': 0.71, 
             'recall_weighted': 0.72, 'f1_weighted': 0.71, 'f1_macro': 0.68, 'kappa': 0.55,
             'per_class': {
                 'Positive': {'precision': 0.75, 'recall': 0.80, 'f1': 0.77, 'support': 50},
                 'Negative': {'precision': 0.70, 'recall': 0.65, 'f1': 0.67, 'support': 40},
                 'Neutral': {'precision': 0.68, 'recall': 0.70, 'f1': 0.69, 'support': 60}
             }},
            {'model': 'TextBlob', 'accuracy': 0.68, 'precision_weighted': 0.67,
             'recall_weighted': 0.68, 'f1_weighted': 0.67, 'f1_macro': 0.65, 'kappa': 0.50,
             'per_class': {
                 'Positive': {'precision': 0.70, 'recall': 0.75, 'f1': 0.72, 'support': 50},
                 'Negative': {'precision': 0.65, 'recall': 0.60, 'f1': 0.62, 'support': 40},
                 'Neutral': {'precision': 0.66, 'recall': 0.68, 'f1': 0.67, 'support': 60}
             }},
            {'model': 'DistilBERT', 'accuracy': 0.78, 'precision_weighted': 0.77,
             'recall_weighted': 0.78, 'f1_weighted': 0.77, 'f1_macro': 0.75, 'kappa': 0.65,
             'per_class': {
                 'Positive': {'precision': 0.80, 'recall': 0.82, 'f1': 0.81, 'support': 50},
                 'Negative': {'precision': 0.75, 'recall': 0.72, 'f1': 0.73, 'support': 40},
                 'Neutral': {'precision': 0.76, 'recall': 0.78, 'f1': 0.77, 'support': 60}
             }},
        ],
        'confusion_matrices': {
            'VADER': [[40, 5, 5], [8, 26, 6], [10, 8, 42]],
            'TextBlob': [[38, 6, 6], [10, 24, 6], [12, 10, 38]],
            'DistilBERT': [[41, 4, 5], [6, 29, 5], [8, 6, 46]],
        },
        'model_agreements': [
            {'models': 'VADER vs TextBlob', 'agreement_rate': 0.72, 'kappa': 0.55},
            {'models': 'VADER vs DistilBERT', 'agreement_rate': 0.68, 'kappa': 0.50},
            {'models': 'TextBlob vs DistilBERT', 'agreement_rate': 0.65, 'kappa': 0.45},
        ]
    }
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'gold_label': ['Positive']*50 + ['Negative']*40 + ['Neutral']*60
    })
    
    generate_all_visualizations(sample_df, sample_results, output_dir="outputs")

