import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import select_k_best, fit_model, validate_model

sns.set_theme(style='whitegrid')

def plot_feature_scores(data):
    plt.clf()
    plt.figure(figsize=(6,10))
    sns.barplot(x='scores', y='features', data=data, palette=sns.color_palette('viridis', n_colors=45))
    plt.title('Feature Scoring - ANOVA')
    plt.xlabel('Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('../assets/images/feature_scores.png')

def plot_heatmap(data):
    plt.clf()
    plt.figure(figsize=(8,6))
    sns.heatmap(data)
    plt.title('Correlation Matrix - Pearson')
    plt.tight_layout()
    plt.savefig('../assets/images/correlation_matrix_test.png')

def plot_bar(data):
    plt.clf()
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=(20, 6))
    palette=sns.color_palette('muted', n_colors=len(data.index))
    sns.barplot(data=data, x=data.index, y='Loss', palette=palette, ax=ax1)
    ax1.title.set_text('Test Data Prediction - Loss')
    sns.barplot(data=data, x=data.index, y='Accuracy', palette=palette, ax=ax2)
    ax2.title.set_text('Test Data Prediction - Accuracy')
    plt.tight_layout()
    plt.savefig('../assets/images/box_plot_loss.png')

def plot_roc_curve(data):
    plt.clf()
    plt.figure(figsize=(8,6))
    palette=sns.color_palette('muted', n_colors=len(data.index))
    i=0
    for algo in data.index:
        sns.lineplot(data=data.loc[algo], x='False Positive', y='True Positive', color=palette[i], label=algo)
        i+=1
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.tight_layout()
    plt.savefig('../assets/images/roc_curve.png')