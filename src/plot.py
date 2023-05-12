import pandas as pd
import matplotlib.pyplot as plt
from model import select_k_best, fit_model, validate_model

def plot_feature_scores(data):
    plt.figure(figsize=(6,10))
    plt.barh('features', 'scores', data=data)
    plt.title('Feature Scoring - ANOVA ')
    plt.xlabel('ANOVA Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('../assets/images/feature_scores.png')