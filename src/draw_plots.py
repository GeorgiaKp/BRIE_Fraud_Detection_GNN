import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse

def plot_combined_metrics(saved_metrics_dir, validation_interval):
    # Load the saved metrics
    val_recalls = np.load(os.path.join(saved_metrics_dir, 'valid_recall.npy'))
    val_f1s = np.load(os.path.join(saved_metrics_dir, 'valid_f1_mac.npy'))
    val_aucs = np.load(os.path.join(saved_metrics_dir, 'valid_auc.npy'))

    print(val_recalls)

    epochs = np.arange(0, len(val_recalls) * validation_interval, validation_interval)
    plt.figure(figsize=(6, 5))

    plt.plot(epochs, [recall * 100 for recall in val_recalls], 'b--', label='Recall', alpha=0.7)
    plt.scatter(epochs, [recall * 100 for recall in val_recalls], color='b', marker='^', s=30, alpha=0.7)

    plt.plot(epochs, [f1 * 100 for f1 in val_f1s], 'g--', label='F1-macro', alpha=0.6)
    plt.scatter(epochs, [f1 * 100 for f1 in val_f1s], color='g', marker='*', s=30, alpha=0.7)

    plt.plot(epochs, [auc * 100 for auc in val_aucs], 'y--', label='AUC', alpha=0.6)
    plt.scatter(epochs, [auc * 100 for auc in val_aucs], color='y', marker=',', s=30, alpha=0.7)

    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Score (%)', fontsize=16)
    # plt.title('Validation Recall, F1-macro, AUC over Epochs', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)  # Set y-axis limits from 0 to 105
    plt.yticks(np.arange(0, 110, 10), fontsize=14)  # Set y-ticks every 10 units
    plt.xticks(fontsize=14)
    plt.savefig(os.path.join(saved_metrics_dir, 'validation_metrics.png'))
    plt.show()

def plot_auc_roc_curve(saved_metrics_dir):
    # Load true labels and predicted scores for the test set
    labels = np.load(os.path.join(saved_metrics_dir, 'test_true_labels.npy'))
    preds = np.load(os.path.join(saved_metrics_dir, 'test_prob_scores.npy'))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)  
    plt.title('Receiver Operating Characteristic', fontsize=20)  
    plt.legend(loc="lower right", fontsize=15)  
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)  
    plt.savefig(os.path.join(saved_metrics_dir, 'auc_roc_curve.png'))
    plt.show()

def plot_confusion_matrix(saved_metrics_dir):
    labels = np.load(os.path.join(saved_metrics_dir, 'test_true_labels.npy'))
    preds = np.load(os.path.join(saved_metrics_dir, 'test_preds.npy'))
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=True, yticklabels=True, 
                annot_kws={"size": 20})  
    plt.title('Confusion Matrix', fontsize=20)  
    plt.xlabel('Predicted label', fontsize=18) 
    plt.ylabel('True label', fontsize=18)  
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)  
    plt.savefig(os.path.join(saved_metrics_dir, 'confusion_matrix.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot combined metrics.')
    parser.add_argument('saved_metrics_dir', type=str, help='Directory where the saved metrics are stored')
    parser.add_argument('validation_interval', type=int, help='Validation interval')

    args = parser.parse_args()
    plot_combined_metrics(args.saved_metrics_dir, args.validation_interval)
    plot_auc_roc_curve(args.saved_metrics_dir)
    plot_confusion_matrix(args.saved_metrics_dir)