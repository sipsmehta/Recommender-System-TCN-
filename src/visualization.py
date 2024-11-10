import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy if available
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_prediction_analysis(y_true, y_pred):
    """
    Create visualizations for prediction analysis
    """
    plt.figure(figsize=(15, 5))
    
    # Scatter plot of predicted vs actual values
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Distribution of prediction errors
    plt.subplot(1, 3, 2)
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    
    # Box plot of predictions by rating category
    plt.subplot(1, 3, 3)
    rating_categories = np.round(y_true)
    plt.boxplot([y_pred[y_true == cat] for cat in sorted(np.unique(rating_categories))])
    plt.title('Prediction Distribution by Rating Category')
    plt.xlabel('Actual Rating Category')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.close()