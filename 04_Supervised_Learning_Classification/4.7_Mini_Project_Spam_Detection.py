"""
Spam Detection Mini-Project

This script demonstrates:
- Text preprocessing and feature extraction
- Multiple classification algorithms
- Model evaluation and comparison
- Hyperparameter tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    cross_val_score, learning_curve
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

def download_nltk_data():
    """Download required NLTK data"""
    print("\n=== Downloading NLTK Data ===")
    nltk.download('stopwords')
    nltk.download('punkt')

def load_data():
    """Load and prepare spam dataset"""
    print("\n=== Loading Data ===")
    
    # Load dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only relevant columns
    df.columns = ['label', 'text']
    
    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def preprocess_text(text):
    """Preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def prepare_data(df):
    """Prepare data for classification"""
    print("\n=== Preparing Data ===")
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_pipelines():
    """Create classification pipelines"""
    print("\n=== Creating Pipelines ===")
    
    # Define pipelines
    pipelines = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(random_state=42))
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC(random_state=42, probability=True))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
    }
    
    return pipelines

def tune_hyperparameters(pipeline, X_train, y_train):
    """Tune hyperparameters using grid search"""
    print("\n=== Tuning Hyperparameters ===")
    
    # Define parameter grid
    param_grid = {
        'tfidf__max_features': [1000, 2000, 3000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1, 10] if 'C' in pipeline.named_steps['clf'].get_params() else None
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5,
        scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    print(f"\n=== Plotting Confusion Matrix for {model_name} ===")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve"""
    print(f"\n=== Plotting ROC Curve for {model_name} ===")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, model_name):
    """Plot feature importance"""
    print(f"\n=== Plotting Feature Importance for {model_name} ===")
    
    # Get feature names and importance
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    if hasattr(model.named_steps['clf'], 'coef_'):
        importance = model.named_steps['clf'].coef_[0]
    elif hasattr(model.named_steps['clf'], 'feature_importances_'):
        importance = model.named_steps['clf'].feature_importances_
    else:
        print("Model does not support feature importance visualization")
        return
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature',
                data=feature_importance.head(20))
    plt.title(f'Top 20 Features - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    """Main function for spam detection project"""
    print("=== Spam Detection Mini-Project ===")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Create pipelines
    pipelines = create_pipelines()
    
    # Train and evaluate models
    results = {}
    for name, pipeline in pipelines.items():
        print(f"\n=== Training {name} ===")
        
        # Tune hyperparameters
        best_model, best_params = tune_hyperparameters(pipeline, X_train, y_train)
        print(f"Best parameters: {best_params}")
        
        # Evaluate model
        results[name] = evaluate_model(best_model, X_test, y_test, name)
        
        # Plot evaluation metrics
        plot_confusion_matrix(y_test, results[name]['y_pred'], name)
        plot_roc_curve(y_test, results[name]['y_prob'], name)
        
        # Plot feature importance if applicable
        plot_feature_importance(best_model, name)
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1 Score': [r['f1'] for r in results.values()]
    })
    print("\nModel Comparison:")
    print(comparison)
    
    print("\nSpam detection project completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 