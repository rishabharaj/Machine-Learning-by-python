"""
Building a Simple Neural Network

This script demonstrates:
- Building a feedforward neural network
- Training on a simple dataset
- Model evaluation and visualization
- Understanding neural network behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

def generate_data():
    """Generate a simple classification dataset"""
    print("\n=== Generating Dataset ===")
    
    # Generate linearly separable data
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def plot_data(X, y, title):
    """Plot the dataset"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def build_model():
    """Build a simple neural network"""
    print("\n=== Building Neural Network ===")
    
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the neural network"""
    print("\n=== Training Model ===")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training history"""
    print("\n=== Plotting Training History ===")
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary"""
    print("\n=== Plotting Decision Boundary ===")
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.savefig('decision_boundary.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    print("\n=== Evaluating Model ===")
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    """Main function"""
    print("=== Building and Training a Simple Neural Network ===")
    
    # Generate and visualize data
    X_train, X_test, y_train, y_test = generate_data()
    plot_data(X_train, y_train, 'Training Data')
    plot_data(X_test, y_test, 'Test Data')
    
    # Build and train model
    model = build_model()
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot decision boundary
    plot_decision_boundary(model, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    print("\nAll neural network examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 