"""
Digit Recognition using CNN

This script demonstrates:
- Loading and preprocessing MNIST dataset
- Building a CNN architecture
- Training and evaluating the model
- Visualizing results and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_data():
    """Load and preprocess MNIST dataset"""
    print("\n=== Loading MNIST Dataset ===")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Reshape data for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def plot_sample_images(x_train, y_train):
    """Plot sample images from the dataset"""
    print("\n=== Plotting Sample Images ===")
    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].squeeze(), cmap='gray')
        plt.xlabel(np.argmax(y_train[i]))
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def build_cnn_model():
    """Build CNN model architecture"""
    print("\n=== Building CNN Model ===")
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the CNN model"""
    print("\n=== Training Model ===")
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=callbacks,
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

def evaluate_model(model, x_test, y_test):
    """Evaluate the model"""
    print("\n=== Evaluating Model ===")
    
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def visualize_predictions(model, x_test, y_test):
    """Visualize model predictions"""
    print("\n=== Visualizing Predictions ===")
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Find some correct and incorrect predictions
    correct_indices = np.where(y_pred_classes == y_true)[0]
    incorrect_indices = np.where(y_pred_classes != y_true)[0]
    
    # Plot correct predictions
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[correct_indices[i]].squeeze(), cmap='gray')
        plt.xlabel(f"True: {y_true[correct_indices[i]]}\nPred: {y_pred_classes[correct_indices[i]]}")
    plt.tight_layout()
    plt.savefig('correct_predictions.png')
    plt.close()
    
    # Plot incorrect predictions
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(incorrect_indices))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[incorrect_indices[i]].squeeze(), cmap='gray')
        plt.xlabel(f"True: {y_true[incorrect_indices[i]]}\nPred: {y_pred_classes[incorrect_indices[i]]}")
    plt.tight_layout()
    plt.savefig('incorrect_predictions.png')
    plt.close()

def main():
    """Main function"""
    print("=== Digit Recognition using CNN ===")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Plot sample images
    plot_sample_images(x_train, y_train)
    
    # Build and train model
    model = build_cnn_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Visualize predictions
    visualize_predictions(model, x_test, y_test)
    
    print("\nAll digit recognition examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 