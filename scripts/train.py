import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
import mlflow
import mlflow.tensorflow
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train fruit classification model')
    parser.add_argument('--dataset-path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--experiment-name', type=str, default='Fruit-Classification', help='MLflow experiment name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Initialize MLflow
    mlflow.set_experiment(args.experiment_name)
    mlflow.tensorflow.autolog()
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("img_size", args.img_size)
        mlflow.log_param("model_name", "MobileNetV2")
        
        # Dataset paths
        dataset_base = args.dataset_path
        train_dir = os.path.join(dataset_base, 'train')
        test_dir = os.path.join(dataset_base, 'test')

        # Verify directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        print(f"Training directory: {train_dir}")
        print(f"Test directory: {test_dir}")

        # Image parameters
        IMG_SIZE = (args.img_size, args.img_size)
        BATCH_SIZE = args.batch_size

        # Enhanced data augmentation to prevent overfitting
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest',
            validation_split=0.2  # Using 20% of training data for validation
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )

        # Test data generator
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        # Get class names and number of classes
        class_names = list(train_generator.class_indices.keys())
        NUM_CLASSES = len(class_names)
        print(f"Found {NUM_CLASSES} classes: {class_names}")

        # Log class information
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("classes", str(class_names))

        # Display sample information
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Test samples: {test_generator.samples}")

        # Load pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(args.img_size, args.img_size, 3)
        )

        # Freeze the base model initially
        base_model.trainable = False

        # Build the model with strong regularization
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.6),  # Increased dropout
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        # Display model summary
        model.summary()

        # Compile the model with only accuracy metric to avoid the error
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Only using accuracy to avoid metric issues
        )

        # Enhanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Train the model
        print("Training the model...")
        history = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Fine-tuning: Unfreeze some layers
        print("Starting fine-tuning...")
        base_model.trainable = True
        # Freeze first 100 layers
        for layer in base_model.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Only accuracy for fine-tuning too
        )

        # Fine-tune the model
        fine_tune_epochs = 20
        total_epochs = len(history.history['loss']) + fine_tune_epochs

        history_fine = model.fit(
            train_generator,
            epochs=total_epochs,
            initial_epoch=len(history.history['loss']),
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Load the best model
        model = keras.models.load_model('best_model.h5')

        # Evaluate the model on test data
        print("\nEvaluating on test data...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        # Predict on test data
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes

        # Classification report
        print("\nClassification Report:")
        report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
        print(classification_report(true_classes, predicted_classes, target_names=class_names))

        # Log classification metrics
        for class_name in class_names:
            mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
            mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])

        # Calculate additional metrics using sklearn
        precision = precision_score(true_classes, predicted_classes, average='weighted')
        recall = recall_score(true_classes, predicted_classes, average='weighted')
        f1 = f1_score(true_classes, predicted_classes, average='weighted')

        mlflow.log_metric("weighted_precision", precision)
        mlflow.log_metric("weighted_recall", recall)
        mlflow.log_metric("weighted_f1", f1)

        print(f"\nAdditional Metrics:")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")

        # Confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(true_classes, predicted_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # Plot training history
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('training_history.png')
        plt.close()

        # Save the final model
        model_path = f"fruits_classification_mobilenetv2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved as '{model_path}'")

        # Log model
        # mlflow.tensorflow.log_model(model=model, artifact_path="model", registered_model_name="FruitClassificationMobileNetV2")
        """model_path = "mlflow_model"
        model.save(model_path)
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=model_path,
            artifact_path="model",
            registered_model_name="FruitClassificationMobileNetV2"
        )"""

        import shutil
        shutil.rmtree(model_path, ignore_errors=True)
        

        # Class-wise accuracy
        print("\nClass-wise Accuracy:")
        class_accuracy = {}
        for i, class_name in enumerate(class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == i)
                class_accuracy[class_name] = class_acc
                mlflow.log_metric(f"accuracy_{class_name}", class_acc)
                print(f"{class_name}: {class_acc:.4f}")

        # Plot class distribution
        class_counts = pd.Series(true_classes).value_counts()
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(class_names)), class_counts.values)
        plt.title('Class Distribution in Test Set', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('class_distribution.png')
        plt.close()

        print("\nTraining completed successfully!")
        print(f"Model saved as: {model_path}")

        # Print final summary
        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()