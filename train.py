# train.py
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

TRAIN_CSV = "sign_mnist_train.csv"
TEST_CSV = "sign_mnist_test.csv"
MODEL_FILE = "gesture_model.h5"

def create_cnn_model(input_shape=(28, 28, 1), num_classes=26):
    """Create a CNN model with residual connections and batch normalization"""
    inputs = layers.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual block 1
    shortcut = x
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    # Second conv block with downsample path for residual
    # Save shortcut before pooling for the next residual connection
    shortcut = layers.Conv2D(128, (1, 1))(x)
    shortcut = layers.BatchNormalization()(shortcut)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)  # Match the pooling in main path
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual block 2
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Now shapes should match
    x = layers.Activation('relu')(x)
    
    # Global average pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def preprocess_data(df):
    """Preprocess the data from the MNIST format"""
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Reshape features to 28x28 images
    X = X.reshape(-1, 28, 28, 1)
    X = X.astype('float32') / 255.0
    
    # Convert labels to categorical (subtract 1 to make 0-based)
    y = y - 1  # Convert 1-26 to 0-25
    y = tf.keras.utils.to_categorical(y, num_classes=26)
    return X, y

def data_augmentation():
    """Create a data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

# Load and preprocess data
print("Loading training data...")
train_df = pd.read_csv(TRAIN_CSV)
print("Loading test data...")
test_df = pd.read_csv(TEST_CSV)

# Split training data into train and validation
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Preprocess all datasets
X_train, y_train = preprocess_data(train_df)
X_val, y_val = preprocess_data(val_df)
X_test, y_test = preprocess_data(test_df)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Create and compile model
model = create_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create data augmentation pipeline
augmentation = data_augmentation()

# Training callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILE,
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

# Train the model
print("Training model...")
history = model.fit(
    augmentation(X_train),
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved to training_history.pkl")
