import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to build, compile, and train the model with given hyperparameters
def train_model(learning_rate=0.001, batch_size=32, epochs=10):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=2)
    return model, history

# Baseline training: learning_rate=0.001, batch_size=32, epochs=10
model, history = train_model(learning_rate=0.001, batch_size=32, epochs=10)

# Plot training history: Accuracy and Loss over epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Experiment with hyperparameters: learning rate and batch size
experiments = [
    {'learning_rate': 0.0001, 'batch_size': 32},
    {'learning_rate': 0.001,  'batch_size': 32},
    {'learning_rate': 0.01,   'batch_size': 32},
    {'learning_rate': 0.001,  'batch_size': 64},
    {'learning_rate': 0.001,  'batch_size': 128},
]

results = {}

for exp in experiments:
    print(f"\nTraining with learning_rate={exp['learning_rate']} and batch_size={exp['batch_size']}")
    _, exp_history = train_model(learning_rate=exp['learning_rate'],
                                 batch_size=exp['batch_size'],
                                 epochs=5)  # Fewer epochs for quick experimentation
    final_val_acc = exp_history.history['val_accuracy'][-1]
    results[(exp['learning_rate'], exp['batch_size'])] = final_val_acc
    print(f"Final Validation Accuracy: {final_val_acc}")

print("\nSummary of Experiments (Learning Rate, Batch Size):")
for params, acc in results.items():
    print(f"LR: {params[0]}, Batch Size: {params[1]} -> Validation Accuracy: {acc}")
