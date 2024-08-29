#%%

# Imports
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from Bio.Seq import Seq

# Print TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Get the current working directory
current_working_directory = os.getcwd()
# Print the current working directory
print("Current Working Directory:", current_working_directory)

#%%

# Load the CSV file
file_path = 'subset_data.csv'
subset_data = pd.read_csv(file_path)

# Display the top few lines
print(subset_data.head())

# Assuming subset_data is already defined
random.seed(2024)
# Randomly sample 100,000 rows from subset data
subset_data = subset_data.sample(n=100000, random_state=42)

# Display the top 5 rows
print(subset_data.head(5))

# Function to label encode sequences
def label_encode_sequence(sequence):
    label_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    return [label_dict[char] for char in sequence]

# Label encode each sequence in the dataset
encoded_sequences = subset_data['sequences'].apply(label_encode_sequence)

# Pad sequences to the same length (if necessary)
max_length = max(encoded_sequences.apply(len))
encoded_sequences_padded = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in encoded_sequences])

# Encode the target variable (SBS signature)
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(subset_data['Signature'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_sequences_padded, target, test_size=0.2, random_state=42)

# Convert input data to the appropriate data types
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('int32')
y_test = np.asarray(y_test).astype('int32')

# Define the input shape based on the sequence length
input_shape = (X_train.shape[1], 1)

# %%
from tensorflow.keras.layers import BatchNormalization

# Set the random seed for reproducibility
random.seed(2024)
np.random.seed(2024)
tf.random.set_seed(2024)

# Build the model
model = Sequential()
model.add(Reshape(input_shape, input_shape=(input_shape[0],)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5)) # Dropout layer to prevent overfitting
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Another Dropout layer
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Multi-class classification

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use for multi-class classification
              metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=350, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%

# Calculate SHAP Value
import shap
import xgboost

# Train an XGBoost model
xgb_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Convert feature names to strings
feature_names = list(map(str, range(input_shape[0])))

# Explain the model's predictions using SHAP
explainer = shap.Explainer(xgb_model, feature_names=feature_names)
shap_values = explainer(X_test)

# Visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns

# Predict the classes for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
conf_matrix_display.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC for each class
plt.figure(figsize=(12, 8))
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

