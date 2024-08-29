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
# Randomly sample rows from subset data
subset_data = subset_data.sample(n=100000, random_state=42)

# Display the top 10 rows
print(subset_data.head(5))

#%%
# Function to generate canonical k-mers
def generate_canonical_kmers(sequence, k):
    n = len(sequence)
    
    # Check if sequence length is less than k
    if ((n - 3) / 2) < k:
        raise ValueError("Sequence length is less than k. Cannot generate k-mers.")
    
    middle = n // 2  # Assuming mutation site is in the middle
    
    left_flank = sequence[:middle - 1]
    right_flank = sequence[middle + 2:]
    
    def get_min_kmer(kmer):
        rev_comp = str(Seq(kmer).reverse_complement())
        return min(kmer, rev_comp)
    
    kmers_left = [get_min_kmer(left_flank[i:i + k]) for i in range(len(left_flank) - k + 1)]
    kmers_right = [get_min_kmer(right_flank[i:i + k]) for i in range(len(right_flank) - k + 1)]
    
    return kmers_left + kmers_right

# Set k value for k-mers
k = 5

# Generate canonical k-mers for each sequence in the dataset
kmer_list = subset_data['sequences'].apply(generate_canonical_kmers, k=k)
print(kmer_list.head(5))

# Get all unique canonical k-mers to create feature space
all_kmers = sorted(set(sum(kmer_list, [])))

# Function to encode k-mers
def encode_kmers(kmers, all_kmers):
    kmer_table = {kmer: 0 for kmer in all_kmers}
    for kmer in kmers:
        kmer_table[kmer] += 1
    return list(kmer_table.values())

# Encode sequences as k-mer vectors
encoded_kmers = pd.DataFrame([encode_kmers(kmers, all_kmers) for kmers in kmer_list], columns=all_kmers)

# Function to extract the flanking nucleotides and the alt nucleotide
def extract_features(sequence, ref, alt):
    n = len(sequence)
    middle = n // 2  # Assuming mutation site is in the middle
    left_flank = sequence[middle - 1]
    right_flank = sequence[middle + 1]
    return [left_flank, right_flank, ref, alt]

#%%
# def extract_features(sequence, ref):
#     n = len(sequence)
#     middle = n // 2  # Assuming mutation site is in the middle
#     left_flank = sequence[middle - 1]
#     right_flank = sequence[middle + 1]
#     return [left_flank, right_flank]


# %%
# Apply the function to extract the features for each row in the dataset
flanking_nucleotides = subset_data.apply(lambda row: extract_features(row['sequences'], row['ref'], row['alt']), axis=1)
# flanking_nucleotides = subset_data.apply(lambda row: extract_features(row['sequences'], row['ref']), axis=1)

# Convert the extracted features to a dataframe
flanking_nucleotides_df = pd.DataFrame(flanking_nucleotides.tolist(), columns=['left_nucleotide', 'right_nucleotide', 'alt'])
# flanking_nucleotides_df = pd.DataFrame(flanking_nucleotides.tolist(), columns=['left_nucleotide', 'right_nucleotide'])

# Convert flanking nucleotides to one-hot encoding
flanking_nucleotides_onehot = pd.get_dummies(flanking_nucleotides_df)

# Combine encoded k-mers and one-hot encoded flanking nucleotides
combined_features = pd.concat([encoded_kmers, flanking_nucleotides_onehot], axis=1)

# Encode the target variable (SBS signature)
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(subset_data['Signature'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)

# Convert input data to the appropriate data types
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('int32')
y_test = np.asarray(y_test).astype('int32')

# Define the input shape based on the combined features
input_shape = X_train.shape[1]

# %%
from tensorflow.keras.layers import BatchNormalization

# Set the random seed for reproducibility
random.seed(2024)
np.random.seed(2024)
tf.random.set_seed(2024)

# Build the model
model = Sequential()
model.add(Reshape((input_shape, 1), input_shape=(input_shape,)))
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

# Explain the model's predictions using SHAP
explainer = shap.Explainer(xgb_model, feature_names=combined_features.columns)
shap_values = explainer(X_test)

# Visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])



# %%
