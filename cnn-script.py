import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from skimage.io import imread
from skimage.transform import resize


np.random.seed(1)

# Define paths
train_dir = '/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/images_train/'
test_dir = '/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/images_test/'
train_csv = '/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/train.csv'
test_csv = '/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/test.csv'
solution_csv = '/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/solution.csv'

# Load CSV files
train_df = pd.read_csv(train_csv)
train_df['ground_truth'] = train_df['ground_truth'].astype(int)
train_df['id'] = train_df['id'].astype(str) + '.png'
test_df = pd.read_csv(test_csv)
test_df['id'] = test_df['id'].astype(str) + '.png'
solution_df = pd.read_csv(solution_csv)
solution_df['id'] = solution_df['id'].astype(str) + '.png'

# Define function to load images
def load_images(directory, df):
    images = []
    for img_name in df['id']:
        img_path = os.path.join(directory, img_name)
        img = imread(img_path)
        img_resized = resize(img, (50, 50, 4))  # Resize to smaller dimension for faster processing
        images.append(img_resized.flatten())  # Flatten the image
    return np.array(images)

# Load and preprocess the images
X_train = load_images(train_dir, train_df)
X_test = load_images(test_dir, test_df)
y_train = train_df['ground_truth'].values

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

# Use the model to make predictions on the validation set
val_predictions = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make probability predictions on the test set
test_probabilities = model.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for class 1

# Add probabilities to the test DataFrame
test_df['predicted_probability'] = test_probabilities

# Compare predictions with the solution provided (test set)
merged_df = pd.merge(test_df, solution_df, on='id')

# Calculate performance metrics
accuracy = accuracy_score(merged_df['ground_truth'], (merged_df['predicted_probability'] > 0.5).astype(int))
precision = precision_score(merged_df['ground_truth'], (merged_df['predicted_probability'] > 0.5).astype(int))
recall = recall_score(merged_df['ground_truth'], (merged_df['predicted_probability'] > 0.5).astype(int))
f1 = f1_score(merged_df['ground_truth'], (merged_df['predicted_probability'] > 0.5).astype(int))

# Compute ROC AUC
roc_auc = roc_auc_score(merged_df['ground_truth'], merged_df['predicted_probability'])

# Print performance metrics
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Save the results to a CSV file
test_df[['id', 'predicted_probability']].to_csv('/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/submission-baseline.csv', index=False)