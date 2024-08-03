import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)
tf.random.set_seed(1)

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
        img_resized = resize(img, (64, 64, 4))  # Resize to 64x64x4
        images.append(img_resized)
    return np.array(images)

# Load and preprocess the images
X_train = load_images(train_dir, train_df)
X_test = load_images(test_dir, test_df)
y_train = train_df['ground_truth'].values

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Normalize the data
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                    validation_data=(X_val, y_val))

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make probability predictions on the test set
test_probabilities = model.predict(X_test).flatten()

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

# Plot Precision vs Recall
precision_vals, recall_vals, _ = precision_recall_curve(merged_df['ground_truth'], merged_df['predicted_probability'])
plt.figure()
plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall')
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(merged_df['ground_truth'], (merged_df['predicted_probability'] > 0.5).astype(int))
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Save the results to a CSV file
#test_df['id'] = test_df['id'].str.replace('.png', '')
#test_df[['id', 'predicted_probability']].to_csv('/Users/jpinelo/Dropbox/JP_Lab/AIRCentre/2-Projects/54-Internal_waves/Data/submission-cnn.csv', index=False)