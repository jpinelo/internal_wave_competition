
# Automatic Identification of Internal Waves
## AI Competition hosted on Kaggle and created by the [AIR Centre](aircentre.org)
### Using a Multi-Layer Perceptron (MLP) classifier (feedforward artificial neural network) fro image classification
+ The repo constains a script and notebooks created for the competition mentioned above;
+ The script was created to test the competition data;
+ The performance results provide a baseline for the competition.

This code implements an image classification pipeline using a Multi-Layer Perceptron (MLP) classifier. It loads and preprocesses image data from specified directories, normalizes the data, and splits it into training, validation, and test sets. The MLP model is trained on the preprocessed data and evaluated using various performance metrics including accuracy, precision, recall, F1 score, and ROC AUC. The code also generates predictions for a test set and saves these predictions to a CSV file for submission. While the model achieves low performance with a validation accuracy of 0.5815 and ROC AUC of 0.6041, theis was the very first approach and teh goal was simple to test the dataset and files created for the competition while creating a baseline. There's potential for improvement by using more advanced techniques like Convolutional Neural Networks, which are better suited for image classification tasks.

### Contents
+ Script:
  + Python script with the implementation of a basic CNN.
  + Definition of the environment to run the script locally.
+ Notebooks: The same script implemented in two Jupyter notebooks:
  + One runs locally
  + One runs on Kaggle.

...
