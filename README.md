
# Automatic Identification of Internal Waves
## AI Competition hosted on Kaggle and created by the [AIR Centre](aircentre.org)
### Using a Multi-Layer Perceptron (MLP) classifier (feedforward artificial neural network) for image classification
+ The repo constains a script and notebooks created for the competition mentioned above;
+ The script was created to test the competition data;
+ The performance results provide a baseline for the competition.

This code implements an image classification pipeline using a Multi-Layer Perceptron (MLP) classifier. It loads and preprocesses image data from specified directories, normalizes the data, and splits it into training, validation, and test sets. The MLP model is trained on the preprocessed data and evaluated using various performance metrics including accuracy, precision, recall, F1 score, and ROC AUC. The code also generates predictions for a test set and saves these predictions to a CSV file for submission. While the model achieves low performance with a validation accuracy of 0.5815 and ROC AUC of 0.6041, theis was the very first approach and teh goal was simple to test the dataset and files created for the competition while creating a baseline. There's potential for improvement by using more advanced techniques like Convolutional Neural Networks, which are better suited for image classification tasks.

#### Addition
Added a CNN solver that:
+ Uses TensorFlow and Keras to build a CNN instead of sklearn's MLPClassifier.
+ The image preprocessing now keeps the 2D structure of the images (64x64x4) instead of flattening them.
+ The CNN architecture consists of three convolutional layers with max pooling, followed by a flatten layer and two dense layers.
+ The model is compiled with binary crossentropy loss and Adam optimizer.
+ The training process uses model.fit() instead of model.fit_transform().
Predictions are made using model.predict() instead of model.predict_proba().
+ No optimization was made.

##### Results of the CNN
+ Test Accuracy: 0.5072
+ Precision: 0.5072
+ Recall: 1.0000
+ F1 Score: 0.6730
+ ROC AUC: 0.5000


### Contents
+ Script:
  + Definition of the environment to run the script locally.
  + Python script with the implementation of Multi-Layer Perceptron (MLP) classifier (feedforward artificial neural network) for image classification.
  + Pthton script with the implementation of a Convolutional Neural Network whose architecture consists of three convolutional layers with max pooling, followed by a flatten layer and two dense layers.

+ Notebooks: The same original MLP script implemented in two Jupyter notebooks:
  + One runs locally
  + One runs on Kaggle.

...
