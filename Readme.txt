Spam Detection Project

This project is a spam detection system using Naïve Bayes, Dirichlet Prior, and Bernoulli Naïve Bayes classifiers. The system trains on a dataset of spam and ham emails and predicts whether a given email is spam or ham.

Prerequisites

To run the project, you will need the following dependencies:

Python 3.7+
pandas
NumPy
seaborn
matplotlib
You can install these dependencies using pip:

pip install pandas numpy seaborn matplotlib

Usage

Clone the repository and navigate to the project folder:
git clone <repository-url>
cd <project-folder>

Place your dataset files (x_train.csv, y_train.csv, x_test.csv, y_test.csv) in the project folder or update the file paths in the code accordingly.

Run the script using the following command:

python spam_detection.py

This command will execute the script with default parameters. The script will load the datasets, train the classifiers, perform predictions, calculate accuracies, and display confusion matrices.

Parameters

The script uses default values for various parameters. You can change these values directly in the script or pass them as arguments to the functions.

alpha: The Dirichlet prior parameter. Default value is 5.
You can modify the alpha value in the train_dl() function or pass a different value when calling the function.

Output

The script will display the accuracy of each classifier and the confusion matrices in the terminal.

Accuracy: The percentage of correctly classified emails.
Confusion Matrix: A matrix that shows the number of true positive, true negative, false positive, and false negative predictions for each classifier.
You can read the output directly from the terminal