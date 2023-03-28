

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x_train = "x_train.csv"
y_train = "y_train.csv"
x_test = "x_test.csv"
y_test = "y_test.csv"

y_train_dataframe = pd.read_csv(y_train, header=0)
x_train_dataframe = pd.read_csv(x_train, header=0)
x_test_dataframe = pd.read_csv(x_test, header=0)
y_test_dataframe = pd.read_csv(y_test, header=0)

print(x_train_dataframe.head())

print(y_train_dataframe.head())

number_of_spam = 0
number_of_ham = 0

for i in y_train_dataframe["Prediction"]:
    if i == 1:
        number_of_spam += 1
    else:
        number_of_ham += 1

print(f"Number of spam emails is {number_of_spam}.")
print(f"Number of ham emails is {number_of_ham}.")
print(F" Percentage of spam emails in the dataset is {100 * (number_of_spam / (number_of_spam + number_of_ham))}.")


def create_dictionary_of_word_occurences(x_dataframe, y_dataframe):
    spam_words_dict = dict()
    ham_words_dict = dict()

    column_names = list(x_dataframe.columns)

    for word in column_names:
        spam_words_dict[word] = 0
        ham_words_dict[word] = 0

    for word in x_dataframe:
        for i, num in enumerate(x_dataframe[word]):

            if y_dataframe["Prediction"][i] == 1:
                spam_words_dict[word] += num
            else:
                ham_words_dict[word] += num

    return spam_words_dict, ham_words_dict


def train(x_dataframe, y_dataframe):
    spam_words_dict, ham_words_dict = create_dictionary_of_word_occurences(x_train_dataframe, y_train_dataframe)
    print(spam_words_dict["the"])
    num_spam = y_train_dataframe['Prediction'].value_counts()[1]
    num_ham = y_train_dataframe['Prediction'].value_counts()[0]

    # priors
    p_spam = num_spam / (num_spam + num_ham)
    p_ham = num_ham / (num_spam + num_ham)

    tot_spam_words = sum(spam_words_dict.values())
    tot_ham_words = sum(ham_words_dict.values())

    # likelihoods

    for key in spam_words_dict:
        spam_words_dict[key] = spam_words_dict[key] / tot_spam_words
    for key in ham_words_dict:
        ham_words_dict[key] = ham_words_dict[key] / tot_ham_words

    return p_spam, p_ham, spam_words_dict, ham_words_dict


def test(p_spam, p_ham, spam_words_likely_dict, ham_words_likely_dict, x_dataframe_test, y_dataframe_test):
    column_names = list(x_dataframe_test.columns)
    results = np.zeros(y_test_dataframe.shape)

    for index, row in x_dataframe_test.iterrows():
        # print(f"Mail {index + 1}:")
        po_spam = np.log(p_spam)
        po_ham = np.log(p_ham)

        for ind_words, num_of_words in enumerate(row):
            for i in range(num_of_words):
                po_spam += np.log(spam_words_likely_dict[column_names[ind_words]])
                po_ham += np.log(ham_words_likely_dict[column_names[ind_words]])

        results[index] = (po_spam > po_ham)

    return results


a, b, c, d = train(y_train_dataframe, y_train_dataframe)

results = test(a, b, c, d, x_test_dataframe, y_test_dataframe)

print(results)


def calculate_accuracy(true_array, pred_array):
    num_correct = np.sum(true_array == pred_array)

    num_total = true_array.shape[0]

    accuracy = num_correct / num_total

    return accuracy


print(f"The accuracy of predictions in  Multinomial naive bayes :  {calculate_accuracy(results, y_test_dataframe.to_numpy())}")

def train_dl(x_dataframe, y_dataframe):
    spam_words_dict, ham_words_dict = create_dictionary_of_word_occurences(x_dataframe, y_dataframe)

    spam_words_dict["the"]
    num_spam = y_train_dataframe['Prediction'].value_counts()[1]
    num_ham = y_train_dataframe['Prediction'].value_counts()[0]

    # priors
    p_spam = num_spam / (num_spam + num_ham)
    p_ham = num_ham / (num_spam + num_ham)

    tot_spam_words = sum(spam_words_dict.values())
    tot_ham_words = sum(ham_words_dict.values())

    # likelihoods
    alpha = 5  # Dirichlet prior of 5
    for key in spam_words_dict:
        spam_words_dict[key] = (spam_words_dict[key] + alpha) / (tot_spam_words + alpha * len(spam_words_dict))
    for key in ham_words_dict:
        ham_words_dict[key] = (ham_words_dict[key] + alpha) / (tot_ham_words + alpha * len(spam_words_dict))

    return p_spam, p_ham, spam_words_dict, ham_words_dict


p_spam_D, p_ham_D, spam_words_dict_D, ham_words_dict_D = train_dl(x_train_dataframe, y_train_dataframe)

results_d = test(p_spam_D, p_ham_D, spam_words_dict_D, ham_words_dict_D, x_test_dataframe, y_test_dataframe)

print(f"The accuracy of predictions in Multinomial naive bayes with dirichlet prior :  {calculate_accuracy(results_d, y_test_dataframe.to_numpy())}")


def create_dictionary_of_word_occurences_bnb(x_dataframe, y_dataframe):
    spam_words_dict = dict()
    ham_words_dict = dict()
    words_dict = dict()

    column_names = list(x_dataframe.columns)

    for word in column_names:
        words_dict[word] = 0
        spam_words_dict[word] = 0
        ham_words_dict[word] = 0

    for word in x_dataframe:
        for i, num in enumerate(x_dataframe[word]):

            if y_dataframe["Prediction"][i] == 1:
                spam_words_dict[word] += num > 0
            else:
                ham_words_dict[word] += num > 0

            words_dict[word] += 1
    return words_dict, spam_words_dict, ham_words_dict


def train_bnb(x_dataframe, y_dataframe):
    words_dict, spam_words_dict, ham_words_dict = create_dictionary_of_word_occurences_bnb(x_train_dataframe,
                                                                                           y_train_dataframe)
    y_array = y_train_dataframe.to_numpy()

    num_spam = y_train_dataframe['Prediction'].value_counts()[1]
    num_ham = y_train_dataframe['Prediction'].value_counts()[0]

    # priors
    p_spam = num_spam / (num_spam + num_ham)
    p_ham = num_ham / (num_spam + num_ham)

    tot_spam_words = sum(spam_words_dict.values())
    tot_ham_words = sum(ham_words_dict.values())

    # likelihoods

    for key in spam_words_dict:
        spam_words_dict[key] = spam_words_dict[key] / num_spam
    for key in ham_words_dict:
        ham_words_dict[key] = ham_words_dict[key] / num_spam

    return p_spam, p_ham, spam_words_dict, ham_words_dict


p_spam_bnb, p_ham_bnb, spam_words_dict_bnb, ham_words_dict_bnb = train_bnb(y_train_dataframe, y_train_dataframe)

results_bnb = test(p_spam_bnb, p_ham_bnb, spam_words_dict_bnb, ham_words_dict_bnb, x_test_dataframe, y_test_dataframe)

print(f"The accuracy of predictions in  Bernoulli naive bayes :  {calculate_accuracy(results_bnb, y_test_dataframe.to_numpy())}")


def plot_confusion_matrix(y_true, y_pred):
    # Create confusion matrix as a NumPy array
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = np.zeros((len(labels), len(labels)))
    for i in range(len(y_true)):
        true_label_index = np.where(labels == y_true[i])[0][0]
        pred_label_index = np.where(labels == y_pred[i])[0][0]
        cm[true_label_index][pred_label_index] += 1

    # Create Seaborn heatmap of confusion matrix
    sns.set()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


plot_confusion_matrix(results_bnb, y_test_dataframe.to_numpy())

plot_confusion_matrix(results_d, y_test_dataframe.to_numpy())

plot_confusion_matrix(results, y_test_dataframe.to_numpy())

input("Press Enter to exit...")
