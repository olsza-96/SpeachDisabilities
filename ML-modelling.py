#Learning various methods for data processing

import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors.classification import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import arff #from liac-arff package
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM

def read_input_file(directory):
    # read CSV file
    data = pd.read_csv(directory, skiprows=1, index_col='id', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

    return data

def data_exploration(data):
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 300)

    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

    corr_matrix = data.corr().abs()

    f, ax = plt.subplots()#.figsize=(9, 8))
    sns.heatmap(corr_matrix, ax=ax, cmap="YlGnBu", linewidths=0.1)
    plt.show()

    #creates upper diagonal of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    datasets_corr = []
    corr_bins= np.arange(0.5, 0.95, 0.05)

    for corr in corr_bins:
        #create mask for columns with high correlation
        to_drop = [column for column in upper.columns if any(upper[column] > corr)]
        temp_dataset = data.drop(to_drop, axis=1)
        datasets_corr.append(data.drop(to_drop, axis=1))
        temp_dataset.to_csv('corr_{}.csv'.format(corr))

    print(datasets_corr)

    return datasets_corr

def data_normalization(data):

    dataset_normalized = []
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

    for element in data:
        a = scaler.fit(element).transform(element)
        dataframed = pd.DataFrame(a, columns=element.columns)
        dataset_normalized.append(dataframed)

    print(len(dataset_normalized))

    return dataset_normalized

def data_balancing(data_list):

    for dataset in data_list:
        #balancing for each treshold
        data_sampling(dataset)
        break

def data_sampling(data):

    target_count = data['class'].value_counts()
    print("Target count " + str(target_count))

    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)
    plt.xticks(target_count.index)
    plt.ylabel("Count")
    #plt.show()

    min_class = target_count.idxmin(axis = 1)
    print("Min class " + (str(min_class)))
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[min_class])
    print('Majority class:', target_count[1 - min_class])
    print('Proportion:', round(target_count[min_class] / target_count[1 - min_class], 2), ': 1')

    #now to split datasets into two
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}
    print("Org values " + str(values['Original']))

    #divide dataset on class 1 and 0
    df_class_min = data[data["class"] == min_class]
    df_class_max = data[data['class'] != min_class]

    #doing undersampling
    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    #doing oversampling
    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]


    #doing SMOTE sampling
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy = 'minority', random_state = RANDOM_STATE)
    y = data['class'].values
    X = data.values

    smote_x, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

    print("Result of sampling " +str(values))

    plt.figure()
    multiple_bar_chart(plt.gca(),
                       [target_count.index[ind_min_class], target_count.index[1 - ind_min_class]],
                       values, 'Target', 'frequency', 'Class balance')
    plt.ylabel("Number of records after sampling")
    plt.legend()
    plt.show()

    #KNN_method(X, y)
    NaiveBayes(X,y)
    #DecisionTree(X,y)
    #RandomForest(X, y)


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                       percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    key_max = max(yvalues.keys(), key=(lambda k: yvalues[k]))
    key_min = min(yvalues.keys(), key=(lambda k: yvalues[k]))

    ax.set_ylim(0.8*yvalues[key_min], 1.1*yvalues[key_max])

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox = True, shadow = True)

def KNN_method(X,y):
    skf = StratifiedKFold(n_splits=4, random_state=42)
    skf.get_n_splits(X,y)

    for train_index, test_index in skf.split(X,y):
        print("Train:", train_index, "Validation:", test_index)
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = y[train_index], y[test_index]

        #here starts KNN
        #how many neighbours want to use in the KNC
        kvalues = [1, 3, 5, 7, 9, 11, 13, 15, 19, 24, 30, 40, 50, 60, 70, 90]
        dist = ['manhattan', 'euclidean', 'chebyshev']
        results = {}
        for element in dist:
            accuracy_results = []
            for k in kvalues:
                knn = KNeighborsClassifier(n_neighbors = k, metric = element)
                knn.fit(trainX, trainY)
                predictedY = knn.predict(testX)
                accuracy_results.append(accuracy_score(testY, predictedY))
            results[element] = accuracy_results
        print("Results of model preparation for: " +str(results))

        plt.figure()
        multiple_line_chart(plt.gca(), kvalues, results, 'KNN variants', 'n', 'accuracy', percentage=True)
        plt.show()

def NaiveBayes(X,y):

    trainX, testX, trainY, testY = train_test_split(X, y, train_size=0.7, stratify=y)

    """
    ta czesc kodu robi estymator tylko dla Gaussa"""
    labels = pd.unique(y)
    gaussian = GaussianNB()
    model = gaussian.fit(trainX, trainY)
    predictedY = gaussian.predict(testX)

    confusion_mtx = confusion_matrix(testY, predictedY, labels)

    plot_confusion_matrix(testY, predictedY, labels)

    plt.figure()
    plot_roc_chart(plt.gca(), {'GaussianNB': model}, testX, testY, 'class')
    plt.show()


    # estimators checking for various methods
    """estimators = {'GaussianNB': GaussianNB(),
                  'MultinomialNB': MultinomialNB(),
                  'BernoulliNB': BernoulliNB()}

    estimator_type = []
    accuracy_result = []

    for estimator in estimators:
        estimator_type.append(estimator)
        estimators[estimator].fit(trainX, trainY)
        predictedY = estimators[estimator].predict(testX)
        accuracy_result.append(accuracy_score(testY, predictedY))

    #plt.figure()
    #bar_chart(plt.gca(), estimator_type, accuracy_result, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
    #plt.show()"""



def DecisionTree(X,y):
    labels = pd.unique(y)

    trainX, testX, trainY, testY = train_test_split(X, y, train_size=0.7, stratify=y)

    #min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
    min_samples_leaf = [1, 2, 3]
    max_depths = [1, 2, 5]
    criteria = ['entropy', 'gini']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                tree.fit(trainX, trainY)
                predictedY = tree.predict(testX)
                yvalues.append(accuracy_score(testY, predictedY))
            values[d] = yvalues
        multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                                 'nr estimators',
                                 'accuracy', percentage=True)

    plt.show()

def RandomForest(X,y):
    trainX, testX, trainY, testY = train_test_split(X, y, train_size=0.7, stratify=y)

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ['sqrt', 'log2']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        accuracy_results = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trainX, trainY)
                predictedY = rf.predict(testX)
                yvalues.append(accuracy_score(testY, predictedY))
            accuracy_results[d] = yvalues
        multiple_line_chart(axs[0, k], n_estimators, accuracy_results, 'Random Forests with %s features' % f,
                                 'nr estimators',
                                 'accuracy', percentage=True)

    plt.show()

def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')

def plot_confusion_matrix(testY, predictedY, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):


    confusion_mtx = confusion_matrix(testY, predictedY, classes)
    print("Confusion matrix: " +str(confusion_mtx))

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_mtx, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_mtx.shape[1]),
           yticks=np.arange(confusion_mtx.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.title('Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_mtx.max() / 2.
    for i in range(confusion_mtx.shape[0]):
        for j in range(confusion_mtx.shape[1]):
            ax.text(j, i, format(confusion_mtx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_mtx[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_roc_chart(ax: plt.Axes, models: dict, testX, testY, target: str = 'class'):
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.5])
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)
    ax.plot([0, 1], [0, 1], color='navy', label='random', linestyle='--')

    for clf in models:
        scores = models[clf].predict_proba(testX)[:,1]
        print(scores)
        fpr, tpr, _ = roc_curve(testY, scores)
        roc_auc = roc_auc_score(testY, scores)
        ax.plot(fpr, tpr, label='%s (auc=%0.2f)' % (clf, roc_auc))
    ax.legend(loc="lower center")


if __name__ == "__main__":
    input_data = read_input_file('/Users/Olga/Desktop/pd_speech_features.csv')
    data = data_exploration(input_data)     #returns list of example datasets
    normalized_data = data_normalization(data)
    data_balancing(normalized_data)
