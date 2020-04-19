from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, precision_recall_fscore_support


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)

    #precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print('Precision', precision)
    print('Recall', recall)
    print('F-score', f1)
    print('Accuracy', accuracy_score(y_test, y_predicted))

    return accuracy, precision, recall, f1



