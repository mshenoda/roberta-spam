import numpy as np

def compute_metrics(y_true, y_pred, positive_label="spam", negative_label="ham"):
    """
    Compute evaluation metrics for binary classification.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        positive_label (optional): Value representing the positive label. Default is 1.
        negative_label (optional): Value representing the negative label. Default is 0.

    Returns:
        accuracy (float): Accuracy metric.
        precision (float): Precision metric.
        recall (float): Recall metric.
        f1 (float): F1-score metric.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate true positives, false positives, and false negatives
    tp = np.sum(y_true == y_pred)  # Count where true positive (both true and predicted labels are the same)
    fp = np.sum((y_true != y_pred) & (y_pred == positive_label))  # Count where false positive (true label is negative, but predicted as positive)
    fn = np.sum((y_true != y_pred) & (y_pred == negative_label))  # Count where false negative (true label is positive, but predicted as negative)
    
    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1


def confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix based on the ground truth and predicted labels.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        list of lists: The confusion matrix.
    """

    # Obtain the unique classes from y_true and y_pred
    classes = list(set(y_true + y_pred))
    classes.sort()

    # Calculate the total number of unique classes
    num_classes = len(classes)

    # Initialize the confusion matrix as a 2D list of zeros
    cm = [[0] * num_classes for _ in range(num_classes)]

    # Iterate over each pair of true and predicted labels
    for true, pred in zip(y_true, y_pred):
        # Find the indices of true and predicted classes in the classes list
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)

        # Increment the corresponding cell in the confusion matrix
        cm[true_idx][pred_idx] += 1

    # Return the confusion matrix
    return cm
