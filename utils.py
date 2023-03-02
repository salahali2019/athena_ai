import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def inference(model, image_data):
    """
    Given a trained model and image data, perform inference and return the predicted classes for each image.

    Args:
        model (object): Trained machine learning model.
        image_data (ndarray): Numpy array of shape (n_samples, height, width, n_channels) containing image data.

    Returns:
        ndarray: Numpy array of shape (n_samples,) containing predicted classes for each image.
    """
    images = []
    predictions = []

    for i in range(image_data.shape[0]):
        prediction = model.predict(image_data[i])
        predictions.append(prediction)

    predictions = np.array(predictions)

    return predictions


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots a confusion matrix given the true and predicted labels, as well as a list of class names.

    Args:
        y_true (numpy.ndarray): True labels (1D array of integers)
        y_pred (numpy.ndarray): Predicted labels (1D array of integers)
        classes (list): List of class names (strings)

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title='Confusion matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.show()


def plot_calibration_curve(y_true, y_prob):
    """
    Plots the calibration curve for a given set of predicted probabilities and true labels.

    Args:
        y_true (ndarray): Numpy array of shape (n_samples,) containing true class labels.
        y_prob (ndarray): Numpy array of shape (n_samples, n_classes) containing predicted probabilities.

    Returns:
        None
    """
    ece, _ = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    mce = np.max(np.abs(np.subtract(ece, np.linspace(0, 1, 10))))

    plt.plot(np.linspace(0, 1, 10), ece, '-o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve (ECE={:.3f}, MCE={:.3f})'.format(ece.mean(), mce))
    plt.savefig('results/calibration_graph.png')

    
def save_false_positives(predictions, y_true, images, classes):
    """
    Given predicted and true classes along with the corresponding image data, save the false positive images of each class.

    Parameters:
        predictions (ndarray): Numpy array of shape (n_samples,) containing predicted classes for each image.
        y_true (ndarray): Numpy array of shape (n_samples,) containing true classes for each image.
        images (ndarray): Numpy array of shape (n_samples, height, width, n_channels) containing image data.
        classes (ndarray): Numpy array of shape (n_classes,) containing the unique classes.

    Returns:
        None
    """
    # Iterate through each class
    for i in range(len(classes)):
        # Get the false positive images for the current class
        false_positives = images[(predictions == classes[i]) & (y_true != classes[i])]
        # If there are any false positives for the current class, save them in a subfolder of 'results/false_positives'
        if len(false_positives) > 0:
            # Create the 'results/false_positives' directory if it doesn't already exist
            if not os.path.exists('results/false_positives'):
                os.makedirs('results/false_positives')
            # Create a subfolder for the current class if it doesn't already exist
            if not os.path.exists('results/false_positives/{}'.format(i)):
                os.makedirs('results/false_positives/{}'.format(i))
            # Iterate through each false positive image for the current class and save it as a PNG file
            for j in range(len(false_positives)):
                img = Image.fromarray(false_positives[j])
                img.save('results/false_positives/{}/{}_{}.png'.format(i, classes[i], j))


def find_false_positive_patterns(results_path='results/false_positives'):
    """
    Given a directory of false positive images, find potential patterns in the images.

    Parameters:
        results_path (str): Path to the directory containing the false positive images.

    Returns:
        None
    """
    # Initialize an empty list to store false positives
    false_positives = []

    # Iterate through each folder in the results_path directory
    for folder in os.listdir(results_path):
        # Iterate through each image in the current folder
        for filename in os.listdir('{}/{}'.format(results_path, folder)):
            # Open the image using PIL and append it to the list of false positives
            img = Image.open('{}/{}/{}'.format(results_path, folder, filename))
            false_positives.append(np.array(img))

    # Convert the list of false positives to a numpy array
    false_positives = np.array(false_positives)

    # Convert the images to features using PCA
    pca = PCA(n_components=2)
    features = pca.fit_transform(false_positives.reshape((-1, np.prod(false_positives.shape[1:]))))

    # Cluster the features using KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
    labels = kmeans.labels_

    # Save the clustered images to files
    for i in range(len(labels)):
        img = Image.fromarray(false_positives[i])
        img.save('results/false_positives/cluster_{}/{}.png'.format(labels[i], i))
