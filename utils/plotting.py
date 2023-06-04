import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_heatmap(cm, saveToFile=None, annot=True, fmt="d", cmap="Blues", xticklabels=None, yticklabels=None):
    """
    Plots a heatmap of the confusion matrix.

    Parameters:
        cm (list of lists): The confusion matrix.
        annot (bool): Whether to annotate the heatmap with the cell values. Default is True.
        fmt (str): The format specifier for cell value annotations. Default is "d" (integer).
        cmap (str): The colormap for the heatmap. Default is "Blues".
        xticklabels (list): Labels for the x-axis ticks. Default is None.
        yticklabels (list): Labels for the y-axis ticks. Default is None.

    Returns:
        None
    """
    
    # Convert the confusion matrix to a NumPy array
    cm = np.array(cm)

    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(cm, cmap=cmap)
    
    # Display cell values as annotations
    if annot:
        # Normalize the colormap to get values between 0 and 1
        norm = Normalize(vmin=cm.min(), vmax=cm.max())
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                value = cm[i, j]
                # Determine text color based on cell value
                text_color = 'white' if norm(value) > 0.5 else 'black'  
                text = ax.text(j, i, format(value, fmt), ha="center", va="center", color=text_color)

    # Set x-axis and y-axis ticks and labels
    if xticklabels:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    # Set labels and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix Heatmap")

    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Show the plot
    if(saveToFile is not None):
        plt.savefig(saveToFile)
        
    plt.show()