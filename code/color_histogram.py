import numpy as np

def color_histogram(x_min, y_min, x_max, y_max, frame, hist_bin):
    """
    Compute the color histogram of a region in an image.

    Parameters:
    - x_min (int): The minimum x-coordinate of the region.
    - y_min (int): The minimum y-coordinate of the region.
    - x_max (int): The maximum x-coordinate of the region.
    - y_max (int): The maximum y-coordinate of the region.
    - frame (numpy.ndarray): The input image.
    - hist_bin (int): The number of bins for the histogram.

    Returns:
    - hist (numpy.ndarray): The computed color histogram.
    """
    
    # Split into RGB channels
    r = frame[y_min:y_max, x_min:x_max, 0]
    g = frame[y_min:y_max, x_min:x_max, 1]
    b = frame[y_min:y_max, x_min:x_max, 2]

    # Get hist values for each channel
    r_hist, _ = np.histogram(r, bins=hist_bin, range=[0, 256])
    g_hist, _ = np.histogram(g, bins=hist_bin, range=[0, 256])
    b_hist, _ = np.histogram(b, bins=hist_bin, range=[0, 256])

    # Concatenate all histograms
    hist = np.concatenate([r_hist, g_hist, b_hist]).reshape(1, -1)
    if np.sum(hist) != 0:
        hist = hist / np.sum(hist)

    return hist

