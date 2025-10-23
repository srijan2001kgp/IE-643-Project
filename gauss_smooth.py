import numpy as np
import csv
import matplotlib.pyplot as plt

def gaussian_kernel(w):
    """
    Creates a 1D Gaussian kernel.

    Args:
        w: The window size for the kernel.

    Returns:
        A normalized 1D Gaussian kernel.
    """
    sigma = w//3
    x = np.linspace(-w, w, 2*w+1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel

def apply_label_smoothing(labels_batch,w=5):
        """Apply label smoothing to binary labels and convolve with Gaussian kernel"""
        smoothed_labels_batch = []
        for label in labels_batch:
            original_length = len(label)
              # Determine padding size
            pad_size = w // 2

              # Pad the labels array
            padded_labels = np.pad(label, (pad_size, pad_size), mode='constant', constant_values=0)
            knl=gaussian_kernel(w)
              # Apply convolution
            convolved_labels = np.convolve(padded_labels, knl,mode='same')

              # Trim the convolved output to match the original labels array size and scale the values appropriately
            trimmed_convolved_labels = convolved_labels[pad_size : pad_size + original_length]
            smoothed_labels_batch.append(trimmed_convolved_labels)

        a=np.array(smoothed_labels_batch)
        return a 


def read_first_n_rows(filepath, num_rows):
    """
    Reads the first 'num_rows' of a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        num_rows (int): The number of rows to read.

    Returns:
        list: A list of lists, where each inner list represents a row.
    """
    first_rows = []
    with open(filepath, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for i, row in enumerate(csv_reader):
            if i >= num_rows:
                break
            first_rows.append(row)
    return first_rows

data=read_first_n_rows('Data_split\\test\\test.csv',5)
data=np.array(data,dtype=np.int8)
data=data[1:,1:]
print(data.shape)
mod=apply_label_smoothing(data,15)
print(mod.shape)
fig,ax=plt.subplots(figsize=(10,6),dpi=100,nrows=len(data),ncols=1)
for i in range(len(data)):
     ax[i].plot(data[i],color='red')
     ax[i].plot(mod[i],color='blue')

plt.tight_layout()
plt.show()
plt.close()