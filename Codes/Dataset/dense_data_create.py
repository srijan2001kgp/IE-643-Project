import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

def add_noise_multichannel(signals, noise_positions, window_size=30, snr_db=0.025):
    """
    Add Gaussian noise to multiple signal channels at specified positions.
    Returns modified signals and a binary label mask.
    """
    num_samples, num_channels = signals.shape
    snr_linear = 10 ** (snr_db / 10)
    noisy_signals = signals.copy()
    label_mask = np.zeros(num_samples, dtype=np.int8)

    for pos in noise_positions:
        end = min(pos + window_size, num_samples)
        window = slice(pos, end)
        label_mask[window] = 1

        # Compute average signal power over all channels in window
        signal_window = signals[window, :]
        signal_power = np.mean(signal_window ** 2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(end - pos, num_channels)
        noisy_signals[window, :] += noise

    return noisy_signals, label_mask


def plot_ecg_data(ecg_id, signals, plot_dir):
    """
    Plots 2-channel ECG data and saves the image.
    """
    width_px, height_px, dpi = 512, 512, 100
    colors = ['red', 'blue']

    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi
    )
    for ch, ax in enumerate(axs):
        ax.plot(signals[:, ch], color=colors[ch], linewidth=1)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ecg_plot_{ecg_id}.png"), dpi=dpi)
    plt.close(fig)


def create(type:str):
    output_dir = f'Dense_data/{type}/images'
    os.makedirs(output_dir, exist_ok=True)

    x = np.load(f'Data_split/{type}/{type}.npy')
    df = pd.read_csv(f'Data_split/{type}/{type}.csv')
    labels = df.iloc[:, 1:].to_numpy(dtype=np.int8)
    ids = df.iloc[:, 0].to_numpy()

    x_mod = np.zeros_like(x)
    num_samples = len(x)  # just in case

    for i in range(num_samples):
        noise_positions = random.sample(range(10, 221), 3)
        x_mod[i], label_mask = add_noise_multichannel(x[i], noise_positions)
        labels[i] = label_mask
        plot_ecg_data(ids[i], x_mod[i], output_dir)

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"Processed {i + 1}/{num_samples} samples")

    # Save modified data
    np.save(f"Dense_data/{type}/{type}.npy", x_mod)
    combined_df = pd.concat([pd.Series(ids, name='ids'),
                             pd.DataFrame(labels)], axis=1)
    combined_df.to_csv(f'Dense_data/{type}/{type}.csv', index=False)


if __name__ == "__main__":
    create("train")
    print("---------------Finished dataset creation-----------------")
