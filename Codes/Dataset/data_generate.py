import wfdb
import matplotlib
matplotlib.use('Agg')  # no GUI rendering
import matplotlib.pyplot as plt
import numpy as np
import os, csv, time, gc

# Base directory for all outputs
BASE_DIR = os.getcwd()

# Ensure directory exists
os.makedirs(BASE_DIR, exist_ok=True)
# === CONFIG ===
# directory_path = "."
plot_dir_name = os.path.join(BASE_DIR, "Plots_ECG")
os.makedirs(plot_dir_name, exist_ok=True)

width_px, height_px, dpi = 512, 512, 100
duration = 2  # seconds per segment

# === Get .dat files ===
file_list = [f.split('.')[0] for f in os.listdir(BASE_DIR) if f.endswith('.dat')]
print(f"Found {len(file_list)} .dat files")

cnt = 0
all_segments = []  # will hold all 2-sec windows

with open(os.path.join(BASE_DIR,"labels_ECG.csv"), 'w', newline='') as f:
    writer = csv.writer(f)

    for file_name in file_list:
        st = time.time()
        header = wfdb.rdheader(file_name)
        freq = header.fs
        mul = int(duration * freq)

        # Read full record and annotations once
        record = wfdb.rdrecord(file_name)
        signal = record.p_signal
        total_samples = signal.shape[0]

        ann = wfdb.rdann(file_name, 'atr')
        ann_samples = np.array(ann.sample)

        n_segments = total_samples // mul
        col_list = ['red', 'blue']

        for i in range(n_segments):
            start = i * mul
            end_t = start + mul
            segment = signal[start:end_t]

            if segment.shape[0] < mul:
                continue

            # Save segment to memory
            all_segments.append(segment.astype(np.float32))

            # Plot and save image
            fig, axs = plt.subplots(figsize=(width_px/dpi, height_px/dpi),
                                    dpi=dpi, nrows=2, ncols=1)
            for ch, ax in enumerate(axs.flat):
                ax.plot(segment[:, ch], color=col_list[ch], linewidth=1)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{plot_dir_name}/ecg_plot_{cnt}.png", dpi=dpi)
            plt.close(fig)

            # Create annotation mask
            mask = np.zeros(mul, dtype=np.int8)
            ann_window = ann_samples[(ann_samples >= start) & (ann_samples < end_t)] - start
            mask[ann_window] = 1

            # Write to CSV
            writer.writerow([cnt] + mask.tolist())
            cnt += 1
        
        print(f"Processed {file_name} in {time.time() - st:.2f} s")

# === Combine and save all segments ===
all_segments = np.array(all_segments, dtype=np.float32)
np.save(os.path.join(BASE_DIR,"ECG_segments.npy"), all_segments)

print(f"Done. Saved {all_segments.shape[0]} segments.")
print(f"Shape of all_segments: {all_segments.shape}")
gc.collect()
