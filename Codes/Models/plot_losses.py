import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "KD_train_1/student_train_1.log"

# Lists to store values
train_losses = []
val_losses = []

# Regex patterns
train_pattern = r"Average Training Loss:\s*([0-9]*\.?[0-9]+)"
val_pattern = r"Validation Loss:\s*([0-9]*\.?[0-9]+)"

# Read the log file and extract losses
with open(log_file, "r") as f:
    for line in f:
        train_match = re.search(train_pattern, line)
        val_match = re.search(val_pattern, line)

        if train_match:
            train_losses.append(float(train_match.group(1)))
        if val_match:
            val_losses.append(float(val_match.group(1)))

# Check what was extracted
print(f"Extracted {len(train_losses)} training losses and {len(val_losses)} validation losses.")
epc=range(1,len(train_losses)+1)
# Plot the losses
plt.figure(figsize=(8, 5))
if train_losses:
    plt.plot(epc,train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
if val_losses:
    plt.plot(epc,val_losses, marker='s', linestyle='--', color='r', label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.xticks(epc)
plt.tight_layout()
plt.savefig('lstm_1.pdf')
plt.show()
