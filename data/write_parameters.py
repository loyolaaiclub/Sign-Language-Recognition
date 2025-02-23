import os

DATA_FOLDER = './data'  # Adjust this path if needed
labels = []

# Collect folder names
for folder in os.listdir(DATA_FOLDER):
    if os.path.isdir(os.path.join(DATA_FOLDER, folder)):
        labels.append(folder)

# Save labels to labels.txt
labels_file = os.path.join(DATA_FOLDER, "labels.txt")
with open(labels_file, "w") as file:
    for label in labels:
        file.write(label + "\n")

print(f"Labels saved to {labels_file}")
