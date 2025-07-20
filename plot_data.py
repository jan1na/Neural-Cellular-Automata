import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('data/pathmnist.npz')

# Access the image and label arrays
X_train = data['train_images']  # shape: (89996, 28, 28, 3)
y_train = data['train_labels']  # shape: (89996, 1)

# Display a few sample images with labels
class_names = [
    "Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
    "Smooth Muscle", "Normal", "Stroma", "Tumor"
]

plt.figure(figsize=(8, 2))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[int(y_train[i])], fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig("pathmnist_examples.png", dpi=300)  # Save for use in poster