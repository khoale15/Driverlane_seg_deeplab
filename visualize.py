import cv2
import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
], dtype=np.uint8)

def visualize_1_image(image, mask_rgb, mask, prefix=""):
        overlay_rgb = cv2.addWeighted(image, 1.0, mask_rgb, .5, 0.0)
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.title(f"{prefix}Mask")
        plt.imshow(mask, cmap="gray", vmin=0, vmax=2)
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.title("Overlay")
        plt.imshow(overlay_rgb)
        plt.axis("off")
        plt.show()

def visualize(images, labels, k=6):
    k = min(k, len(images))
    for i in range(k):
        image = np.ascontiguousarray(images[i]) # HWC uint8 (works for ndarray/memmap)
        mask = labels[i].copy()
        mask[mask < 0] = 2
        mask = mask.astype("uint8")
        mask_rgb = COLOR_MAP[mask] # HWC uint8

        visualize_1_image(image, mask_rgb, mask)

def plot_metrics(train_loss, val_loss, val_miou):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    # mIoU plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_miou, label='Validation mIoU', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.legend()

    plt.tight_layout()
    plt.show()