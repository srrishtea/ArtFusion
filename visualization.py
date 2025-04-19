import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    @staticmethod
    def display_image(image, title=None):
        """
        Displays a single image using matplotlib.
        Args:
            image: The image to display (numpy array).
            title: Optional title for the image.
        """
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def compare_images(original, stylized):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(stylized)
        plt.title('Stylized Image')
        plt.axis('off')

        plt.show()

    @staticmethod
    def save_side_by_side(content_image, stylized_image, output_path):
        """
        Saves two images side by side in a single output file.
        Args:
            content_image: The original content image (numpy array).
            stylized_image: The stylized image (numpy array).
            output_path: The path to save the combined image.
        """
        # Ensure the images are in the range [0, 1] for saving
        content_image = np.clip(content_image, 0, 1)
        stylized_image = np.clip(stylized_image, 0, 1)

        # Combine the two images side by side
        combined_image = np.concatenate((content_image, stylized_image), axis=1)

        # Save the combined image
        plt.figure(figsize=(10, 5))
        plt.imshow(combined_image)
        plt.axis('off')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()