# ArtFusion 🎨 - Fast Neural Style Transfer Studio

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Backend](https://img.shields.io/badge/Backend-TensorFlow/Keras-orange.svg)](https://www.tensorflow.org/)

Transform your images and videos into stunning works of art! ArtFusion uses the power of **Fast Neural Style Transfer** to apply artistic styles to your content in near real-time, all through an easy-to-use web interface built with Streamlit.

---

## Introduction: Fast Neural Style Transfer vs. Traditional NST

**Neural Style Transfer (NST)** is a fascinating technique that emerged from deep learning research, allowing us to separate the *content* of one image from the *style* of another and combine them. The original approach, pioneered by Gatys et al., involved using a pre-trained Convolutional Neural Network (CNN, typically VGG) to extract content and style features. It then iteratively optimized a new image (starting from noise or the content image) to minimize a combined loss function:
*   **Content Loss:** Ensures the output image retains the subject matter of the content image.
*   **Style Loss:** Ensures the output image matches the textural patterns and color palettes of the style image across different network layers.

**The Challenge with Traditional NST:** While powerful and flexible (it can work with *any* content and style image pair), the optimization process is computationally expensive and slow. Generating a single stylized image can take minutes or even longer, making it unsuitable for real-time applications or video processing.

**Enter Fast Neural Style Transfer:** To overcome the speed limitation, researchers like Johnson et al. proposed a different approach. Instead of optimizing an output image directly, they trained a *separate, feed-forward neural network for each specific style*.
*   **Training:** A dedicated "Style Transfer Network" is trained on a large dataset of content images. During training, it learns to transform any input image into the *target artistic style* while preserving content, using the same perceptual loss functions (content and style loss) derived from a fixed loss network (like VGG).
*   **Inference (Stylization):** Once trained, applying the style is incredibly fast. You simply pass your content image through the specific trained network in a single forward pass.

**Comparison:**

| Feature         |        Traditional NST         |                 Fast NST                      |
| :-------------- | :----------------------------- | :-------------------------------------------- |
| **Speed**       | Slow (optimization per image)  | Fast (single forward pass per image)          |
| **Flexibility** | High (any style image)         | Lower (requires a trained network per style)  |
| **Training**    | None (optimization at runtime) | Required (one network per style, offline)     |
| **Real-time**   | Difficult                      | Yes                                           |
| **Video**       | Very slow / Impractical        | Feasible                                      |

---

## Project Implementation: How ArtFusion Works

This project provides a user-friendly interface to experiment with Fast Neural Style Transfer:

1.  **Backend:** Uses TensorFlow and Keras to load and run pre-trained Fast Style Transfer models (`.keras` or `.h5` format). Each model is trained for a specific artistic style.
2.  **Frontend:** A web application built with Streamlit allows users to:
    *   Select from a list of available artistic styles (detected from models found in the `Model` folder).
    *   View a preview of the selected style image.
    *   Upload their own content image (JPG, PNG, JPEG) or video (MP4, AVI, MOV, MKV).
3.  **Processing Pipeline:**
    *   **Image Input:** The uploaded image is preprocessed (resized, normalized) and fed into the selected style transfer model. The model outputs the stylized image tensor.
    *   **Video Input:** The uploaded video is processed frame-by-frame. Each frame is extracted, preprocessed, passed through the style transfer model, deprocessed, and then re-encoded into a new output video file. OpenCV is used for video reading and writing.
    *   **Output:** The stylized image or video is displayed in the web app, and a download button is provided.
4.  **Model & Style Management:** The application automatically discovers available styles by looking for model files (`.keras`, `.h5`) in the `Model` directory and corresponding style preview images (with the same base name) in the `dataset/style_images` directory.

---

## Visual Showcase

Here's an example of ArtFusion transforming a content image using different artistic styles:

**Original Content Image:**

![Original Content Image](https://drive.google.com/uc?export=view&id=1SD98d-NCgVq4CwQqcleetwl8rxeTOv_p)

**Transformed Image :**

![Transformed Image](https://drive.google.com/uc?export=view&id=17RD430i3OVpN0OHD73EKlu4kcZsdbkjE)

---

## Getting Started

Follow these steps to set up and run the ArtFusion Streamlit application locally.

### Prerequisites

*   **Python:** Version **3.10** is required. You can download it from python.org.
*   **pip:** Python's package installer (usually comes with Python).
*   **Git:** To clone the repository (optional, you can also download the code as a ZIP).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/srrishtea/ArtFusion.git
    cd ArtFusion
    ```
2.  **Install Dependencies:**
        ```txt
        streamlit
        numpy
        tensorflow
        Pillow
        opencv-python
        ```
3.  **Download and Place Models & Style Images:**
    *   Download the pre-trained models from the [link](https://drive.google.com/drive/folders/1HTT0nRcji4szbZYJFEijGiTCScwX-ygx?usp=drive_link) .
    *   Place the downloaded model files (e.g., `starry_night.keras`) inside the `Model` folder in the project's root directory.
    *   Download the style images dataset from the [link](https://drive.google.com/drive/folders/1UZXB5zvPZYw5Y64Z8EJoNmdbIt3J-QZi?usp=sharing) .
    *   Ensure the style preview images (e.g., `starry_night.jpg`) are placed inside the `dataset/style_images` folder. The base name of the style image (without extension) **must match** the base name of its corresponding model file.

### Running the Streamlit App

1.  **Navigate to the Web App Directory:**
    Make sure your terminal/command prompt is inside the project's root directory where you cloned the repository. Then change into the `Web App` folder:
    ```bash
    cd "Web App"
    ```

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

3.  Streamlit will start the server, and the application should automatically open in your default web browser. You can also navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

## Understanding the Concepts

Beyond just using the app, understanding the underlying concepts is valuable.

### Total Variation Loss (TV Loss)

While the core losses for style transfer are Content Loss and Style Loss, sometimes an additional loss term called **Total Variation (TV) Loss** is used, particularly during the *training* of Fast NST networks :
*   **Purpose:** TV Loss acts as a spatial regularizer. It encourages smoothness in the generated image by penalizing large differences between adjacent pixel values.
*   **Effect:** It helps to reduce high-frequency artifacts, noise, or pixelation in the output image, leading to a more visually coherent and smoother result. While it might slightly reduce fine details, it often improves the overall quality of the stylized output.

### The Fast Neural Style Transfer Paper

The foundational work for the technique used in this project is:

*   **Title:** "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
*   **Authors:** Justin Johnson, Alexandre Alahi, Li Fei-Fei
*   **Conference:** European Conference on Computer Vision (ECCV), 2016
*   **Link:** arXiv:1603.08155

This paper details the architecture of the style transfer network and the use of perceptual loss functions for training feed-forward networks capable of fast stylization.
