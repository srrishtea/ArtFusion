import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image 
import glob 
import cv2 
import tempfile 
from io import BytesIO
import time 
import uuid 

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'Model')
STYLE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, 'dataset', 'style_images')
OUTPUT_FOLDER = os.path.join(APP_DIR, 'output') 
os.makedirs(OUTPUT_FOLDER, exist_ok=True) 

DEFAULT_STYLE_PREVIEW_SIZE = (150, 150)
COMMON_IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png')
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv'] 

IMG_HEIGHT = 224
IMG_WIDTH = 224


@st.cache_resource 
def load_style_transfer_model(model_path):
    print(f"DEBUG: Attempting to load model from: {model_path}") # Debug print
    if not os.path.exists(model_path):
        st.error(f"Error: os.path.exists check failed. Model file not found at {model_path}")
        return None
    try:
        with st.spinner(f"Loading style model: {os.path.basename(model_path)}..."):
            model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except FileNotFoundError: 
         st.error(f"Error: Explicit FileNotFoundError caught for path: {model_path}. Please ensure the file exists and is accessible.")
         print(f"[!] Explicit FileNotFoundError: {model_path}")
         return None
    except Exception as e:
        if "unable to open file" in str(e).lower() or ".keras format" in str(e):
             st.error(f"Error loading Keras model from {model_path}: Failed to open or parse the file. It might be corrupted, not a valid Keras file, or saved in an incompatible format. Ensure it's a '.keras' zip archive or a compatible '.h5' file.")
        else:
            st.error(f"Error loading Keras model from {model_path}: {e}")
        print(f"[!] Error loading Keras model: {e}")
        return None

def find_style_image(style_name, style_folder):
    for ext in COMMON_IMG_EXTENSIONS:
        pattern = os.path.join(style_folder, f"{style_name}{ext.replace('*','',1)}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0] 
    return None 

def get_available_styles(model_folder, style_image_folder):
    styles = {}
    if not os.path.isdir(model_folder):
        st.error(f"Model folder not found: {model_folder}")
        return styles

    model_files_keras = glob.glob(os.path.join(model_folder, '*.keras'))
    model_files_h5 = glob.glob(os.path.join(model_folder, '*.h5'))
    model_files = model_files_keras + model_files_h5 # Combine the lists

    if not model_files:
        st.warning(f"No '.keras' or '.h5' models found in {model_folder}")
        return styles

    for model_path in model_files:
        model_filename = os.path.basename(model_path)
        style_name = os.path.splitext(model_filename)[0] 

        style_image_path = find_style_image(style_name, style_image_folder)

        if style_image_path:
            styles[style_name] = {
                "model_path": model_path,
                "style_image_path": style_image_path
            }
        else:
            st.warning(f"Could not find a matching style image for model: {model_filename} in {style_image_folder}")
            print(f"[!] Warning: No style image found for {style_name} in {style_image_folder}")

    sorted_styles = dict(sorted(styles.items()))
    return sorted_styles


def preprocess_input_image(image_pil, target_dim=(IMG_HEIGHT, IMG_WIDTH)):
    """Prepares PIL image for model input."""
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img_resized = image_pil.resize(target_dim, Image.LANCZOS) 
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array / 255.0 
    img_array_batch = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array_batch, dtype=tf.float32)

def deprocess_output_image(tensor):
    tensor = tf.convert_to_tensor(tensor) 
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = (tensor + 1.0) / 2.0
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    img_array = tensor.numpy()
    img_array_uint8 = (img_array * 255.0).astype(np.uint8)
    return Image.fromarray(img_array_uint8)

def process_video(video_bytes, style_model, output_path, progress_bar):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name
    tfile.close() 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        os.unlink(video_path) 
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        st.warning("Warning: Could not determine total frame count. Progress bar may not be accurate.")
        total_frames = None 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (IMG_WIDTH, IMG_HEIGHT)) 

    frame_count = 0
    st.info(f"Processing video: ~{total_frames if total_frames else '?'} frames at {fps:.2f} FPS.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break 

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            input_tensor = preprocess_input_image(image_pil, target_dim=(IMG_HEIGHT, IMG_WIDTH))
            stylized_tensor_tanh = style_model(input_tensor, training=False)
            output_image_pil = deprocess_output_image(stylized_tensor_tanh)
            stylized_array_rgb = np.array(output_image_pil)
            stylized_frame_bgr = cv2.cvtColor(stylized_array_rgb, cv2.COLOR_RGB2BGR)
            out.write(stylized_frame_bgr)

            frame_count += 1
            if total_frames:
                progress_bar.progress(frame_count / total_frames)
            else:
                 progress_bar.progress(min(1.0, frame_count / 1000.0)) 

        except Exception as frame_e:
            st.warning(f"Skipping frame {frame_count} due to error: {frame_e}")
            print(f"[!] Error processing frame {frame_count}: {frame_e}")
            continue 

    cap.release()
    out.release()
    os.unlink(video_path) 
    cv2.destroyAllWindows() 

    if frame_count == 0:
        st.error("Error: No frames were processed from the video.")
        return None

    return output_path

st.set_page_config(layout="wide", page_title="Neural Style Transfer Studio", page_icon="🎨")

st.title("🎨 Neural Style Transfer Studio")
st.markdown("Upload your image or video and choose an artistic style to transform it!")
st.markdown("---") 

st.sidebar.header("🖌️ Choose a Style")
available_styles = get_available_styles(MODEL_FOLDER, STYLE_IMAGE_FOLDER)

if not available_styles:
    st.sidebar.error("No style models found or configured correctly. Please check the `Model` folder and ensure corresponding style images exist in `dataset/style_images`.")
    st.stop() 

style_names = list(available_styles.keys())
selected_style_name = st.sidebar.selectbox(
    "Select the artistic style:",
    options=style_names,
    index=0 
)

selected_style_info = available_styles[selected_style_name]
if "style_image_path" in selected_style_info and selected_style_info["style_image_path"]:
    try:
        style_img_preview = Image.open(selected_style_info["style_image_path"])
        st.sidebar.image(
            style_img_preview,
            caption=f"Style: {selected_style_name.replace('_', ' ').title()}",
            use_container_width=True 
        )
    except Exception as e:
        st.sidebar.warning(f"Could not load style preview: {e}")
else:
    st.sidebar.markdown("*(No preview available for this style)*")

selected_model_path = selected_style_info["model_path"]
model = load_style_transfer_model(selected_model_path)

st.header("🖼️ Upload Content Image or Video 🎬")
content_file = st.file_uploader(
    "Choose the image or video you want to transform...",
    type=['jpg', 'png', 'jpeg'] + VIDEO_EXTENSIONS 
)

if content_file is not None and model is not None:
    file_details = {"FileName": content_file.name, "FileType": content_file.type, "FileSize": content_file.size}
    is_image = content_file.type.startswith('image/')
    is_video = content_file.type.startswith('video/')

    if is_image:
        st.subheader("Image Processing")
        col1, col2 = st.columns(2)
        try:
            content_image_pil = Image.open(content_file).convert('RGB')

            with col1:
                st.image(content_image_pil, caption="Original Uploaded Image", use_column_width=True)

            if st.button(f"✨ Apply '{selected_style_name.replace('_', ' ').title()}' Style to Image!", key=f"apply_img_{selected_style_name}"):
                with st.spinner(f"Applying style: {selected_style_name}... 🎨"):
                    try:
                        input_tensor = preprocess_input_image(content_image_pil)
                        stylized_tensor_tanh = model(input_tensor, training=False)
                        output_image_pil = deprocess_output_image(stylized_tensor_tanh)

                        with col2:
                            st.image(output_image_pil, caption=f"Stylized Result", use_column_width=True)
                            buf = BytesIO()
                            output_image_pil.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            st.download_button(
                                label="⬇️ Download Stylized Image",
                                data=byte_im,
                                file_name=f"stylized_{selected_style_name}_{os.path.splitext(content_file.name)[0]}.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"❌ An error occurred during image stylization: {e}")
                        print(f"[!] Prediction error (Image): {e}")

        except Exception as e:
            st.error(f"❌ Error loading or processing the uploaded image: {e}")
            print(f"[!] Image loading error: {e}")

    elif is_video:
        st.subheader("Video Processing")
        st.info("Video processing can take a significant amount of time depending on the video length and your hardware.")

        if st.button(f"🎬 Apply '{selected_style_name.replace('_', ' ').title()}' Style to Video!", key=f"apply_vid_{selected_style_name}"):
            progress_bar = st.progress(0.0)
            status_text = st.empty() # Placeholder for status updates
            status_text.text("Starting video processing...")

            # Generate a unique output filename
            unique_id = uuid.uuid4().hex[:8]
            output_filename = f"stylized_{selected_style_name}_{os.path.splitext(content_file.name)[0]}_{unique_id}.mp4"
            output_video_path = os.path.join(OUTPUT_FOLDER, output_filename)

            video_bytes = content_file.read() 

            start_time = time.time()
            with st.spinner(f"Applying style to video frames... This might take a while! ⏳"):
                try:
                    processed_path = process_video(video_bytes, model, output_video_path, progress_bar)

                    if processed_path:
                        end_time = time.time()
                        processing_time = end_time - start_time
                        status_text.success(f"Video processing complete! Took {processing_time:.2f} seconds.")
                        st.video(processed_path) 

                        with open(processed_path, "rb") as file:
                            btn = st.download_button(
                                label="⬇️ Download Stylized Video",
                                data=file,
                                file_name=output_filename, 
                                mime="video/mp4"
                            )
                    else:
                        status_text.error("Video processing failed. Check logs for details.")
                        progress_bar.progress(0.0) 

                except Exception as e:
                    status_text.error(f"❌ An error occurred during video processing: {e}")
                    print(f"[!] Video processing error: {e}")
                    progress_bar.progress(0.0) 

    else:
        st.warning(f"Unsupported file type: {content_file.type}. Please upload an image or a video.")

elif content_file is None:
    st.info("☝️ Upload an image or video using the uploader above to get started.")

elif model is None:
    st.error(f"Cannot apply style. The selected model ('{selected_style_name}') failed to load. Please check the model file and logs.")

st.markdown("---")
st.markdown("Fast Neural Style Transfer By Srishti Sharma.")
