import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import pickle
import warnings
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Disable Streamlit's file watcher
# st.set_option("server.fileWatcherType", "none")

# Define class names manually (ensure the order matches the training dataset)
class_names = ['food-detection',
 'apple_pie',
 'chocolate_cake',
 'french_fries',
 'hot_dog',
 'ice_cream',
 'nachos',
 'onion_rings',
 'pancakes',
 'pizza',
 'ravioli',
 'samosa',
 'spring_rolls']

# Create metadata for visualization
metadata = MetadataCatalog.get("food_detection_metadata")
metadata.thing_classes = class_names

# Load the model configuration and weights
@st.cache_resource
def load_model(output_dir):
    # Log the model loading process
    print('Loading Food detection Model...')
    
    # Load configuration
    config_path = os.path.join(output_dir, "config.pkl")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    print(f"Model configuration loaded from {config_path}")

    # Load model weights
    model_weights_path = os.path.join(output_dir, "model_final.pth")
    cfg.MODEL.WEIGHTS = model_weights_path
    print(f"Model weights loaded from {model_weights_path}")

    # Force the model to run on the CPU
    cfg.MODEL.DEVICE = "cpu"

    # Create predictor
    predictor = DefaultPredictor(cfg)
    print('Loaded Food detection Model')
    return predictor, cfg

# Define paths
output_dir = "output_food_detection_mask_rcnn"

# Load the model once
predictor, cfg = load_model(output_dir)

# Streamlit UI
st.markdown("# 🍴 Food Detection Model")
st.markdown("Upload an image to detect food items, bounding boxes, and prediction details.")

# File uploader
uploaded_file = st.file_uploader("### 📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a temporary file path
    temp_file_path = os.path.join("temp_uploaded_image.jpg")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image using cv2
    image = cv2.imread(temp_file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization

    # Run the model on the image
    outputs = predictor(image)

    # Extract prediction details
    instances = outputs["instances"].to("cpu")
    prediction_details = []
    if "instances" in outputs:
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item()
            class_name = class_names[class_id]  # Map class ID to class name
            score = instances.scores[i].item()
            prediction_details.append(f"{class_name}: {score:.2f}%")

    # Visualize the predictions with distinct bounding boxes
    visualizer = Visualizer(
        image_rgb[:, :, ::-1],  # Convert RGB to BGR for visualization
        metadata=metadata,
        scale=0.8,  # Adjust scale for better visualization
        instance_mode=ColorMode.IMAGE  # Keep the background in color
    )

    predicted_image = image.copy()

    # Customize bounding box thickness and color based on image background
    instances = outputs["instances"].to("cpu")
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor.numpy()[0]  # Extract bounding box coordinates
        class_id = instances.pred_classes[i].item()
        class_name = class_names[class_id]
        score = instances.scores[i].item()

        # Calculate average brightness of the image to decide border color
        avg_brightness = np.mean(predicted_image)
        border_color = (0, 0, 0) if avg_brightness > 128 else (255, 255, 255)  # Black for bright images, white for dark images

        # Draw bounding box with dynamic color and thickness
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(predicted_image, (x1, y1), (x2, y2), border_color, thickness=10)  # Dynamic border color with thickness 6

        # Add class name and confidence score as label with a background
        label = f"{class_name}: {score:.2f}"

        # Get the text size to calculate the rectangle dimensions
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.5, thickness=16)

        # Define the rectangle coordinates
        rect_x1, rect_y1 = x1, y1 - text_height - 10  # Top-left corner of the rectangle
        rect_x2, rect_y2 = x1 + text_width, y1  # Bottom-right corner of the rectangle

        # Draw the rectangle (background) with a contrasting color
        background_color = (255, 255, 255) if border_color == (0, 0, 0) else (0, 0, 0)  # White for black borders, black for white borders
        cv2.rectangle(predicted_image, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, thickness=-1)  # Filled rectangle

        # Draw the text on top of the rectangle
        cv2.putText(predicted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.5, color=border_color, thickness=16)

    # Convert the image back to RGB for visualization
    predicted_image_rgb = predicted_image[:, :, ::-1]  # Reverse the color channels (BGR to RGB)
    # Create three columns for input, output, and prediction details
    col1, col2, col3 = st.columns([3, 3, 1.4])  # Adjust proportions: col1 and col2 are larger than col3

    # Display the original image in the first column
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image_rgb, use_container_width=True)

    # Display the predicted image in the second column
    with col2:
        st.markdown("### 📊 Predicted Output")
        # Convert the predicted image from BGR to RGB
        #predicted_image_rgb = vis.get_image()[:, :, ::-1]
        st.image(predicted_image_rgb, use_container_width=True)

    # Display the prediction details in the third column
    with col3:
        st.markdown("### 🔍Details")
        for detail in prediction_details:
            # Split the class name and confidence score
            class_name, confidence = detail.split(":")
            # Remove the '%' symbol and convert confidence to percentage
            confidence_value = float(confidence.strip().replace('%', '')) * 100
            st.markdown(f"**:blue[{class_name.strip()}]**: {confidence_value:.2f}%")