import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from io import BytesIO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1F77B4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F2F6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 25px;
        border-radius: 5px;
        margin: 5px 0;
        background: linear-gradient(90deg, #FF4B4B 0%, #FFA726 50%, #66BB6A 100%);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
    }
    .fresh-result {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 20px 0;
    }
    .rotten-result {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 20px 0;
    }
    .webcam-container {
        border: 3px solid #1F77B4;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .capture-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    .capture-btn:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🍎 Fruit Freshness Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
    This app uses a deep learning model to classify fruits as fresh or rotten. 
    Upload an image of an apple, banana, or orange, or use your webcam for real-time detection!
    """)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
    st.title("Fruit Freshness Detection")
    st.markdown("---")
    
    st.subheader("How to use:")
    st.markdown("""
    1. **Upload Image**: Upload an image of a fruit
    2. **Webcam**: Use your camera for real-time detection
    3. The model will analyze the image
    4. View the prediction and confidence scores
    """)
    
    st.markdown("---")
    st.subheader("Supported Fruits")
    st.markdown("""
    - 🍎 Apples (fresh/rotten)
    - 🍌 Bananas (fresh/rotten)
    - 🍊 Oranges (fresh/rotten)
    """)
    
    st.markdown("---")
    st.info("This model uses MobileNetV2 architecture trained on thousands of fruit images.")

# Load the model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('fruits_classification_mobilenetv2_20250826_233820.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Class names mapping
class_names = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

# Preprocess image
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        # Convert OpenCV BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
    else:
        img = image
    
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Handle different image formats (RGBA -> RGB, Grayscale -> RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Get fresh/rotten prediction
def get_fresh_rotten_prediction(prediction, class_names):
    # Calculate fresh and rotten probabilities
    fresh_prob = np.sum(prediction[0][:3])  # First three classes are fresh
    rotten_prob = np.sum(prediction[0][3:])  # Last three classes are rotten
    
    # Determine fruit type based on highest probability within category
    fruit_types = ["apple", "banana", "orange"]
    fresh_fruit_probs = prediction[0][:3]
    rotten_fruit_probs = prediction[0][3:]
    
    if fresh_prob > rotten_prob:
        fruit_type = fruit_types[np.argmax(fresh_fruit_probs)]
        confidence = fresh_prob
        status = "fresh"
    else:
        fruit_type = fruit_types[np.argmax(rotten_fruit_probs)]
        confidence = rotten_prob
        status = "rotten"
    
    return status, fruit_type, confidence, fresh_prob, rotten_prob

# Display prediction with confidence bars
def display_prediction(prediction, class_names):
    # Get fresh/rotten prediction
    status, fruit_type, confidence, fresh_prob, rotten_prob = get_fresh_rotten_prediction(prediction, class_names)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        
        # Display emoji based on fruit type
        emoji = "🍎" if fruit_type == "apple" else "🍌" if fruit_type == "banana" else "🍊"
        
        # Display result with appropriate styling
        if status == "fresh":
            st.markdown(f"**{emoji} {fruit_type.capitalize()} is FRESH ✅**")
            st.markdown(f"Confidence: {confidence*100:.2f}%")
            
            # Confidence bar for fresh
            bar_html = f"""
            <div style="background: #e0e0e0; border-radius: 5px; height: 20px; width: 100%; margin: 5px 0;">
                <div style="background: #4CAF50; 
                            height: 100%; width: {confidence*100}%; 
                            border-radius: 5px; text-align: center; color: white; 
                            font-weight: bold; line-height: 20px;">
                    {confidence*100:.1f}%
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
        else:
            st.markdown(f"**{emoji} {fruit_type.capitalize()} is ROTTEN ❌**")
            st.markdown(f"Confidence: {confidence*100:.2f}%")
            
            # Confidence bar for rotten
            bar_html = f"""
            <div style="background: #e0e0e0; border-radius: 5px; height: 20px; width: 100%; margin: 5px 0;">
                <div style="background: #F44336; 
                            height: 100%; width: {confidence*100}%; 
                            border-radius: 5px; text-align: center; color: white; 
                            font-weight: bold; line-height: 20px;">
                    {confidence*100:.1f}%
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Create a visualization of fresh vs rotten probabilities
        fig, ax = plt.subplots(figsize=(8, 5))
        
        categories = ['Fresh', 'Rotten']
        probabilities = [fresh_prob * 100, rotten_prob * 100]
        colors = ['#4CAF50', '#F44336']
        
        bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Fresh vs Rotten Confidence')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, probability in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{probability:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    return status, fruit_type, confidence

# Webcam capture function
def capture_image():
    st.markdown("### 📷 Webcam Capture")
    st.markdown("Click the button below to capture an image from your webcam")
    
    # Initialize session state for captured image
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    
    # Use OpenCV to capture from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera permissions.")
        return None
    
    # Create placeholder for video feed
    frame_placeholder = st.empty()
    capture_button = st.button("Capture Image", type="primary")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam")
            break
            
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        if capture_button:
            st.session_state.captured_image = frame_rgb
            break
    
    cap.release()
    frame_placeholder.empty()
    
    return st.session_state.captured_image

# Simple webcam using streamlit-camera-input-lite (fallback option)
def simple_webcam_capture():
    try:
        from camera_input_lite import camera_input
        st.markdown("### 📷 Take a Photo")
        image = camera_input(label="Take a picture of your fruit")
        return image
    except ImportError:
        st.warning("For better webcam experience, install: pip install streamlit-camera-input-lite")
        return None

# Main app functionality
def main():
    model = load_model()
    
    if model is None:
        st.error("Please ensure the model file 'fruits_classification_mobilenetv2.h5' is in the same directory as this app.")
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a fruit image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image of an apple, banana, or orange"
        )
        
        if uploaded_file is not None:
            process_and_display_image(uploaded_file, model)
    
    with tab2:
        st.markdown("### Real-time Fruit Freshness Detection")
        
        # Webcam options
        webcam_option = st.radio(
            "Choose webcam method:",
            ["Simple Camera Input", "OpenCV Webcam"],
            help="Select how you want to use your webcam"
        )
        
        if webcam_option == "Simple Camera Input":
            # Try using camera_input_lite
            try:
                from camera_input_lite import camera_input
                
                image = camera_input("Take a picture of your fruit")
                if image is not None:
                    process_and_display_image(image, model)
                    
            except ImportError:
                st.warning("""
                **Install camera input package for better experience:**
                ```bash
                pip install streamlit-camera-input-lite
                ```
                """)
                st.info("Falling back to OpenCV webcam...")
                webcam_option = "OpenCV Webcam"
        
        if webcam_option == "OpenCV Webcam":
            st.markdown("""
            **Instructions for OpenCV Webcam:**
            1. Click 'Start Webcam' below
            2. Allow camera permissions if prompted
            3. Position your fruit in front of the camera
            4. Click 'Capture Image' when ready
            5. The image will be analyzed automatically
            """)
            
            if st.button("🎥 Start Webcam", type="primary"):
                captured_image = capture_image()
                if captured_image is not None:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(captured_image)
                    process_and_display_image(pil_image, model)

# Function to process and display image (common for both upload and webcam)
def process_and_display_image(image_input, model):
    # Display the image
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Captured Image", use_column_width=True)
    
    with col2:
        # Auto-analyze for webcam, manual button for uploads
        analyze = True
        if hasattr(image_input, 'name'):  # It's a file upload
            analyze = st.button("Analyze Image", type="primary", use_container_width=True)
        
        if analyze:
            with st.spinner("Analyzing image..."):
                # Preprocess and predict
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Display results
                st.success("Analysis Complete!")
                status, fruit_type, confidence = display_prediction(prediction, class_names)
                
                # Display final verdict with appropriate styling
                emoji = "🍎" if fruit_type == "apple" else "🍌" if fruit_type == "banana" else "🍊"
                
                if status == "fresh":
                    st.markdown(f"""
                    <div class="fresh-result">
                        <h2 style="color: #2E7D32; text-align: center;">
                            {emoji} This {fruit_type} is FRESH ✅
                        </h2>
                        <p style="font-size: 1.2rem; text-align: center;">Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rotten-result">
                        <h2 style="color: #C62828; text-align: center;">
                            {emoji} This {fruit_type} is ROTTEN ❌
                        </h2>
                        <p style="font-size: 1.2rem; text-align: center;">Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

# Additional analysis section (moved outside main to avoid duplication)
def show_additional_analysis():
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Model Information", "How It Works", "Tips for Best Results"])
    
    with tab1:
        st.markdown("""
        **Model Architecture:** MobileNetV2
        - **Input Size:** 224x224 pixels
        - **Number of Classes:** 6 (fresh/rotten for apples, bananas, oranges)
        - **Training Data:** Thousands of fruit images
        - **Accuracy:** >95% on test data
        """)
        
        # Display model summary
        if st.button("Show Model Summary"):
            summary = []
            model.summary(print_fn=lambda x: summary.append(x))
            summary_text = "\n".join(summary)
            st.text_area("Model Architecture", summary_text, height=300)
    
    with tab2:
        st.markdown("""
        ### How the Model Works:
        
        1. **Image Preprocessing**: The uploaded image is resized to 224x224 pixels and normalized
        2. **Feature Extraction**: MobileNetV2 extracts important features from the image
        3. **Classification**: The model analyzes these features to determine freshness
        4. **Confidence Scoring**: The model calculates confidence for fresh vs rotten
        
        The model was trained using transfer learning on a pre-trained MobileNetV2 network,
        with additional layers specifically trained for fruit freshness detection.
        """)
    
    with tab3:
        st.markdown("""
        ### For Best Results:
        
        - Use clear, well-lit images
        - Focus on the fruit (avoid background clutter)
        - Take photos from multiple angles if unsure
        - Ensure the fruit occupies most of the image frame
        
        **Common Issues:**
        - Blurry images reduce accuracy
        - Poor lighting can affect predictions
        - Mixed freshness (partially rotten) may give uncertain results
        """)

# Display sample images when no input is provided
def show_sample_images():
    st.info("👆 Please upload an image or use webcam to get started.")
    
    # Sample images section
    st.markdown("### Sample Images You Can Test:")
    
    # Create a grid of sample images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Fresh Apple**")
        st.image("https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=400", 
                 use_column_width=True)
    
    with col2:
        st.markdown("**Rotten Banana**")
        st.image("https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400", 
                 use_column_width=True)
    
    with col3:
        st.markdown("**Fresh Orange**")
        st.image("https://images.unsplash.com/photo-1547514701-42782101795e?w=400", 
                 use_column_width=True)
    
    st.markdown("""
    *Note: These are example images. Please upload your own images for analysis.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Built with Streamlit, TensorFlow, and OpenCV | Fruit Freshness Classification Model</p>
    <p>For demonstration purposes only | Accuracy may vary with real-world conditions</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    # Show additional analysis and sample images
    show_additional_analysis()
    show_sample_images()