import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import os

# Set environment variables to avoid TensorFlow issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
except ImportError:
    st.error("TensorFlow not properly installed. Please check your installation.")
    st.stop()

# App configuration
st.set_page_config(
    page_title="Bone Cancer Detection",
    page_icon="🦴",
    layout="wide"
)

# Constants from the notebook
IMG_SIZE = 224
THRESHOLD = 0.001  # Best threshold from model evaluation

@st.cache_resource
def load_model():
    """Load the trained U-Net model"""
    try:
        # Try multiple loading strategies for compatibility
        with tf.device('/CPU:0'):  # Force CPU usage to avoid GPU issues
            # First attempt: Load with custom objects
            try:
                model = keras.models.load_model('__improved_best_unet_segmentation.h5', compile=False)
                return model
            except Exception as e1:
                st.warning(f"First load attempt failed: {str(e1)}")
                
                # Second attempt: Load weights only and rebuild model
                try:
                    # Load model architecture exactly matching your notebook
                    from tensorflow.keras import layers, models
                    
                    def attention_block(F_g, F_l, F_int):
                        """Attention gate for U-Net"""
                        g = layers.Conv2D(filters=F_int, kernel_size=1, strides=1, padding='valid')(F_g)
                        g = layers.BatchNormalization()(g)
                        
                        x = layers.Conv2D(filters=F_int, kernel_size=1, strides=1, padding='valid')(F_l)
                        x = layers.BatchNormalization()(x)
                        
                        psi = layers.Activation('relu')(layers.add([g, x]))
                        psi = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='valid')(psi)
                        psi = layers.BatchNormalization()(psi)
                        psi = layers.Activation('sigmoid')(psi)
                        
                        return layers.multiply([F_l, psi])

                    def conv_block(inputs, filters, dropout_rate=0.1):
                        """Convolutional block with batch normalization and dropout"""
                        x = layers.Conv2D(filters, 3, padding='same')(inputs)
                        x = layers.BatchNormalization()(x)
                        x = layers.Activation('relu')(x)
                        x = layers.Dropout(dropout_rate)(x)
                        
                        x = layers.Conv2D(filters, 3, padding='same')(x)
                        x = layers.BatchNormalization()(x)
                        x = layers.Activation('relu')(x)
                        
                        return x

                    def build_improved_unet(input_shape=(224, 224, 3), filters=32):
                        """Improved U-Net with attention gates and better architecture"""
                        inputs = layers.Input(input_shape)
                        
                        # Encoder
                        c1 = conv_block(inputs, filters)
                        p1 = layers.MaxPooling2D((2, 2))(c1)
                        
                        c2 = conv_block(p1, filters*2)
                        p2 = layers.MaxPooling2D((2, 2))(c2)
                        
                        c3 = conv_block(p2, filters*4)
                        p3 = layers.MaxPooling2D((2, 2))(c3)
                        
                        c4 = conv_block(p3, filters*8)
                        p4 = layers.MaxPooling2D((2, 2))(c4)
                        
                        # Bottleneck
                        c5 = conv_block(p4, filters*16, dropout_rate=0.2)
                        
                        # Decoder with attention gates
                        u6 = layers.UpSampling2D((2, 2))(c5)
                        att6 = attention_block(F_g=u6, F_l=c4, F_int=filters*4)
                        u6 = layers.concatenate([u6, att6])
                        c6 = conv_block(u6, filters*8)
                        
                        u7 = layers.UpSampling2D((2, 2))(c6)
                        att7 = attention_block(F_g=u7, F_l=c3, F_int=filters*2)
                        u7 = layers.concatenate([u7, att7])
                        c7 = conv_block(u7, filters*4)
                        
                        u8 = layers.UpSampling2D((2, 2))(c7)
                        att8 = attention_block(F_g=u8, F_l=c2, F_int=filters)
                        u8 = layers.concatenate([u8, att8])
                        c8 = conv_block(u8, filters*2)
                        
                        u9 = layers.UpSampling2D((2, 2))(c8)
                        att9 = attention_block(F_g=u9, F_l=c1, F_int=filters//2)
                        u9 = layers.concatenate([u9, att9])
                        c9 = conv_block(u9, filters)
                        
                        # Output
                        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
                        
                        model = models.Model(inputs, outputs)
                        return model
                    
                    # Build model and load weights
                    model = build_improved_unet()
                    model.load_weights('__improved_best_unet_segmentation.h5')
                    st.success("Model loaded successfully using weights-only approach!")
                    return model
                    
                except Exception as e2:
                    st.warning("Model loading failed - running in demo mode for educational purposes")
                    st.info("Demo mode simulates cancer detection to show app functionality")
                    return None
                    
    except Exception as e:
        st.error(f"Critical error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction - matching notebook implementation"""
    # Convert PIL image to OpenCV format first for consistency with notebook
    img_array = np.array(image)
    
    # Convert to RGB if needed (notebook converts from file path)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Already RGB
        pass
    else:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Store original dimensions for scaling back later
    original_height, original_width = img_array.shape[:2]
    
    # Convert back to PIL for exact notebook preprocessing
    img_pil = Image.fromarray(img_array)
    
    # Exact preprocessing from notebook: convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_rgb = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img_rgb) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)
    
    return input_arr, img_rgb, (original_height, original_width)

def predict_cancer_demo(input_arr):
    """Demo prediction function when model is not available"""
    # Create a more realistic demo prediction based on image characteristics
    img = input_arr[0]
    
    # Convert to grayscale for analysis (matching medical image analysis)
    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Create realistic prediction mask
    pred_mask = np.zeros((IMG_SIZE, IMG_SIZE))
    
    # Enhanced algorithm for more realistic bone cancer simulation
    # Look for bone-like structures and suspicious regions
    for i in range(0, IMG_SIZE, 15):
        for j in range(0, IMG_SIZE, 15):
            region = gray[i:i+15, j:j+15]
            if region.size > 0:
                # Calculate region statistics
                variance = np.var(region)
                mean_intensity = np.mean(region)
                edge_strength = np.var(np.gradient(region))
                
                # Higher probability in regions with medical characteristics
                # Suspicious regions: moderate intensity, high variance, edge structures
                if 0.15 < mean_intensity < 0.8 and variance > 0.008:
                    # Base probability from intensity and texture
                    prob = min(variance * 1.5 + edge_strength * 0.5, 0.008)
                    
                    # Add some spatial clustering (cancer tends to be clustered)
                    if i > 50 and i < IMG_SIZE-50 and j > 50 and j < IMG_SIZE-50:
                        prob *= 1.2  # Higher probability in central region
                    
                    # Add some randomness but keep it realistic
                    prob += np.random.normal(0, 0.0005)
                    pred_mask[i:i+15, j:j+15] = max(0, prob)
    
    # Apply smoothing for more realistic appearance
    try:
        from scipy.ndimage import gaussian_filter
        pred_mask = gaussian_filter(pred_mask, sigma=1.5)
        
        # Add some realistic cancer-like clusters
        if np.random.random() > 0.3:  # 70% chance of detection
            # Add 1-3 realistic cancer regions
            num_regions = np.random.randint(1, 4)
            for _ in range(num_regions):
                center_x = np.random.randint(40, IMG_SIZE-40)
                center_y = np.random.randint(40, IMG_SIZE-40)
                size = np.random.randint(15, 35)
                
                # Create circular/elliptical region
                y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                mask_region = ((x - center_x)**2 + (y - center_y)**2) <= size**2
                pred_mask[mask_region] += np.random.uniform(0.003, 0.012)
        
        # Ensure realistic probability range
        pred_mask = np.clip(pred_mask, 0, 0.015)
        
    except ImportError:
        pass  # Skip advanced processing if scipy not available
    
    return pred_mask

def predict_cancer(model, input_arr, threshold=THRESHOLD):
    """Predict cancer segmentation - matching notebook implementation"""
    if model is not None:
        # Exact prediction from notebook: model.predict(input_arr)[0].squeeze()
        pred_mask = model.predict(input_arr)[0].squeeze()
    else:
        # Demo mode
        st.info("🔍 Running in demo mode - showing simulated cancer detection for educational purposes")
        pred_mask = predict_cancer_demo(input_arr)
    
    # Create binary mask with proper threshold (matching notebook)
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8)
    
    return pred_mask, pred_mask_bin

def create_advanced_visualization(img_rgb, pred_mask_bin, original_dims=None):
    """Create advanced visualization matching notebook implementation"""
    # Find contours in the predicted mask (matching notebook)
    pred_mask_uint8 = (pred_mask_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to remove noise (matching notebook)
    min_contour_area = 10  # Adjust as needed
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create visualization on a copy of the model-sized image (matching notebook)
    img_viz = np.array(img_rgb).copy()
    
    # Draw contours and bounding boxes (matching notebook)
    for cnt in contours:
        # Draw contour in red with thickness=2 for better visibility
        cv2.drawContours(img_viz, [cnt], -1, (255, 0, 0), 2)
        
        # Draw bounding box in green
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    results = {
        'visualization': img_viz,
        'contours': contours,
        'num_detections': len(contours)
    }
    
    # If original dimensions provided, create scaled version
    if original_dims is not None:
        original_height, original_width = original_dims
        scale_x = original_width / IMG_SIZE
        scale_y = original_height / IMG_SIZE
        
        # Create visualization on original scale
        original_img_rgb = np.array(img_rgb)
        original_img_rgb = cv2.resize(original_img_rgb, (original_width, original_height))
        
        for cnt in contours:
            # Scale contour to original image dimensions
            scaled_cnt = cnt.copy()
            scaled_cnt = scaled_cnt.astype(np.float32)
            scaled_cnt[:, :, 0] *= scale_x
            scaled_cnt[:, :, 1] *= scale_y
            scaled_cnt = scaled_cnt.astype(np.int32)
            
            # Draw scaled contour on original image
            cv2.drawContours(original_img_rgb, [scaled_cnt], -1, (255, 0, 0), 2)
            
            # Draw scaled bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            cv2.rectangle(original_img_rgb, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
        
        results['original_scale_viz'] = original_img_rgb
        results['scale_factors'] = (scale_x, scale_y)
    
    return results

def create_overlay_visualization(original_image, pred_mask_bin, alpha=0.4):
    """Simple overlay for backward compatibility"""
    # Convert to numpy array
    img_array = np.array(original_image)
    
    # Create colored overlay for cancer regions
    overlay = np.zeros_like(img_array)
    overlay[pred_mask_bin > 0] = [255, 0, 0]  # Red color for cancer regions
    
    # Blend original image with overlay
    result = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
    
    return result

def analyze_prediction(pred_mask_bin):
    """Analyze prediction results"""
    cancer_pixels = np.sum(pred_mask_bin > 0)
    total_pixels = pred_mask_bin.shape[0] * pred_mask_bin.shape[1]
    cancer_percentage = (cancer_pixels / total_pixels) * 100
    
    has_cancer = cancer_pixels > 0
    
    return has_cancer, cancer_percentage, cancer_pixels

def main():
    st.title("🦴 Bone Cancer Detection System")
    st.markdown("Upload an X-ray image to detect potential bone cancer regions using AI-powered segmentation.")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a U-Net deep learning model to:
        - Detect bone cancer regions in X-ray images
        - Highlight suspicious areas in red
        - Provide confidence metrics
        
        **Model Details:**
        - Architecture: Improved U-Net
        - Input Size: 224x224 pixels
        - Optimal Threshold: 0.001
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for research purposes only. 
        Always consult with medical professionals 
        for proper diagnosis and treatment.
        """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Unable to load the model. Please ensure '__improved_best_unet_segmentation.h5' is in the current directory.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original X-ray")
            st.image(original_image, caption="Uploaded X-ray Image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing image..."):
                # Preprocess image with exact notebook implementation
                input_arr, resized_img, original_dims = preprocess_image(original_image)
                
                # Make prediction with proper threshold
                pred_mask, pred_mask_bin = predict_cancer(model, input_arr, threshold=THRESHOLD)
                
                # Create advanced visualization matching notebook
                viz_results = create_advanced_visualization(resized_img, pred_mask_bin, original_dims)
                
                # Analyze results
                has_cancer, cancer_percentage, cancer_pixels = analyze_prediction(pred_mask_bin)
                
                # Display results
                if has_cancer:
                    st.error(f"⚠️ **Potential cancer regions detected!**")
                    st.metric("Cancer Coverage", f"{cancer_percentage:.2f}%")
                    st.metric("Affected Pixels", f"{cancer_pixels:,}")
                else:
                    st.success("✅ **No cancer regions detected**")
                    st.metric("Cancer Coverage", "0.00%")
        
        # Visualization section
        st.subheader("🎯 Detailed Analysis")
        
        # Create tabs for different visualizations (matching notebook)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Advanced Segmentation", 
            "Probability Heatmap", 
            "Binary Mask", 
            "Original Scale", 
            "Technical Analysis"
        ])
        
        with tab1:
            # Advanced segmentation with contours and bounding boxes
            st.subheader("🎯 Segmentation with Contours & Bounding Boxes")
            if has_cancer and viz_results['num_detections'] > 0:
                st.image(viz_results['visualization'], 
                        caption=f"Detected {viz_results['num_detections']} potential cancer region(s) - Red: contours, Green: bounding boxes", 
                        use_column_width=True)
                st.info(f"✅ Found {viz_results['num_detections']} cancer region(s) with contour area > 10 pixels")
            else:
                st.image(resized_img, caption="No significant cancer regions detected", use_column_width=True)
                st.success("✅ No cancer regions detected above threshold")
        
        with tab2:
            # Probability heatmap matching notebook (using 'jet' colormap)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(pred_mask, cmap='jet', vmin=0, vmax=np.max(pred_mask) if np.max(pred_mask) > 0 else 1)
            ax.set_title('Cancer Probability Heatmap')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            # Binary mask with threshold info
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(pred_mask_bin, cmap='gray')
            ax.set_title(f'Binary Mask (threshold={THRESHOLD})')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with tab4:
            # Original scale visualization
            if 'original_scale_viz' in viz_results:
                st.subheader("🔍 Detection on Original Image Scale")
                st.image(viz_results['original_scale_viz'], 
                        caption="Cancer detection scaled to original image dimensions", 
                        use_column_width=True)
                scale_x, scale_y = viz_results['scale_factors']
                st.info(f"Scaling factors: X={scale_x:.2f}, Y={scale_y:.2f}")
            else:
                st.image(original_image, caption="Original image (no scaling needed)", use_column_width=True)
        
        with tab5:
            # Technical analysis matching notebook layout
            st.subheader("📊 Technical Analysis")
            
            # Side by side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original
            ax1.imshow(resized_img)
            ax1.set_title('Original X-ray (Resized)')
            ax1.axis('off')
            
            # Advanced visualization
            if has_cancer and viz_results['num_detections'] > 0:
                ax2.imshow(viz_results['visualization'])
                ax2.set_title(f'Predicted Segmentation ({viz_results["num_detections"]} regions)')
            else:
                ax2.imshow(resized_img)
                ax2.set_title('No Cancer Detected')
            ax2.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Additional metrics
        with st.expander("📊 Technical Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image Dimensions", f"{original_image.size[0]} x {original_image.size[1]}")
            with col2:
                st.metric("Model Input Size", f"{IMG_SIZE} x {IMG_SIZE}")
            with col3:
                st.metric("Detection Threshold", f"{THRESHOLD}")
            
            st.markdown("**Prediction Confidence Distribution:**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(pred_mask.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Prediction Probabilities')
            ax.legend()
            st.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()