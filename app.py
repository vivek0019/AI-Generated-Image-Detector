import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page configuration
st.set_page_config(
    page_title="AI Image Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .section-container {
        background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #ffffff;
        display: inline-block;
    }
    
    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #0f1f0f;
        border-left: 4px solid #4ade80;
        color: #4ade80;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .model-card {
        background-color: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    .prediction-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .probability-text {
        color: #a0a0a0;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    .final-result {
        background: linear-gradient(145deg, #ffffff, #e0e0e0);
        color: #0a0a0a;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(255, 255, 255, 0.2);
    }
    
    .final-result-label {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .final-result-prob {
        font-size: 1.3rem;
        color: #404040;
        font-weight: 500;
    }
    
    .image-container {
        background-color: #1a1a1a;
        border: 2px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .image-label {
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a2a2a, transparent);
        margin: 2rem 0;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #141414;
        border: 2px dashed #404040;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #ffffff;
        background-color: #1a1a1a;
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .uploadedFile {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }
    
    .stProgress > div > div {
        background-color: #ffffff;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    .step-indicator {
        display: inline-block;
        background-color: #ffffff;
        color: #0a0a0a;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 0.75rem;
    }
    
    .metric-container {
        background-color: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">AI Generated Image Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ensemble Model Classification with GradCAM Visualization</p>', unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load model function
@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()
# Main layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)

    st.markdown(
        '<p class="section-title"><span class="step-indicator">1</span>Model Information</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="success-box">✓ ResNet18 model loaded successfully</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-card">📦 Model : ResNet18 Binary Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="step-indicator">2</span>Upload Image</p>', unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Select an image to analyze",
        type=["jpg","png","jpeg"],
        help="Upload an image for authenticity detection"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results section
if uploaded_image:
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<p class="image-label">📷 Uploaded Image</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    label_map = {0:"Fake",1:"Real"}
    
   
    
with st.spinner('Analyzing image...'):

    with torch.no_grad():
        outputs = model(input_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        ).cpu().numpy()[0]

    pred = np.argmax(probs)

    label_map = {
        0: "Fake",
        1: "Real"
    }

    final_label = label_map[pred]
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Prediction Result
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

with col_res2:
    st.markdown(f"""
    <div class="final-result">
        <div class="final-result-label">{final_label}</div>
        <div class="final-result-prob">
            Real: {probs[1]:.1%} | Fake: {probs[0]:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Metrics
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.metric(
        "Prediction",
        final_label
    )

with col_m2:
    st.metric(
        "Confidence",
        f"{max(probs):.1%}"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# GradCAM Heatmap
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown(
    '<p class="section-title">GradCAM Heatmap Visualization</p>',
    unsafe_allow_html=True
)

with st.spinner('Generating visualization...'):

    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(pred)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]

    img_np = np.array(
        image.resize((224,224))
    ) / 255.0

    visualization = show_cam_on_image(
        img_np,
        grayscale_cam,
        use_rgb=True
    )

col_viz1, col_viz2, col_viz3 = st.columns([1,2,1])

with col_viz2:

    st.markdown(
        '<div class="image-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="image-label">🔥 Attention Heatmap</p>',
        unsafe_allow_html=True
    )

    st.image(
        visualization,
        use_container_width=True
    )

    st.markdown(
        '''
        <p style="text-align:center;
        color:#a0a0a0;
        font-size:0.9rem;
        margin-top:1rem;">
        Red areas indicate regions the model focused on
        for classification
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_image:

    st.markdown(
        '''
        <div class="info-box">
        Upload an image to begin analysis
        </div>
        ''',
        unsafe_allow_html=True
    )

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: #404040; font-size: 0.85rem; padding: 2rem 0;">
    <p>AI Generated Image Detection System | Device: {device.type.upper()}</p>
</div>
""", unsafe_allow_html=True)
