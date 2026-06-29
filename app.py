import time
import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Generated Image Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:#0a0a0a;
    color:white;
}

.main-title{
    font-size:3rem;
    font-weight:700;
    text-align:center;
    color:white;
    margin-bottom:0.5rem;
}

.subtitle{
    text-align:center;
    color:#a0a0a0;
    font-size:1.1rem;
    margin-bottom:3rem;
}

.section-container{
    background:linear-gradient(145deg,#1a1a1a,#0f0f0f);
    border:1px solid #2a2a2a;
    border-radius:16px;
    padding:2rem;
    margin:1.5rem 0;
}

.section-title{
    font-size:1.4rem;
    font-weight:600;
    margin-bottom:1.5rem;
    border-bottom:2px solid white;
    display:inline-block;
    color:white;
}

.image-container{
    background:#1a1a1a;
    border-radius:15px;
    padding:20px;
    border:1px solid #2a2a2a;
}

.image-label{
    text-align:center;
    font-weight:600;
    margin-bottom:15px;
}

.final-result{
    background:linear-gradient(145deg,#ffffff,#e0e0e0);
    color:black;
    border-radius:18px;
    padding:35px;
    text-align:center;
}

.final-result-label{
    font-size:3rem;
    font-weight:700;
}

.final-result-prob{
    font-size:1.2rem;
}

.metric-container{
    background:#141414;
    border:1px solid #2a2a2a;
    border-radius:12px;
    padding:20px;
    text-align:center;
}

.divider{
    height:1px;
    background:#2a2a2a;
    margin:30px 0;
}

.info-box{
    background:#1a1a1a;
    border-left:5px solid white;
    padding:15px;
    border-radius:10px;
}

.success-box{
    background:#102510;
    border-left:5px solid #4ade80;
    color:#4ade80;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown(
    '<h1 class="main-title">AI Generated Image Detection</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">ResNet-18 Classification with GradCAM Visualization</p>',
    unsafe_allow_html=True
)

# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -------------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():

    model=models.resnet18(weights=None)

    model.fc=nn.Linear(
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

model=load_model()

with st.sidebar:

    st.title("About")

    st.markdown("""
### AI Generated Image Detection

This application detects whether an uploaded image is:

- 📷 Real
- 🤖 AI Generated

### Model Information

**Architecture:** ResNet-18

**Framework:** PyTorch

**Dataset Size:** 21,000 Images

**Input Resolution:** 224 × 224

**Explainability:** Grad-CAM

### Workflow

1. Upload Image

2. AI Prediction

3. Confidence Score

4. Grad-CAM Visualization
""")
# -------------------------------------------------------
# IMAGE UPLOAD SECTION
# -------------------------------------------------------

st.markdown('<div class="section-container">', unsafe_allow_html=True)

st.markdown(
    '<p class="section-title">Upload Image</p>',
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

st.info("Supported formats: JPG • JPEG • PNG")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# RESULT SECTION
# -------------------------------------------------------

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.markdown(
            '<div class="image-container">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="image-label">Uploaded Image</p>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------------------

    input_tensor = transform(image).unsqueeze(0).to(device)

    label_map = {
        0: "Fake",
        1: "Real"
    }

    # ---------------------------------------------------
    # MODEL INFERENCE
    # ---------------------------------------------------

    start_time = time.time()

    with st.spinner("Analyzing Image..."):

        with torch.no_grad():

            outputs = model(input_tensor)

            probs = torch.softmax(
                outputs,
                dim=1
            ).cpu().numpy()[0]

    end_time = time.time()

    inference_time = end_time - start_time

    pred = np.argmax(probs)

    final_label = label_map[pred]

    confidence = float(np.max(probs))

    # ---------------------------------------------------
    # RESULT TITLE
    # ---------------------------------------------------

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="section-title">Prediction Result</p>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------
    # SUCCESS / ERROR MESSAGE
    # ---------------------------------------------------

    if final_label == "Real":
        st.success(f"Prediction: {final_label}")
    else:
        st.error(f"Prediction: {final_label}")

    # ---------------------------------------------------
    # RESULT CARD
    # ---------------------------------------------------

    c1, c2, c3 = st.columns([1,2,1])

    with c2:

        result_html = (
            '<div class="final-result">'
            '<div class="final-result-label">' + final_label + '</div>'
            '<div class="final-result-prob">'
            'Confidence: ' + f"{confidence:.2%}" + '<br><br>'
            'Real: ' + f"{probs[1]:.2%}" + '<br>'
            'Fake: ' + f"{probs[0]:.2%}"
            '</div>'
            '</div>'
        )

        st.markdown(result_html, unsafe_allow_html=True)

    # ---------------------------------------------------
    # CONFIDENCE BAR
    # ---------------------------------------------------

    st.write("### Confidence")

    st.progress(confidence)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Prediction",
            final_label
        )

    with metric2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    with metric3:

        st.metric(
            "Inference Time",
            f"{inference_time:.3f} sec"
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # -------------------------------------------------------
    # GRAD-CAM VISUALIZATION
    # -------------------------------------------------------

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="section-title">Grad-CAM Visualization</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating Grad-CAM..."):

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
        )[0]

        img_np = np.array(
            image.resize((224,224)),
            dtype=np.float32
        ) / 255.0

        visualization = show_cam_on_image(
            img_np,
            grayscale_cam,
            use_rgb=True
        )

    # -------------------------------------------------------
    # ORIGINAL + HEATMAP
    # -------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Grad-CAM Heatmap"
        )

        st.image(
            visualization,
            use_container_width=True
        )

    st.caption(
        "The highlighted regions indicate the areas that most influenced the model's prediction."
    )

    # -------------------------------------------------------
    # DOWNLOAD BUTTON
    # -------------------------------------------------------

    output = Image.fromarray(visualization)

    buffer = BytesIO()

    output.save(
        buffer,
        format="PNG"
    )

    st.download_button(

        label="📥 Download Grad-CAM",

        data=buffer.getvalue(),

        file_name="gradcam_visualization.png",

        mime="image/png"

    )

    # -------------------------------------------------------
    # EXPLANATION
    # -------------------------------------------------------

    with st.expander("How does Grad-CAM work?"):

        st.write("""

Grad-CAM (Gradient-weighted Class Activation Mapping) helps explain deep learning predictions.

Instead of treating the model as a black box, it highlights image regions that contributed the most toward the final prediction.

• Red / Yellow → High contribution

• Green → Moderate contribution

• Blue → Low contribution

This visualization improves the interpretability of the AI model.

""")

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
<div class="info-box">
📤 Upload an image to begin AI image analysis.
</div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div style="text-align:center;color:#808080;padding:20px;font-size:15px;">

<b>AI Generated Image Detection System</b>

<br><br>

Powered by

PyTorch • ResNet-18 • Grad-CAM • Streamlit

<br><br>

Running on <b>{device.type.upper()}</b>

<br><br>

Developed by <b>Vivek Potdar</b>

</div>
""",
    unsafe_allow_html=True
)
