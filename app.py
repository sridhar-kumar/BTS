import streamlit as st
import numpy as np
import torch
from PIL import Image
import time
from streamlit.components.v1 import html

from model import load_trained_model
from inference import create_mask, overlay_image, tumor_stats
from report import generate_pdf
from storage import save_case, load_cases
from gradcam import real_gradcam

# ================= CONFIG =================
st.set_page_config(page_title="Brain Tumor Clinical System", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>

.block-container { padding-top: 2rem; }

.title { font-size:38px; font-weight:700; }
.subtitle { opacity:0.6; margin-bottom:20px; }

.card {
    padding: 25px;
    border-radius: 18px;
    background: #e7f0ff;
    border: 1px solid #c7d9ff;
    margin-bottom: 10px;
}

.card-gap {
    margin-top:15px;
    margin-bottom:25px;
}

.small-box {
    background:#fff;
    padding:15px;
    border-radius:14px;
    border:1px solid #e5e7eb;
    margin-bottom:15px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ================= MODEL =================
@st.cache_resource
def get_model():
    return load_trained_model()

# ================= GAUGE =================
def show_gauge(percent, risk, color):

    progress = percent * 2.2

    gauge_html = f"""
    <html>
    <body style="margin:0; text-align:center;">

    <svg width="160" height="90">
        <path d="M10 80 A70 70 0 0 1 150 80"
        stroke="#e5e7eb" stroke-width="12" fill="none"
        stroke-linecap="round"/>

        <path d="M10 80 A70 70 0 0 1 150 80"
        stroke="{color}" stroke-width="12" fill="none"
        stroke-linecap="round"
        stroke-dasharray="{progress},300"/>
    </svg>

    <div style="margin-top:-10px;">
        <h3 style="color:{color}; margin:0;">{risk}</h3>
        <p style="margin:0;">{percent:.2f}%</p>
    </div>

    </body>
    </html>
    """

    html(gauge_html, height=120)

# ================= SESSION =================
if "role" not in st.session_state:
    st.session_state.role = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ================= HOME =================
if st.session_state.role is None:

    st.markdown("<div class='title'>🧠 Brain Tumor Clinical System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-Powered Radiology Assistant</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🏥 Doctor Portal</h3>
            <p>Clinical MRI Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='card-gap'>", unsafe_allow_html=True)
        if st.button("Enter Doctor Mode"):
            st.session_state.role = "doctor"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h3>⚕ Patient Screening</h3>
            <p>Quick Abnormality Detection</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='card-gap'>", unsafe_allow_html=True)
        if st.button("Start Screening"):
            st.session_state.role = "patient"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ================= LOGIN =================
elif st.session_state.role == "doctor" and not st.session_state.authenticated:

    st.markdown("## Doctor Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "doctor" and pwd == "123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back"):
        st.session_state.role = None
        st.rerun()

    st.stop()

# ================= DOCTOR =================
elif st.session_state.role == "doctor":

    st.sidebar.title("Doctor Portal")
    page = st.sidebar.radio("Menu", ["Analyzer","Case History"])

    if st.sidebar.button("Logout"):
        st.session_state.role = None
        st.session_state.authenticated = False
        st.rerun()

    if page == "Analyzer":

        st.markdown("## MRI Analyzer")

        files = st.file_uploader("Upload MRI Images", accept_multiple_files=True)
        patient_id = st.text_input("Patient ID")

        if files:
            model = get_model()

            for file in files:

                image = Image.open(file).convert("L")
                img = np.array(image.resize((256,256)))

                tensor = torch.tensor(img/255.0).float().unsqueeze(0).unsqueeze(0)

                start = time.time()
                with torch.no_grad():
                    pred = model(tensor).numpy()[0,0]
                t = round(time.time()-start,3)

                mask = create_mask(pred)
                overlay = overlay_image(img, mask)
                pixels, percent, _ = tumor_stats(mask)
                heatmap = real_gradcam(model, tensor, img)

                # ===== RISK =====
                if percent >= 15:
                    risk = "High"
                    color = "#ef4444"
                elif percent >= 5:
                    risk = "Medium"
                    color = "#f59e0b"
                else:
                    risk = "Low"
                    color = "#22c55e"

                # ===== IMAGES =====
                c1,c2,c3,c4 = st.columns(4)
                c1.image(img, caption="MRI")
                c2.image(mask, caption="Mask")
                c3.image(overlay, caption="Overlay")
                c4.image(heatmap, caption="Heatmap")

                # ===== METRICS =====
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class='small-box'>
                        <h4>Tumor Percentage</h4>
                        <h2>{percent:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div class='small-box'>", unsafe_allow_html=True)
                    st.markdown("<h4>Risk Level</h4>", unsafe_allow_html=True)
                    show_gauge(percent, risk, color)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class='small-box'>
                        <h4>Tumor Pixels</h4>
                        <h2>{pixels}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='small-box'>
                        <h4>Analysis Time</h4>
                        <h2>{t}s</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== PDF =====
                pdf = generate_pdf(
                    patient_id,
                    percent,
                    percent,
                    mri=img,
                    heatmap=heatmap
                )

                c1, c2 = st.columns(2)

                c1.download_button(
                    "⬇ Download Report (PDF)",
                    data=pdf,
                    file_name=f"brain_report_{patient_id}.pdf",
                    mime="application/pdf"
                )

                if c2.button("💾 Save Case"):
                    save_case(patient_id, {"tumor": percent})

    else:
        st.markdown("## Case History")
        cases = load_cases()
        for c in cases:
            st.write(c)

# ================= PATIENT =================
elif st.session_state.role == "patient":

    st.markdown("## Patient Screening")

    file = st.file_uploader("Upload MRI")

    if file:
        model = get_model()

        image = Image.open(file).convert("L")
        img = np.array(image.resize((256,256)))

        tensor = torch.tensor(img/255.0).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(tensor).numpy()[0,0]

        mask = create_mask(pred)
        _, percent, _ = tumor_stats(mask)

        if percent >= 3:
           st.warning("Mild irregular patterns detected. It's recommended to consult a medical professional for further evaluation.")
        else:
           st.success("No significant abnormalities detected. Everything looks healthy.")
           st.caption("You can relax — but always maintain regular health checkups.")
    if st.button("⬅ Back"):
        st.session_state.role = None
        st.rerun()