import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
from io import BytesIO
from datetime import datetime
from functools import lru_cache


import base64

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import requests


# ============================================================
#              LOAD MODELS (CACHED)
# ============================================================

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        "/Users/akashngowda/Desktop/dental/diseas fresh/oral_diseases_model.keras"
    )


@st.cache_resource
def load_classes():
    with open("/Users/akashngowda/Desktop/dental/diseas fresh/class_indices.json") as f:
        classes = json.load(f)
        return {int(k): v for k, v in classes.items()}


@st.cache_resource
def load_yolo_model():
    return YOLO("/Users/akashngowda/Desktop/dental/dataset/archive/best.pt")


cnn_model = load_cnn_model()
class_names = load_classes()
yolo_model = load_yolo_model()


# ============================================================
#              PREDICTION FUNCTIONS
# ============================================================

def predict_cnn(pil_img, img_size=300):
    img = pil_img.resize((img_size, img_size))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = cnn_model.predict(arr)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100
    return class_names[class_id], confidence


def predict_yolo(uploaded_file):
    """Run YOLO on a Streamlit UploadedFile and return annotated image + findings."""
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        temp_path = tmp.name

    # YOLO prediction
    results = yolo_model(temp_path)[0]

    # Read & convert image
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    findings = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)
        label = f"{yolo_model.names[cls]} ({conf * 100:.1f}%)"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )

        findings.append(
            {
                "class": yolo_model.names[cls],
                "confidence": round(conf * 100, 2),
                "box": (x1, y1, x2, y2),
            }
        )

    annotated_pil = Image.fromarray(img)

    summary = (
        "No abnormalities detected."
        if not findings
        else "\n".join(
            ["Detected Findings:"]
            + [
                f"- {f['class']}  |  {f['confidence']}%  |  Box: {f['box']}"
                for f in findings
            ]
        )
    )

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return annotated_pil, findings, summary


# ============================================================
#              SMART AUTO-DETECTION
# ============================================================

def is_xray_image(pil_img):
    """
    Heuristic: low colour saturation → likely X-ray.
    Tuned to avoid misclassifying coloured photos with UI overlays.
    """
    img = pil_img.convert("RGB").resize((256, 256))
    arr = np.array(img).astype("float32") / 255.0

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    s = (maxc - minc) / (maxc + 1e-6)

    # Ignore edges (often coloured frames/overlays)
    s_center = s[32:-32, 32:-32]

    frac_high_sat = (s_center > 0.25).mean()
    return frac_high_sat < 0.06  # slightly stricter for X-ray detection



# ============================================================
#              GEMINI EXPLAINABLE AI (XAI) - FIXED WORKING VERSION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")

GEMINI_MODEL = "models/gemini-2.5-flash"   # ✅ guaranteed working model
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)


def call_gemini(prompt_text):
    if not GEMINI_API_KEY:
        return "Gemini API key missing."

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    try:
        r = requests.post(
            GEMINI_ENDPOINT,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()

        # Parse returned text safely
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "Explainable-AI summary unavailable. (Parsing error)"

    except Exception as e:
        return f"Explainable-AI summary unavailable. (API error: {str(e)})"


def generate_explanation(prediction_text):
    """Generate simple XAI clinical insights."""
    prompt = f"""
    You are a senior dental specialist. Based on this AI prediction:

    {prediction_text}

    Provide EXACTLY three short bullet points explaining:
    • What this finding means clinically
    • Why it matters for oral health
    • What dentists usually recommend next

    Keep it short and simple.
    """

    return call_gemini(prompt)


def wrap_text(c, text, x, y, max_width, lh=14):
    from textwrap import wrap
    c.setFont("Helvetica", 11)
    for line in wrap(text, width=90):
        c.drawString(x, y, line)
        y -= lh
    return y


def draw_pil(c, pil_img, x, y, max_w, max_h):
    if pil_img is None:
        return
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h)

    buf = BytesIO()
    pil_img.save(buf, "PNG")
    buf.seek(0)

    c.drawImage(ImageReader(buf), x, y, w * scale, h * scale)


def generate_pdf(name, age, symptoms, report_type, result_text, explanation_text, original_img, annotated_img):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50

    # -------------------------
    # HEADER
    # -------------------------
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, y, "Dental AI Diagnostic Report")
    y -= 40

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25
    c.drawString(50, y, f"Patient Name: {name}")
    y -= 18
    c.drawString(50, y, f"Age: {age}")
    y -= 18
    c.drawString(50, y, f"Report Type: {report_type}")
    y -= 30

    # -------------------------
    # SYMPTOMS
    # -------------------------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Symptoms & Notes:")
    y -= 20
    y = wrap_text(c, symptoms, 50, y, 500)
    y -= 30

    # -------------------------
    # AI RAW RESULTS
    # -------------------------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "AI Detection Results:")
    y -= 20
    y = wrap_text(c, result_text, 50, y, 500)
    y -= 30

    # -------------------------
    # XAI INTERPRETATION
    # Convert Gemini long text to clean bullet sections
    # -------------------------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Clinical Interpretation:")
    y -= 25

    # Split into lines cleaned up for PDF readability
    cleaned = explanation_text.replace("*", "").replace("•", "").split("\n")
    for line in cleaned:
        if line.strip() == "":
            continue

        # Make headings bold
        if ":" not in line:
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 11)

        y = wrap_text(c, line.strip(), 50, y, 500)
        y -= 4

        if y < 100:
            c.showPage()
            y = h - 60

    # -------------------------
    # PAGE 2 — IMAGES SIDE BY SIDE
    # -------------------------
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, h - 40, "Image Analysis")

    # Image sizes
    IMG_W = 240
    IMG_H = 240

    # Original
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, h - 80, "Original Image:")
    draw_pil(c, original_img, 50, h - 350, IMG_W, IMG_H)

    # Annotated (if exists)
    if annotated_img:
        c.drawString(330, h - 80, "YOLO Annotated Image:")
        draw_pil(c, annotated_img, 310, h - 350, IMG_W, IMG_H)

    c.save()
    buf.seek(0)
    # ---- Convert PDF to base64 for MongoDB storage ----
    import base64
    pdf_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return buf


# ============================================================
#              STREAMLIT UI CONFIG
# ============================================================

st.set_page_config(
    page_title="Integrated Dental AI Diagnostic System",
    layout="wide",
    page_icon="🦷",
)

# ------------------------ Sidebar ------------------------ #

with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;'>🦷 Dental AI</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:13px;'>AI-powered analysis of dental photos and X-rays.</p>",
        unsafe_allow_html=True,
    )

    theme = st.radio("Theme", ["Dark", "Light"], index=0, help="Switch UI theme")

    st.markdown("---")
    st.markdown("**Modes**")
    st.markdown("• CNN – real-world photos")
    st.markdown("• YOLO – X-ray detection")
    st.markdown("• Auto – chooses model for you")
    st.markdown("• Camera – live capture for CNN")

    st.markdown("---")
    st.markdown(
        "<small>Tip: Use high-quality images for best results.</small>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🔍 Test Gemini API")
    if st.button("Run API Connectivity Test"):
        try:
            test_payload = {"contents": [{"parts": [{"text": "hello"}]}]}
            r = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json=test_payload,
                timeout=30,
            )
            st.write("Endpoint:", GEMINI_ENDPOINT)
            st.write("Status Code:", r.status_code)
            try:
                st.write("Response:", r.json())
            except:
                st.write("Raw Response:", r.text)
        except Exception as e:
            st.error(f"Test Failed: {e}")

# ------------------------ Global CSS ------------------------ #

# Theme colors
if theme == "Dark":
    TEXT_COLOR = "#ffffff"
    CARD_BG = "rgba(255,255,255,0.95)"   # Light card on dark background
    CARD_TEXT = "#0f172a"               # Dark text INSIDE cards (fix)
    BORDER_COLOR = "#22c55e"
else:
    TEXT_COLOR = "#0f172a"
    CARD_BG = "#ffffff"
    CARD_TEXT = "#0f172a"
    BORDER_COLOR = "#3b82f6"

st.markdown(
    f"""
<style>

/* ========= GLOBAL BACKGROUND ========== */
.stApp {{
    {"background: linear-gradient(-45deg,#050816,#111827,#020617,#0f172a) !important;" if theme=="Dark" else "background: linear-gradient(135deg,#ecfdf5,#e0f2fe,#fef9c3) !important;"}
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}}

@keyframes gradientMove {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

/* ========= GLOBAL TEXT COLOR ========= */
h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_COLOR} !important;
}}

/* Do NOT force global color on all p/div/span – allows proper card text colors */
body {{
    color: {TEXT_COLOR} !important;
}}

/* ========= HEADER BLOCK ========== */
.main-block {{
    background: {"rgba(255,255,255,0.12)" if theme=="Dark" else "rgba(255,255,255,0.65)"};
    border-radius: 18px;
    padding: 22px 26px 30px 26px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}}

/* ========= MODEL CARDS ========== */
.model-card {{
    background: {CARD_BG} !important;
    border-radius: 18px;
    padding: 22px 22px 18px 22px;
    border-left: 6px solid {BORDER_COLOR};
    box-shadow: 0 10px 25px rgba(15,23,42,0.18);
    transition: transform .18s ease, box-shadow .18s ease;
    color: {CARD_TEXT} !important;
}}

.model-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(15,23,42,0.30);
}}

.model-title {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
    color: {CARD_TEXT} !important;
}}

.model-desc {{
    font-size: 14px;
    opacity: 0.85;
    color: {CARD_TEXT} !important;
}}

/* ========= TAB + FOOTER ========= */
.footer {{
    text-align:center;
    font-size:12px;
    opacity:0.70;
    margin-top: 18px;
    color:{TEXT_COLOR} !important;
}}

</style>
""",
    unsafe_allow_html=True,
)
# ============================================================
#              HEADER
# ============================================================

st.markdown(
    """
<div class="main-block">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1 style="margin-bottom:4px;">🦷 Integrated Dental AI Diagnostic System</h1>
      <p style="font-size:14px; opacity:0.85;">
        Advanced multi-model analysis for real-world dental images and radiographs.
      </p>
    </div>
    <div style="font-size:12px; text-align:right; opacity:0.9;">
      <b>Status:</b> Models loaded and ready ✅<br/>
      <b>Mode:</b> {mode}
    </div>
  </div>
</div>
""".format(
        mode=theme + " theme"
    ),
    unsafe_allow_html=True,
)

# ============================================================
#              LAYOUT TABS
# ============================================================

tab_analysis, tab_report, tab_dashboard = st.tabs(["🧪 Analysis Workspace", "📄 Patient Report", "📚 Patient Records"])

# ============================================================
#              ANALYSIS TAB
# ============================================================

with tab_analysis:
    st.markdown("<h3>AI Models Available</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
        <div class="model-card">
          <div class="model-title">⚙️ Disease Classification (CNN)</div>
          <div class="model-desc">
            Uses real-world intra-oral photos to detect conditions such as calculus,
            caries, gingivitis, ulcers and discoloration.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
        <div class="model-card">
          <div class="model-title">🖥️ X-ray Detection (YOLO)</div>
          <div class="model-desc">
            Detects fillings, implants, crowns, root-canal treatments and impacted teeth
            in dental radiographs.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
        <div class="model-card">
          <div class="model-title">🤖 Auto Detect (Smart)</div>
          <div class="model-desc">
            Analyses the image first and automatically routes it to the correct model
            (CNN for photos, YOLO for X-rays).
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- model selection ---
    st.subheader("Select Analysis Mode")
    selected_model = st.selectbox(
        "Choose how you want to analyse the image:",
        [
            "Disease Classification (CNN)",
            "X-ray Detection (YOLO)",
            "Auto Detect (Smart)",
            "Real-Time Camera (CNN)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("")

    # --- upload vs camera ---
    uploaded_file = None
    camera_file = None

    if selected_model == "Real-Time Camera (CNN)":
        st.markdown("### 📷 Real-time Webcam Capture")
        st.write(
            "Use your device camera to capture a live intra-oral photo. "
            "Once captured, click **Analyze Captured Image**."
        )
        camera_file = st.camera_input("Live camera preview", label_visibility="collapsed")
    else:
        st.markdown("### 📤 Upload Dental Image")
        uploaded_file = st.file_uploader(
            "Upload a dental photo or X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

    # --------------------------------------------------------
    #           PROCESS & DISPLAY RESULTS
    # --------------------------------------------------------
    source_file = camera_file if camera_file is not None else uploaded_file

    if source_file is not None:
        src_img = Image.open(source_file).convert("RGB")

        st.markdown("### 🖼 Input Image")
        st.image(src_img, caption="Uploaded / Captured Image", width="stretch")

        # CNN mode
        if selected_model == "Disease Classification (CNN)":
            if st.button("🔍 Analyze with CNN", use_container_width=True):
                with st.spinner("Running CNN disease classifier…"):
                    label, conf = predict_cnn(src_img)
                st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.2f}%**")

                st.session_state.report_type = "CNN Disease Classification"
                st.session_state.text = f"Disease: {label}\nConfidence: {conf:.2f}%"
                st.session_state.explanation = generate_explanation(st.session_state.text)
                st.session_state.original = src_img
                st.session_state.annotated = None

        # YOLO mode
        elif selected_model == "X-ray Detection (YOLO)":
            if st.button("🩻 Analyze X-ray with YOLO", use_container_width=True):
                with st.spinner("Running YOLO radiograph detector…"):
                    annotated, finds, summary = predict_yolo(source_file)
                st.success("YOLO detection complete ✅")
                st.image(
                    annotated,
                    caption="YOLO Annotated X-ray",
                    width="stretch",
                )
                st.text(summary)

                st.session_state.report_type = "YOLO X-ray Detection"
                st.session_state.text = summary
                st.session_state.explanation = generate_explanation(st.session_state.text)
                st.session_state.original = src_img
                st.session_state.annotated = annotated

        # Auto detect
        elif selected_model == "Auto Detect (Smart)":
            if st.button("🤖 Auto Analyze", use_container_width=True):
                with st.spinner("Detecting image type and running appropriate model…"):
                    xray_flag = is_xray_image(src_img)

                if xray_flag:
                    st.info("Detected **X-ray** ➜ running **YOLO** model.")
                    with st.spinner("Running YOLO radiograph detector…"):
                        annotated, finds, summary = predict_yolo(source_file)
                    st.success("YOLO detection complete ✅")
                    st.image(
                        annotated,
                        caption="YOLO Annotated X-ray",
                        width="stretch",
                    )
                    st.text(summary)

                    st.session_state.report_type = "Smart Auto Detect (YOLO)"
                    st.session_state.text = summary
                    st.session_state.explanation = generate_explanation(st.session_state.text)
                    st.session_state.original = src_img
                    st.session_state.annotated = annotated
                else:
                    st.info("Detected **real-world photo** ➜ running **CNN** model.")
                    with st.spinner("Running CNN disease classifier…"):
                        label, conf = predict_cnn(src_img)
                    st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.2f}%**")

                    st.session_state.report_type = "Smart Auto Detect (CNN)"
                    st.session_state.text = (
                        f"Disease: {label}\nConfidence: {conf:.2f}%"
                    )
                    st.session_state.explanation = generate_explanation(st.session_state.text)
                    st.session_state.original = src_img
                    st.session_state.annotated = None

        # Camera + CNN
        elif selected_model == "Real-Time Camera (CNN)" and camera_file is not None:
            if st.button("📸 Analyze Captured Image", use_container_width=True):
                with st.spinner("Running CNN disease classifier on camera frame…"):
                    label, conf = predict_cnn(src_img)
                st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.2f}%**")

                st.session_state.report_type = "Camera-Based CNN Diagnosis"
                st.session_state.text = f"Disease: {label}\nConfidence: {conf:.2f}%"
                st.session_state.explanation = generate_explanation(st.session_state.text)
                st.session_state.original = src_img
                st.session_state.annotated = None

    else:
        st.info("Upload an image or capture one with the camera to begin analysis.")


# ============================================================
#              REPORT TAB
# ============================================================

with tab_report:
    st.subheader("Patient Details & PDF Report")

    if "report_type" in st.session_state:
        st.markdown("### 🧠 Explainable AI Summary")
        st.write(st.session_state.explanation)

        with st.form("patient_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                name = st.text_input("Patient Name")
            with col_b:
                age = st.number_input("Age", min_value=1, max_value=120, step=1)

            symptoms = st.text_area("Symptoms / Clinical Notes")

            generate_clicked = st.form_submit_button("📄 Generate PDF Report")

        if generate_clicked:
            with st.spinner("Generating PDF report…"):
                pdf_bytes = generate_pdf(
                    name=name,
                    age=age,
                    symptoms=symptoms,
                    report_type=st.session_state.report_type,
                    result_text=st.session_state.text,
                    explanation_text=st.session_state.explanation,
                    original_img=st.session_state.original,
                    annotated_img=st.session_state.annotated,
                )

            # -----------------------------
            # SAVE REPORT TO MONGODB ATLAS
            # -----------------------------
            import base64, pymongo, os

            MONGO_URI = "YOUR_URL"

            try:
                client = pymongo.MongoClient(
                    MONGO_URI,
                    serverSelectionTimeoutMS=15000,
                    connectTimeoutMS=15000,
                    tls=True,
                    tlsAllowInvalidCertificates=True
                )
                client.server_info()
                db = client["DentalAI"]
                reports = db["PatientReports"]
                st.success("MongoDB Connected Successfully ✔️")
            except Exception as e:
                st.error(f"MongoDB Connection Failed ❌\n{e}")
                reports = None

            report_doc = {
                "name": name,
                "age": age,
                "symptoms": symptoms,
                "report_type": st.session_state.report_type,
                "ai_results": st.session_state.text,
                "xai_summary": st.session_state.explanation,
                "timestamp": datetime.now().isoformat(),
                "pdf_base64": base64.b64encode(pdf_bytes.getvalue()).decode("utf-8")
            }

            if reports is not None:
                try:
                    reports.insert_one(report_doc)
                    st.success("Patient data successfully stored in MongoDB Atlas ✔️")
                except Exception as e:
                    st.error(f"Failed to save to MongoDB: {e}")
                    os.makedirs("reports_backup", exist_ok=True)
                    fname = f"reports_backup/{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    with open(fname, "wb") as f:
                        f.write(pdf_bytes.getvalue())
                    st.warning(f"Report saved locally at {fname}")
            else:
                os.makedirs("reports_backup", exist_ok=True)
                fname = f"reports_backup/{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                with open(fname, "wb") as f:
                    f.write(pdf_bytes.getvalue())
                st.warning("MongoDB offline — report saved locally at " + fname)

            st.success("Report generated successfully!")
            st.download_button(
                "⬇️ Download Dental Report",
                data=pdf_bytes,
                file_name="dental_report.pdf",
                mime="application/pdf",
            )
    else:
        st.info("Run an analysis in the **Analysis Workspace** tab first.")

# ============================================================
#              DASHBOARD TAB (VIEW SAVED RECORDS)
# ============================================================

with tab_dashboard:
    st.subheader("📚 Patient Records Dashboard")

    import pymongo

    MONGO_URI = "mongodb+srv://dental_admin:akash2004@cluster0.nz99qzc.mongodb.net/DentalAI?retryWrites=true&w=majority&tls=true"

    # Try connecting to MongoDB safely
    try:
        client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        client.server_info()
        db = client["DentalAI"]
        reports = db["PatientReports"]
        db_ok = True
    except Exception as e:
        st.error(f"MongoDB Connection Failed ❌\n{e}")
        reports = None
        db_ok = False

    # If DB connection failed → graceful fallback
    if not db_ok or reports is None:
        st.warning("⚠️ MongoDB is offline. Patient records cannot be loaded.")
        st.info("PDF reports can still be generated and downloaded locally.")
    else:
        # Safely fetch records
        try:
            records = list(reports.find().sort("timestamp", -1)) if reports is not None else []
        except Exception as e:
            st.error(f"Error fetching records: {e}")
            records = []

        if len(records) == 0:
            st.info("No patient records found yet.")
        else:
            search = st.text_input("Search by patient name or diagnosis")

            def match(rec):
                if not search:
                    return True
                s = search.lower()
                return (
                    s in rec.get("name", "").lower()
                    or s in rec.get("ai_results", "").lower()
                    or s in rec.get("report_type", "").lower()
                )

            filtered = [r for r in records if match(r)]
            st.write(f"Showing **{len(filtered)}** results")

            for rec in filtered:
                with st.expander(f"{rec.get('name', 'Unknown')} — {rec.get('report_type')}"):
                    st.markdown(f"**Age:** {rec.get('age')}")
                    st.markdown(f"**Symptoms:** {rec.get('symptoms')}")
                    st.markdown(f"**AI Results:** {rec.get('ai_results')}")
                    st.markdown(f"**XAI Summary:** {rec.get('xai_summary')}")
                    st.markdown(f"**Timestamp:** {rec.get('timestamp')}")

                    import base64
                    pdf_bytes = base64.b64decode(rec.get("pdf_base64", ""))

                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=f"{rec.get('name')}_report.pdf",
                        mime="application/pdf",
                        key=f"download_{str(rec['_id'])}"
                    )

# ============================================================
#              FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
  © {year} Dental AI | Built with ❤️ using Streamlit, TensorFlow & YOLO.
</div>
""".format(
        year=datetime.now().year
    ),
    unsafe_allow_html=True,
)
