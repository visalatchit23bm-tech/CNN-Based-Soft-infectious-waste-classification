from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from datetime import datetime
import os

from predict import predict_image
st.markdown("---")
st.header("📊 Dashboard")

history_file = "history/prediction_history.csv"

if os.path.exists(history_file):

    history = pd.read_csv(history_file)

    if len(history) > 0:

        history["Confidence"] = (
            history["Confidence"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

        total = len(history)

        general = len(
            history[
                history["Prediction"].str.lower() == "general"
            ]
        )

        infectious = len(
            history[
                history["Prediction"].str.lower() == "infectious"
            ]
        )

        avg = history["Confidence"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📂 Total", total)
        c2.metric("🟢 General", general)
        c3.metric("🔴 Infectious", infectious)
        c4.metric("🎯 Avg Confidence", f"{avg:.2f}%")

    else:
        st.info("No predictions available.")
# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Biomedical Waste Classification",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#0E1117;
    color:#FFFFFF;
}

/* Main Content */
.main .block-container{
    background-color:#0E1117;
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#161B22;
}

/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color:white;
}

/* Titles */
h1,h2,h3,h4,h5,h6{
    color:#00E5FF;
}

/* Normal Text */
p,label,div,span{
    color:white;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:#1C2430;
    border-radius:15px;
    padding:20px;
    border:1px solid #2F3A4D;
    box-shadow:0px 0px 12px rgba(0,229,255,0.15);
}

/* Metric Labels */
div[data-testid="metric-container"] label{
    color:#BBBBBB;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:48px;
    border-radius:12px;
    background:#00BFFF;
    color:white;
    border:none;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0099CC;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    background:#1C2430;
    border-radius:12px;
    padding:15px;
}

/* Success Box */
.stSuccess{
    background:#123524;
}

/* Info Box */
.stInfo{
    background:#132238;
}

/* Warning */
.stWarning{
    background:#3A2A00;
}

/* Error */
.stError{
    background:#3A1414;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    background:#161B22;
}

/* Horizontal line */
hr{
    border:1px solid #2B3545;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.image("images/logo.png", width=220)

st.sidebar.markdown("# Biomedical Waste")

st.sidebar.markdown("---")

st.sidebar.success("👩‍💻 Developed by")

st.sidebar.markdown("## Visalatchi T")

st.sidebar.markdown("Biomedical Engineering Student")

st.sidebar.markdown("---")

st.sidebar.info("🤖 MobileNetV2 AI Model")

st.sidebar.markdown("---")

st.sidebar.subheader("Waste Categories")

st.sidebar.write("""
🟢 General Waste

🔴 Infectious Waste
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Technologies")

st.sidebar.write("""
✔ Python

✔ TensorFlow

✔ Streamlit

✔ MobileNetV2
""")

# ----------------------------------------
# Language Selection
# ----------------------------------------

language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "தமிழ்"]
)

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

if language == "English":
    st.title("🧪 CNN-Based Soft Infectious Biomedical Waste Classification System")
else:
    st.title("🧪 தொற்று உயிரியல் மருத்துவக் கழிவு வகைப்படுத்தும் அமைப்பு")

st.write("### Developed by Visalatchi T")

st.markdown("---")

if language == "English":
    st.write("""
This application classifies biomedical waste into:

🟢 General Waste

🔴 Infectious Waste

using MobileNetV2.
""")
else:
    st.write("""
இந்த செயலி உயிரியல் மருத்துவக் கழிவுகளை வகைப்படுத்துகிறது.

🟢 பொதுக் கழிவு

🔴 தொற்று கழிவு

MobileNetV2 மாதிரியை பயன்படுத்துகிறது.
""")

# -------------------------------------------------
# IMAGE INPUT
# -------------------------------------------------

st.header("📷 Select Image Source")

input_method = st.radio(
    "Choose Input Method",
    ["📁 Upload Image", "📷 Capture from Webcam"],
    horizontal=True
)

image = None
image_name = ""

if input_method == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Biomedical Waste Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_name = uploaded_file.name

else:

    camera_photo = st.camera_input("Take a Picture")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        image_name = "Captured_Image.jpg"

# -------------------------------------------------
# IMAGE PREVIEW
# -------------------------------------------------

if image is not None:

    st.markdown("---")
    st.subheader("🖼 Uploaded Image")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(
            image,
            caption=image_name,
            use_container_width=True
        )

    with col2:
        st.success("✅ Image Loaded Successfully")
        st.info("""
Ready for prediction.

Click the **Predict** button below.
""")

    # -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------

predict_btn = st.button(
    "🔍 Predict",
    use_container_width=True
)

if predict_btn:

    if image is None:

        st.warning("⚠ Please upload or capture an image first.")

    else:

        with st.spinner("🔄 Predicting..."):

            # Get prediction from model
            predicted_class, confidence, probabilities = predict_image(image)
            # ---------------------------------------
# Save Prediction History
# ---------------------------------------

history_file = "history/prediction_history.csv"

new_data = pd.DataFrame({
    "Date": [datetime.now().strftime("%d-%m-%Y")],
    "Time": [datetime.now().strftime("%H:%M:%S")],
    "Image": [image_name],
    "Prediction": [predicted_class],
    "Confidence": [f"{confidence:.2f}"]
})

if os.path.exists(history_file):
    history = pd.read_csv(history_file)
else:
    history = pd.DataFrame(
        columns=["Date", "Time", "Image", "Prediction", "Confidence"]
    )

history = pd.concat([history, new_data], ignore_index=True)

history.to_csv(history_file, index=False)
 # ---------------------------------
        # Translate prediction (Tamil)
        # ---------------------------------

display_prediction = predicted_class

if language == "தமிழ்":

            if predicted_class.lower() == "general":
                display_prediction = "பொதுக் கழிவு"

            elif predicted_class.lower() == "infectious":
                display_prediction = "தொற்று கழிவு"

        
# Display Prediction
#  # ---------------------------------
st.success(f"### Prediction : {display_prediction}")

st.info(f"Confidence : {confidence:.2f}%")

st.progress(confidence / 100)

        # ---------------------------------
        # Confidence Chart
        # ---------------------------------

st.markdown("---")
st.subheader("📊 Prediction Confidence")
classes = ["General", "Infectious"]

scores = probabilities * 100

fig, ax = plt.subplots(figsize=(5,5))

fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")

colors = ["#00E676", "#FF1744"]

ax.pie(
    [general, infectious],
    labels=["General","Infectious"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    textprops={"color":"white"}
)

ax.set_title(
    "Prediction Distribution",
    color="white",
    fontsize=16
)

st.pyplot(fig)
st.markdown("---")
st.subheader("🥧 Prediction Distribution")

if os.path.exists(history_file):

    history = pd.read_csv(history_file)

    if len(history) > 0:

        general = len(
            history[
                history["Prediction"].str.lower() == "general"
            ]
        )

        infectious = len(
            history[
                history["Prediction"].str.lower() == "infectious"
            ]
        )

        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            [general, infectious],
            labels=["General", "Infectious"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Prediction Distribution")

        st.pyplot(fig)
        st.markdown("---")

st.subheader("📥 Download Prediction History")

if os.path.exists(history_file):

    with open(history_file, "rb") as file:

        st.download_button(
            label="⬇ Download CSV",
            data=file,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
st.markdown("---")
st.subheader("📄 Download Prediction Report")

if st.button("Generate PDF Report"):

    pdf_path = "Biomedical_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>Biomedical Waste Prediction Report</b>", styles["Title"]))

    elements.append(
        Paragraph(
            f"Prediction : {predicted_class}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Confidence : {confidence:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Date : {datetime.now()}",
            styles["BodyText"]
        )
    )

    doc.build(elements)

    with open(pdf_path, "rb") as pdf:

        st.download_button(
            "⬇ Download PDF",
            pdf,
            file_name="Biomedical_Report.pdf",
            mime="application/pdf"
        )
    # -------------------------------------------------------
    # Waste Description
    # -------------------------------------------------------

st.markdown("---")
st.subheader("📝 Waste Description")

if predicted_class.lower() == "infectious":
        st.error("""
### 🔴 Infectious Biomedical Waste

Infectious biomedical waste contains materials contaminated
with blood, body fluids, microorganisms, or infectious agents.

Examples:

✔ Gloves

✔ Masks

✔ Cotton

✔ Gauze

✔ Bandages

These wastes should be handled carefully to prevent infection.
""")
else:
        st.success("""
### 🟢 General Waste

General biomedical waste is non-infectious and
does not contain harmful microorganisms.

Examples:

✔ Plastic

✔ Paper

✔ Glass

✔ Metal

✔ Food Waste

These materials can be disposed of through normal waste
management procedures.
""")

    # -------------------------------------------------------
    # Disposal Instructions
    # -------------------------------------------------------

st.markdown("---")
st.subheader("🗑 Disposal Instructions")

if predicted_class.lower() == "infectious":
        st.warning("""
🟡 Dispose in the **Yellow Biomedical Waste Bin**

✔ Wear gloves

✔ Seal the waste bag properly

✔ Do not mix with general waste

✔ Follow Biomedical Waste Management Rules
""")
else:
        st.info("""
🟢 Dispose in the **General Waste Bin**

✔ Segregate recyclable materials

✔ Dispose according to municipal waste rules

✔ Keep infectious waste separate
""")

    # -------------------------------------------------------
    # Biomedical Safety Tips
    # -------------------------------------------------------

st.markdown("---")
st.subheader("🩺 Biomedical Safety Tips")

st.success("""
✔ Always wear gloves.

✔ Wash hands after handling biomedical waste.

✔ Do not mix infectious and general waste.

✔ Follow color-coded waste segregation.

✔ Dispose waste immediately after use.

✔ Use PPE whenever necessary.
""")

