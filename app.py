import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

# =============== CONFIG =================
st.set_page_config(page_title="Telehealth AI", layout="wide")
BASE_PATH = os.path.join("dataset", "Disease symptom prediction")
HISTORY_FILE = "patient_history.csv"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= STYLE =================
st.markdown("""
<style>
.card {background:white;padding:1.5rem;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,0.08);margin-bottom:1rem;}
.symptom {padding:6px 12px;border-radius:20px;display:inline-block;margin:4px;font-size:14px;}
.red { background:#ffe5e5; color:#b30000; }
.orange { background:#fff2cc; color:#b36b00; }
.green { background:#e5ffe5; color:#006600; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    symptom_df = pd.read_csv(os.path.join(BASE_PATH, "dataset.csv"))
    desc_df = pd.read_csv(os.path.join(BASE_PATH, "symptom_Description.csv"))
    prec_df = pd.read_csv(os.path.join(BASE_PATH, "symptom_precaution.csv"))
    severity_df = pd.read_csv(os.path.join(BASE_PATH, "Symptom-severity.csv"))
    return symptom_df, desc_df, prec_df, severity_df

symptom_df, desc_df, prec_df, severity_df = load_data()
all_diseases = sorted(symptom_df["Disease"].unique())
severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# ================= HELPERS =================
def get_symptoms(disease):
    row = symptom_df[symptom_df["Disease"] == disease]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else []

def get_description(disease):
    row = desc_df[desc_df["Disease"] == disease]
    return row.iloc[0]["Description"] if not row.empty else ""

def get_precautions(disease):
    row = prec_df[prec_df["Disease"] == disease]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else []

def severity_level(symptom):
    s = severity_map.get(symptom, 0)
    if s >= 7: return "red", "Severe"
    if s >= 4: return "orange", "Moderate"
    return "green", "Mild"

def save_history(disease):
    record = {"disease": disease}
    df = pd.DataFrame([record])
    if os.path.exists(HISTORY_FILE):
        df = pd.concat([pd.read_csv(HISTORY_FILE), df])
    df.to_csv(HISTORY_FILE, index=False)

# ================= SYMPTOM NORMALIZATION =================
SYMPTOM_SYNONYMS = {
    "fever": ["high_fever"],
    "fatigue": ["lethargy"],
    "weakness": ["weakness_in_limbs"],
    "pallor": ["pallor"],
    "frequent_urination": ["frequent_urination"],
    "excessive_thirst": ["excessive_thirst"]
}

# ================= OCR DIAGNOSIS =================
def diagnose_from_text(text):
    text = text.lower()
    detected = []

    hb = re.search(r"hemoglobin[:\s]*([\d\.]+)", text)
    if hb and float(hb.group(1)) < 13:
        detected += ["fatigue", "weakness", "pallor"]

    wbc = re.search(r"wbc[:\s]*([\d\.]+)", text)
    if wbc and float(wbc.group(1)) > 11000:
        detected.append("fever")

    sugar = re.search(r"(glucose|blood sugar)[:\s]*([\d\.]+)", text)
    if sugar and float(sugar.group(2)) > 140:
        detected += ["frequent_urination", "excessive_thirst"]

    detected = list(set(detected))

    normalized = []
    for s in detected:
        normalized += SYMPTOM_SYNONYMS.get(s, [s])

    disease_scores = {}
    for d in all_diseases:
        overlap = set(normalized) & set(get_symptoms(d))
        if overlap:
            disease_scores[d] = len(overlap)

    return detected, disease_scores

# ================= HEADER =================
st.title("🩺 Telehealth Assistant")
st.caption("AI-assisted symptom & disease education platform")

tabs = st.tabs(["Search", "Insights", "History", "📄 Scan Report"])

# ================= SEARCH TAB =================
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    query = st.text_input("Search disease")
    matches = get_close_matches(query, all_diseases, n=10, cutoff=0.3) if query else []
    disease = st.selectbox("Select result", matches)

    if disease:
        symptoms = get_symptoms(disease)
        sev_scores = [severity_map.get(s, 0) for s in symptoms]
        avg_sev = sum(sev_scores)/len(sev_scores) if sev_scores else 0

        st.markdown(f"### {disease}")
        for s in symptoms:
            color, label = severity_level(s)
            st.markdown(f"<span class='symptom {color}'>{s} · {label}</span>", unsafe_allow_html=True)

        st.write("#### Description")
        st.write(get_description(disease))

        st.write("#### Precautions")
        for p in get_precautions(disease):
            st.write("•", p)

        if st.button("Save Lookup"):
            save_history(disease)
            st.success("Saved")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= INSIGHTS =================
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        counts = hist["disease"].value_counts()
        fig, ax = plt.subplots()
        ax.barh(counts.index, counts.values)
        st.pyplot(fig)
    else:
        st.info("No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= HISTORY =================
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if os.path.exists(HISTORY_FILE):
        st.dataframe(pd.read_csv(HISTORY_FILE))
    else:
        st.info("No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= REPORT SCAN =================
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Scan Medical Report")
    uploaded = st.file_uploader("Upload report (PNG/JPG)", type=["png","jpg","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=700)

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        if st.button("Extract & Diagnose"):
            text = pytesseract.image_to_string(gray)
            st.text_area("Extracted text", text, height=250)

            detected, diseases = diagnose_from_text(text)

            if detected:
                st.success("Detected symptoms: " + ", ".join(detected))
            else:
                st.info("No symptoms detected.")

            if diseases:
                best = max(diseases, key=diseases.get)
                st.error(f"Possible disease: {best}")
            else:
                st.info("No matching disease found.")

            st.warning("⚠ Educational use only — not a diagnosis.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚠ Educational use only — not a medical diagnosis.")  

