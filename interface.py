# demo_interface.py

import streamlit as st
from pathlib import Path
from graph.langgraph_orchestration import run_medilens_pipeline
from tools.med_context_loader import MedicalContextLoader
from fpdf import FPDF
import base64
import asyncio
import tempfile

st.set_page_config(page_title="🩻 MediLens Diagnostic Assistant", layout="centered")
st.title("🩺 MediLens AI Diagnostic Report Generator")

# Upload section
uploaded_file = st.file_uploader("Upload a medical image (JPG or PNG)", type=["jpg", "jpeg", "png"])
clinical_query = st.text_input("Optional: Clinical question or patient context")
use_context = st.checkbox("Use medical context documents", value=True)
output_format = st.radio("Save report as:", ["Markdown (.md)", "PDF (.pdf)"])

# Load context if needed
context = MedicalContextLoader().load_context() if use_context else ""

# Run button
if st.button("Generate Report") and uploaded_file:
    # Save image to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    with st.spinner("Running MediLens diagnostic pipeline..."):
        result = asyncio.run(run_medilens_pipeline(
            image_path=image_path,
            clinical_query=clinical_query,
            context=context
        ))

    report = result.get("final_report", "[Error generating report]")
    st.success("✅ Report generated successfully!")

    st.subheader("📋 Final Report")
    st.text_area("Report Preview", report, height=300)

    # Save report in desired format
    filename = Path(uploaded_file.name).stem + "_report"
    if "Markdown" in output_format:
        report_path = Path(tempfile.gettempdir()) / f"{filename}.md"
        report_path.write_text(report, encoding="utf-8")
        with open(report_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/markdown;base64,{b64}" download="{report_path.name}">📥 Download Markdown Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:  # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in report.split("\n"):
            pdf.multi_cell(0, 10, line)
        report_path = Path(tempfile.gettempdir()) / f"{filename}.pdf"
        pdf.output(str(report_path))
        with open(report_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{report_path.name}">📥 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Please upload an image and click 'Generate Report' to begin.")
