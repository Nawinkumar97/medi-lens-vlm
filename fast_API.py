# api.py (with optional local report saving)

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from graph.langgraph_orchestration import run_medilens_pipeline
from tools.med_context_loader import MedicalContextLoader
from fpdf import FPDF
import os

app = FastAPI(title="MediLens API")

# Enable CORS for local frontend dev (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/diagnose/")
async def diagnose(
    image: UploadFile,
    query: str = Form(""),
    use_context: bool = Form(False),
    save_report: bool = Form(False),
    report_format: str = Form("md")
):
    """
    Accepts an image file and optional query/context.
    Returns a structured AI-generated diagnostic report and optionally saves it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    context = MedicalContextLoader().load_context() if use_context else ""

    try:
        result = await run_medilens_pipeline(
            image_path=image_path,
            clinical_query=query,
            context=context
        )
        report = result.get("final_report", "")

        # Optional saving
        saved_path = None
        if save_report:
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(image.filename).stem or "diagnosis"
            save_name = f"reports/{base_name}_{timestamp}.{report_format}"

            if report_format == "pdf":
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                for line in report.split("\n"):
                    pdf.multi_cell(0, 10, line)
                pdf.output(save_name)
            else:
                with open(save_name, "w", encoding="utf-8") as f:
                    f.write(report)
            saved_path = save_name

        return JSONResponse(content={
            "status": "success",
            "report": report,
            "saved_path": saved_path
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)