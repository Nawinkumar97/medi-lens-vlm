# main.py (with validation logging and corrected image path handling)

import os
import asyncio
import argparse
from pathlib import Path
from graph.langgraph_orchestration import run_medilens_pipeline
from tools.med_context_loader import MedicalContextLoader
from fpdf import FPDF  # Optional: install with `pip install fpdf`
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_context_if_needed(use_context: bool) -> str:
    """Load context documents if requested."""
    if use_context:
        logger.info("📚 Loading medical context documents...")
        return MedicalContextLoader().load_context()
    return ""

def save_report(output_dir: str, filename: str, report_text: str, as_pdf: bool = False):
    """Save the report as .md or .pdf to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / filename

    if as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in report_text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(str(path.with_suffix(".pdf")))
    else:
        with open(path.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write(report_text)

    print(f"💾 Report saved to: {path.with_suffix('.pdf' if as_pdf else '.md')}")

def parse_arguments():
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the MediLens AI diagnostic pipeline")
    parser.add_argument("--image", help="Path to a medical image OR folder of images")
    parser.add_argument("--query", default="", help="Clinical question or context")
    parser.add_argument("--use-context", action="store_true", help="Include knowledge base context")
    parser.add_argument("--output", default="reports", help="Output folder for saving reports")
    parser.add_argument("--as-pdf", action="store_true", help="Save reports as PDF instead of Markdown")
    return parser.parse_args()

async def run_pipeline_for_image(image_path: str, query: str, context: str, output: str, as_pdf: bool):
    print(f"\n🚀 Running pipeline for image: {image_path}")
    logger.info(f"🖼️ Validating image path: {image_path}")

    if not os.path.isfile(image_path):
        logger.error(f"❌ Image path does not exist or is not a file: {image_path}")
        return

    result = await run_medilens_pipeline(image_path=image_path, clinical_query=query, context=context)
    report_text = result["final_report"]
    print("\n🧾 Report Preview:\n")
    print(report_text[:500] + "...\n")

    base_name = Path(image_path).stem + "_report"
    save_report(output, base_name, report_text, as_pdf=as_pdf)

async def main():
    args = parse_arguments()
    context = load_context_if_needed(args.use_context)

    if not args.image:
        print("❌ Please provide --image as a file or folder path.")
        return

    image_path = Path(args.image)
    if image_path.is_file():
        await run_pipeline_for_image(str(image_path), args.query, context, args.output, args.as_pdf)
    elif image_path.is_dir():
        images = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        for img in images:
            await run_pipeline_for_image(str(img), args.query, context, args.output, args.as_pdf)
    else:
        print("❌ Provided image path is invalid.")

if __name__ == "__main__":
    asyncio.run(main())
