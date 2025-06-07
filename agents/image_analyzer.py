# agents/image_analyzer_agent.py

import base64
import os
import re
from typing import Dict, List
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ImageAnalyzerAgent:
    """
    Agent responsible for analyzing medical images using GPT-4 Vision.
    Performs image validation, encoding, prompt creation, API interaction,
    and parsing of structured diagnostic information.
    """

    def __init__(self, api_key: str = None):
        from utilis import config
        self.api_key = api_key or config.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass as parameter.")

        # Initialize GPT-4 Vision LLM from OpenAI
        self.llm = ChatOpenAI(
            model=config.GPT_MODEL,
            api_key=self.api_key,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 encoding for API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    def validate_medical_image(self, image_path: str) -> bool:
        """Check image dimensions and format validity."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = (img.format or "").upper()
                if width < 100 or height < 100 or width > 4096 or height > 4096:
                    return False
                if img_format not in ["JPEG", "PNG"]:
                    return False
                return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def analyze_image(self, image_path: str, clinical_query: str = "") -> Dict:
        """Full analysis pipeline for a medical image."""
        if not self.validate_medical_image(image_path):
            raise ValueError("Invalid medical image format or size")

        base64_image = self.encode_image(image_path)
        analysis_prompt = self._build_analysis_prompt(clinical_query)

        try:
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.bmp': 'image/bmp',
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff'
            }.get(ext, 'image/jpeg')

            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }}
                ]
            )
            response = self.llm.invoke([message])
            analysis_result = self._parse_analysis_response(response.content)

            return {
                "status": "success",
                "analysis": analysis_result,
                "confidence_score": self._extract_confidence_score(response.content),
                "image_path": image_path,
                "clinical_query": clinical_query
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "image_path": image_path
            }

    def _build_analysis_prompt(self, clinical_query: str) -> str:
        """Construct a detailed system prompt for GPT-4 Vision."""
        base_prompt = """
        As an expert medical imaging specialist, analyze this medical image systematically.
        Provide detailed analysis including:
        - VISUAL ANALYSIS: Type, structures, normal/abnormal findings, technical issues
        - MORPHOLOGICAL ASSESSMENT: Size, density, symmetry, borders
        - DIFFERENTIAL DIAGNOSIS: Primary and alternatives
        - CONFIDENCE SCORE: 1 to 10 with reasoning
        """
        if clinical_query:
            base_prompt += f"\nCLINICAL CONTEXT: {clinical_query}\n"
        return base_prompt.strip()

    def _parse_analysis_response(self, response_content: str) -> Dict:
        """Parse GPT-4V output into structured findings."""
        sections = {
            "visual_findings": "",
            "anatomical_assessment": "",
            "abnormal_findings": "",
            "differential_diagnosis": "",
            "technical_quality": "",
            "recommendations": ""
        }
        current_section = "visual_findings"
        for line in response_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            if "VISUAL ANALYSIS" in upper:
                current_section = "visual_findings"
            elif "MORPHOLOGICAL" in upper or "ASSESSMENT" in upper:
                current_section = "anatomical_assessment"
            elif "DIFFERENTIAL" in upper:
                current_section = "differential_diagnosis"
            elif "TECHNICAL" in upper:
                current_section = "technical_quality"
            elif "RECOMMEND" in upper:
                current_section = "recommendations"
            else:
                sections[current_section] += line + "\n"
        return {k: v.strip() for k, v in sections.items()}

    def _extract_confidence_score(self, response_content: str) -> float:
        """Extract numeric confidence score (1–10) and normalize to 0–1."""
        patterns = [
            r'confidence[:\s]+(\d+)',
            r'certainty[:\s]+(\d+)',
            r'confidence.*?(\d+).*?(?:out of|/)\s*10'
        ]
        for pattern in patterns:
            match = re.search(pattern, response_content.lower())
            if match:
                score = int(match.group(1))
                return min(score / 10.0, 1.0)
        return 0.7  # default if not found

    def get_structured_findings(self, analysis_result: Dict) -> Dict:
        """Convert raw analysis dict to downstream-friendly structured format."""
        if analysis_result.get("status") != "success":
            return {"error": "Analysis failed"}

        analysis = analysis_result.get("analysis", {})
        return {
            "modality": self._extract_modality(analysis.get("visual_findings", "")),
            "anatomical_region": self._extract_anatomical_region(analysis.get("visual_findings", "")),
            "key_findings": self._extract_key_findings(analysis.get("abnormal_findings", "")),
            "normal_structures": self._extract_normal_findings(analysis.get("visual_findings", "")),
            "image_quality": self._assess_image_quality(analysis.get("technical_quality", "")),
            "confidence": analysis_result.get("confidence_score", 0.7),
            "differential_diagnoses": self._extract_differentials(analysis.get("differential_diagnosis", ""))
        }

    def _extract_modality(self, visual_findings: str) -> str:
        modalities = ["x-ray", "ct", "mri", "ultrasound", "mammography", "pet", "nuclear"]
        text = visual_findings.lower()
        for m in modalities:
            if m in text:
                return m
        return "unknown"

    def _extract_anatomical_region(self, visual_findings: str) -> str:
        regions = ["chest", "abdomen", "pelvis", "head", "neck", "spine", "extremity", "heart", "lung"]
        text = visual_findings.lower()
        for r in regions:
            if r in text:
                return r
        return "unspecified"

    def _extract_key_findings(self, text: str) -> List[str]:
        return self._extract_bullets(text, max_items=5)

    def _extract_differentials(self, text: str) -> List[str]:
        return self._extract_bullets(text, max_items=3)

    def _extract_normal_findings(self, text: str) -> List[str]:
        keywords = ["normal", "unremarkable", "within normal limits", "no evidence"]
        return [line.strip() for line in text.split('\n') if any(k in line.lower() for k in keywords)]

    def _assess_image_quality(self, text: str) -> str:
        quality_map = {
            "excellent": ["excellent", "optimal", "high quality"],
            "good": ["good", "adequate", "satisfactory"],
            "fair": ["fair", "moderate", "acceptable"],
            "poor": ["poor", "suboptimal", "limited", "degraded"]
        }
        lower = text.lower()
        for q, keywords in quality_map.items():
            if any(k in lower for k in keywords):
                return q
        return "not assessed"

    def _extract_bullets(self, text: str, max_items: int = 5) -> List[str]:
        lines = text.split('\n')
        items = []
        for line in lines:
            clean = re.sub(r'^[\d\-\*\s\.•]+', '', line).strip()
            if clean:
                items.append(clean)
        return items[:max_items]
