# agents/image_analyzer_agent.py

import base64
import os
from typing import Dict, List, Optional
from langchain.agents import Agent
from langchain.schema import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class ImageAnalyzerAgent:
    """
    Agent responsible for analyzing medical images using GPT-4 Vision.
    Extracts visual features, anatomical structures, and potential abnormalities.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",  # GPT-4 with vision capabilities
            api_key=self.api_key,
            temperature=0.1,  # Low temperature for medical analysis
            max_tokens=1000
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
        """Basic validation to ensure image is suitable for medical analysis."""
        try:
            with Image.open(image_path) as img:
                # Check if image is too small or too large
                width, height = img.size
                if width < 100 or height < 100:
                    return False
                if width > 4096 or height > 4096:
                    return False
                return True
        except Exception:
            return False
    
    def analyze_image(self, image_path: str, clinical_query: str = "") -> Dict:
        """
        Perform comprehensive medical image analysis.
        
        Args:
            image_path: Path to the medical image
            clinical_query: Optional clinical context or specific question
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.validate_medical_image(image_path):
            raise ValueError("Invalid medical image format or size")
        
        base64_image = self.encode_image(image_path)
        
        # Construct the analysis prompt
        analysis_prompt = self._build_analysis_prompt(clinical_query)
        
        try:
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            
            # Parse and structure the response
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
        """Build comprehensive analysis prompt for medical image analysis."""
        base_prompt = """
        As an expert medical imaging specialist, analyze this medical image systematically. Provide a detailed analysis including:

        **VISUAL ANALYSIS:**
        1. Image Type & Quality: Identify modality (X-ray, CT, MRI, ultrasound, etc.) and assess image quality
        2. Anatomical Structures: Describe visible anatomical structures and their appearance
        3. Normal Findings: List structures that appear normal and within expected parameters
        4. Abnormal Findings: Identify any abnormalities, lesions, or concerning features
        5. Technical Factors: Comment on positioning, contrast, artifacts, or technical limitations

        **MORPHOLOGICAL ASSESSMENT:**
        1. Size & Shape: Describe dimensions and morphology of key structures
        2. Density/Signal: Comment on tissue density, signal intensity, or echogenicity
        3. Borders & Margins: Describe edge characteristics of any abnormalities
        4. Symmetry: Note any asymmetries between bilateral structures
        5. Associated Findings: Identify secondary signs or related abnormalities

        **DIFFERENTIAL CONSIDERATIONS:**
        1. Primary Impressions: List most likely diagnostic possibilities
        2. Alternative Diagnoses: Consider other potential explanations
        3. Uncertainty Areas: Identify regions requiring further evaluation
        4. Recommended Follow-up: Suggest additional imaging or views if needed

        **CONFIDENCE ASSESSMENT:**
        Rate your confidence in the analysis on a scale of 1-10 and explain any limitations.
        """
        
        if clinical_query:
            base_prompt += f"\n\n**CLINICAL CONTEXT:**\n{clinical_query}\n\nPlease address this specific clinical question in your analysis."
        
        base_prompt += "\n\nProvide your analysis in a structured, professional medical imaging report format."
        
        return base_prompt
    
    def _parse_analysis_response(self, response_content: str) -> Dict:
        """Parse the GPT-4V response into structured components."""
        # Simple parsing - in production, you might want more sophisticated NLP parsing
        sections = {
            "visual_findings": "",
            "anatomical_assessment": "",
            "abnormal_findings": "",
            "differential_diagnosis": "",
            "technical_quality": "",
            "recommendations": ""
        }
        
        # Basic section extraction (you can enhance this with regex or NLP)
        lines = response_content.split('\n')
        current_section = "visual_findings"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify section headers
            if "VISUAL ANALYSIS" in line.upper() or "FINDINGS" in line.upper():
                current_section = "visual_findings"
            elif "MORPHOLOGICAL" in line.upper() or "ASSESSMENT" in line.upper():
                current_section = "anatomical_assessment"
            elif "DIFFERENTIAL" in line.upper() or "DIAGNOSIS" in line.upper():
                current_section = "differential_diagnosis"
            elif "TECHNICAL" in line.upper() or "QUALITY" in line.upper():
                current_section = "technical_quality"
            elif "RECOMMEND" in line.upper() or "FOLLOW" in line.upper():
                current_section = "recommendations"
            else:
                sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def _extract_confidence_score(self, response_content: str) -> float:
        """Extract confidence score from the response."""
        # Look for confidence indicators in the text
        import re
        
        # Search for patterns like "confidence: 8/10" or "confidence level: 7"
        confidence_patterns = [
            r'confidence[:\s]+(\d+)[/\s]*(?:10)?',
            r'certainty[:\s]+(\d+)[/\s]*(?:10)?',
            r'confidence.*?(\d+).*?(?:out of|/)\s*10'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_content.lower())
            if match:
                score = int(match.group(1))
                return min(score / 10.0, 1.0)  # Normalize to 0-1 scale
        
        # Default confidence if no explicit score found
        return 0.7
    
    def get_structured_findings(self, analysis_result: Dict) -> Dict:
        """Extract key findings in a structured format for other agents."""
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
        """Extract imaging modality from findings."""
        modalities = ["x-ray", "ct", "mri", "ultrasound", "mammography", "pet", "nuclear"]
        text_lower = visual_findings.lower()
        
        for modality in modalities:
            if modality in text_lower:
                return modality
        return "unknown"
    
    def _extract_anatomical_region(self, visual_findings: str) -> str:
        """Extract primary anatomical region."""
        regions = ["chest", "abdomen", "pelvis", "head", "neck", "spine", "extremity", "heart", "lung"]
        text_lower = visual_findings.lower()
        
        for region in regions:
            if region in text_lower:
                return region
        return "unspecified"
    
    def _extract_key_findings(self, abnormal_findings: str) -> List[str]:
        """Extract key abnormal findings as a list."""
        if not abnormal_findings:
            return []
        
        # Simple extraction - split by common delimiters
        findings = []
        for line in abnormal_findings.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove bullet points and numbering
                clean_line = re.sub(r'^[\d\.\-\*\•\s]+', '', line)
                if clean_line:
                    findings.append(clean_line)
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_normal_findings(self, visual_findings: str) -> List[str]:
        """Extract normal findings."""
        normal_keywords = ["normal", "unremarkable", "within normal limits", "no evidence"]
        findings = []
        
        for line in visual_findings.split('\n'):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in normal_keywords):
                findings.append(line.strip())
        
        return findings
    
    def _assess_image_quality(self, technical_quality: str) -> str:
        """Assess overall image quality."""
        quality_indicators = {
            "excellent": ["excellent", "optimal", "high quality"],
            "good": ["good", "adequate", "satisfactory"],
            "fair": ["fair", "moderate", "acceptable"],
            "poor": ["poor", "suboptimal", "limited", "degraded"]
        }
        
        text_lower = technical_quality.lower()
        
        for quality, keywords in quality_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                return quality
        
        return "not assessed"
    
    def _extract_differentials(self, differential_text: str) -> List[str]:
        """Extract differential diagnoses."""
        if not differential_text:
            return []
        
        differentials = []
        for line in differential_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove bullet points and numbering
                clean_line = re.sub(r'^[\d\.\-\*\•\s]+', '', line)
                if clean_line and len(clean_line) > 10:  # Filter out very short items
                    differentials.append(clean_line)
        
        return differentials[:3]  # Limit to top 3 differentials

# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    analyzer = ImageAnalyzerAgent()
    
    # Example analysis (you would use actual image path)
    # result = analyzer.analyze_image("path/to/medical_image.jpg", "Patient presents with chest pain")
    # print(result)