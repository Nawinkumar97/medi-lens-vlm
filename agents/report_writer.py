# agents/report_writer.py

import logging
from typing import Dict
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from utilis import config

logger = logging.getLogger(__name__)

class ReportWriterAgent:
    """
    This agent is responsible for synthesizing all the results from previous agents—
    image analysis, reasoning, critique—into a clean, professional medical report.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Please set it via .env or pass explicitly.")

        self.llm = ChatOpenAI(
            model=config.GPT_MODEL,
            api_key=self.api_key,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

    def compile_report(self, image_findings: Dict, reasoning: str, critique: str) -> Dict:
        """
        Generates a complete diagnostic report using structured findings, reasoning, and critique.

        Args:
            image_findings: Dictionary from the image analyzer
            reasoning: String from the medical reasoner
            critique: String from the risk critic

        Returns:
            Dict with the full synthesized diagnostic report.
        """
        report_prompt = self._build_report_prompt(image_findings, reasoning, critique)

        try:
            message = HumanMessage(content=report_prompt)
            response = self.llm.invoke([message])

            return {
                "status": "success",
                "final_report": response.content.strip(),
                "components": {
                    "image_findings": image_findings,
                    "reasoning": reasoning,
                    "critique": critique
                }
            }
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components": {
                    "image_findings": image_findings,
                    "reasoning": reasoning,
                    "critique": critique
                }
            }

    def _build_report_prompt(self, image_findings: Dict, reasoning: str, critique: str) -> str:
        """
        Combine all diagnostic components into a structured prompt for report generation.
        """
        return f"""
        You are a medical AI assistant tasked with compiling a complete diagnostic report.

        ---
        IMAGE FINDINGS:
        Modality: {image_findings.get('modality')}
        Region: {image_findings.get('anatomical_region')}
        Key Abnormalities: {', '.join(image_findings.get('key_findings', []))}
        Normal Structures: {', '.join(image_findings.get('normal_structures', []))}
        Image Quality: {image_findings.get('image_quality')}
        Confidence Score: {image_findings.get('confidence')}

        ---
        DIAGNOSTIC REASONING:
        {reasoning}

        ---
        CRITICAL REVIEW:
        {critique}

        ---
        Please compile these inputs into a professional, structured medical imaging report suitable
        for clinical documentation. Use medical terminology, and ensure the flow is logical and complete.
        Include sections such as:
        - Imaging Summary
        - Diagnostic Impression
        - Reviewer Comments
        - Confidence Level
        - Recommendations
        """.strip()

# Example usage
if __name__ == "__main__":
    agent = ReportWriterAgent()
    findings = {
        "modality": "X-ray",
        "anatomical_region": "chest",
        "key_findings": ["bilateral infiltrates", "cardiomegaly"],
        "normal_structures": ["diaphragm", "costophrenic angles"],
        "image_quality": "good",
        "confidence": 0.9
    }
    reasoning = "The presence of bilateral infiltrates and enlarged cardiac silhouette is suggestive of pulmonary edema."
    critique = "Consider pneumonia as a differential. Recommend follow-up with BNP levels and clinical correlation."

    print(agent.compile_report(findings, reasoning, critique)['final_report'])
