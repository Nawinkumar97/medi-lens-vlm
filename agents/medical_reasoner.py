# agents/medical_reasoner.py

import logging
from typing import Dict
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from utilis import config

logger = logging.getLogger(__name__)

class MedicalReasonerAgent:
    """
    This agent synthesizes structured image findings and contextual medical knowledge
    to propose a clinical interpretation, suggest a diagnosis, and raise differentials.
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

    def reason_over_findings(self, findings: Dict, retrieved_context: str = "") -> Dict:
        """
        Generate a medical reasoning report given image findings and optional medical text context.

        Args:
            findings: A dictionary of structured image findings (e.g., modality, region, key findings)
            retrieved_context: Optional string of domain-specific knowledge or guidelines

        Returns:
            Dictionary with reasoning, suggested diagnosis, differentials, and explanation.
        """
        reasoning_prompt = self._build_reasoning_prompt(findings, retrieved_context)

        try:
            message = HumanMessage(content=reasoning_prompt)
            response = self.llm.invoke([message])

            return {
                "status": "success",
                "diagnostic_reasoning": response.content.strip(),
                "input_findings": findings,
                "used_context": retrieved_context
            }
        except Exception as e:
            logger.error(f"Error during diagnostic reasoning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "input_findings": findings
            }

    def _build_reasoning_prompt(self, findings: Dict, context: str) -> str:
        """Construct a diagnostic reasoning prompt from findings and context."""
        prompt = f"""
        You are a clinical decision support assistant. Given the following imaging findings and optional medical knowledge,
        reason through what diagnosis is most likely, what differentials should be considered, and explain your reasoning.

        ---
        FINDINGS:
        Modality: {findings.get('modality', 'N/A')}
        Region: {findings.get('anatomical_region', 'N/A')}
        Key Findings: {', '.join(findings.get('key_findings', []))}
        Normal Structures: {', '.join(findings.get('normal_structures', []))}
        Image Quality: {findings.get('image_quality', 'N/A')}
        Confidence Score: {findings.get('confidence', 'N/A')}
        ---
        CONTEXT:
        {context or 'No additional context provided.'}
        ---

        Please structure your response into:
        1. Primary Diagnostic Impression
        2. Differential Diagnoses (at least 2)
        3. Justification / Reasoning
        4. Any additional recommended tests or steps
        """
        return prompt.strip()

# Example test
if __name__ == "__main__":
    agent = MedicalReasonerAgent()
    dummy_findings = {
        "modality": "X-ray",
        "anatomical_region": "chest",
        "key_findings": ["bilateral infiltrates", "cardiomegaly"],
        "normal_structures": ["diaphragm", "costophrenic angles"],
        "image_quality": "good",
        "confidence": 0.85
    }
    print(agent.reason_over_findings(dummy_findings, retrieved_context="Pulmonary edema can present with bilateral infiltrates and enlarged heart.")['diagnostic_reasoning'])
