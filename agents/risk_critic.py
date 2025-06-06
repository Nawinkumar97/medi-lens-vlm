# agents/risk_critic.py

import logging
from typing import Dict
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from utilis import config

logger = logging.getLogger(__name__)

class RiskCriticAgent:
    """
    This agent performs critical review of a proposed medical reasoning output.
    It evaluates risks, edge cases, alternative diagnoses, and uncertainties.
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

    def critique_diagnosis(self, diagnostic_report: str, context: str = "") -> Dict:
        """
        Evaluate the diagnostic reasoning critically.

        Args:
            diagnostic_report: The output from the medical reasoner (string)
            context: Additional background if available

        Returns:
            Dict containing risks, alternative explanations, and review notes.
        """
        critique_prompt = self._build_critique_prompt(diagnostic_report, context)

        try:
            message = HumanMessage(content=critique_prompt)
            response = self.llm.invoke([message])

            return {
                "status": "success",
                "critique": response.content.strip(),
                "original_report": diagnostic_report,
                "context_used": context
            }
        except Exception as e:
            logger.error(f"Error during risk critique: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_report": diagnostic_report
            }

    def _build_critique_prompt(self, reasoning_text: str, context: str = "") -> str:
        """
        Construct a prompt to critically review a medical diagnosis.
        """
        prompt = f"""
        You are a senior clinical reviewer. Given the diagnostic reasoning below, identify possible errors,
        risks, and overlooked differentials. Suggest improvements or clarifications.

        ---
        DIAGNOSTIC REPORT:
        {reasoning_text}
        ---
        CONTEXT:
        {context or 'No additional context provided.'}
        ---

        Please structure your output as:
        1. Potential Risks or Missed Diagnoses
        2. Alternative Interpretations
        3. Suggestions for Improvement or Clarification
        4. Clinical Safety Notes (if any)
        """
        return prompt.strip()

# Example test
if __name__ == "__main__":
    agent = RiskCriticAgent()
    diagnosis = "Primary Diagnosis: Pulmonary edema. Differentials: Pneumonia, ARDS. Justification: bilateral infiltrates and cardiomegaly."
    result = agent.critique_diagnosis(diagnosis)
    print(result['critique'])
