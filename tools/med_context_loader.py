# tools/med_context_loader.py

import os
from pathlib import Path
from utilis import config

class MedicalContextLoader:
    """
    Loads and aggregates medical reference documents from the configured knowledge base directory.
    Useful for supplying background context to the medical reasoning or critique agents.
    """

    def __init__(self, kb_path: str = None):
        self.kb_path = Path(kb_path or config.KNOWLEDGE_BASE_PATH)
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Medical knowledge directory not found: {self.kb_path}")

    def load_context(self) -> str:
        """
        Reads all .txt files in the knowledge base directory and concatenates their contents.

        Returns:
            A single string containing the concatenated context text.
        """
        content = []
        for file in self.kb_path.glob("*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content.append(f"\n---\n# {file.name}\n" + f.read().strip())
            except Exception as e:
                print(f"⚠️ Error reading {file.name}: {e}")
        return "\n".join(content).strip()

# Example usage
if __name__ == "__main__":
    loader = MedicalContextLoader()
    context_text = loader.load_context()
    print("📚 Loaded Context:\n")
    print(context_text[:1000])  # Preview first 1000 characters
