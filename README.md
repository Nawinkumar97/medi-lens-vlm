# 🩺 MediLens: Multi-Agent VLM-Powered Medical Image Diagnostic Assistant

MediLens is a multi-agent AI system built with **LangGraph**, **OpenAI's GPT-4 Vision**, and **LangChain** to perform intelligent medical image analysis. It simulates the diagnostic reasoning process of a medical team using **Vision-Language Models (VLMs)**, context retrieval, reasoning, critique, and report generation.

---

## 🎯 Objective

Given a **medical image** (e.g., X-ray, MRI, pathology scan) and a **clinical query**, MediLens:
1. Extracts visual insights using GPT-4 Vision
2. Retrieves relevant medical context from structured or simulated sources
3. Applies clinical reasoning to formulate a diagnostic hypothesis
4. Critically evaluates potential risks and differentials
5. Produces a polished diagnostic report

---

## 🧠 Multi-Agent Architecture

| Agent              | Role Description |
|--------------------|------------------|
| 🧑‍⚕️ `ImageAnalyzerAgent` | Performs visual analysis using GPT-4V |
| 📚 `RetrieverAgent`       | Fetches relevant clinical knowledge |
| 🧠 `MedicalReasonerAgent` | Combines image + text for diagnostic logic |
| ⚖️ `RiskCriticAgent`      | Evaluates risks, edge cases, and uncertainties |
| 📝 `ReportWriterAgent`    | Compiles a final structured clinical report |
| 🔄 `CoordinatorAgent`     | Orchestrates flow via LangGraph |

---

## 🧰 Tech Stack

- 🔁 **LangGraph** — State machine-style multi-agent orchestration  
- 🤖 **OpenAI GPT-4V** — Vision-language model for multimodal reasoning  
- 🧱 **LangChain** — Tools, memory, agent interfaces  
- 🔎 **FAISS / Chroma** — Optional vector DB for retrieval  
- 🌐 **Gradio** — (Optional) Web UI for image upload and query  
- 🐍 **Python 3.10+**  

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/medi-lens-vlm.git
cd medi-lens-vlm
