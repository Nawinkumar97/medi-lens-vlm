# graph/langgraph_orchestration.py (Async Version with Streaming and Comments)

import asyncio
from langgraph.graph import StateGraph, END
from agents.image_analyzer import ImageAnalyzerAgent
from agents.medical_reasoner import MedicalReasonerAgent
from agents.risk_critic import RiskCriticAgent
from agents.report_writer import ReportWriterAgent

# Define the shared state class used across all nodes
class MediLensState(dict):
    pass

# Node 1: Run image analysis using GPT-4V
async def image_analysis_node(state: MediLensState) -> MediLensState:
    print("🔍 Running Image Analyzer Agent...")
    analyzer = ImageAnalyzerAgent()
    result = analyzer.analyze_image(state["image_path"], state.get("clinical_query", ""))
    findings = analyzer.get_structured_findings(result)
    print("✅ Image findings extracted.")
    state["image_findings"] = findings
    return state

# Node 2: Use the extracted findings for diagnostic reasoning
async def reasoning_node(state: MediLensState) -> MediLensState:
    print("🧠 Running Medical Reasoner Agent...")
    reasoner = MedicalReasonerAgent()
    result = reasoner.reason_over_findings(state["image_findings"], state.get("retrieved_context", ""))
    reasoning = result["diagnostic_reasoning"] if result["status"] == "success" else result["error"]
    print("📄 Reasoning:")
    print(reasoning)
    state["reasoning"] = reasoning
    return state

# Node 3: Critically evaluate the reasoning for missed risks or differentials
async def critique_node(state: MediLensState) -> MediLensState:
    print("⚠️ Running Risk Critic Agent...")
    critic = RiskCriticAgent()
    result = critic.critique_diagnosis(state["reasoning"], state.get("retrieved_context", ""))
    critique = result["critique"] if result["status"] == "success" else result["error"]
    print("🔎 Critique:")
    print(critique)
    state["critique"] = critique
    return state

# Node 4: Generate a structured medical report
async def report_writer_node(state: MediLensState) -> MediLensState:
    print("📝 Running Report Writer Agent...")
    writer = ReportWriterAgent()
    result = writer.compile_report(state["image_findings"], state["reasoning"], state["critique"])
    report = result["final_report"] if result["status"] == "success" else result["error"]
    print("📑 Final Report Generated.")
    print(report)
    state["final_report"] = report
    return state

# Build LangGraph state machine with nodes and transitions
builder = StateGraph(MediLensState)
builder.add_node("ImageAnalysis", image_analysis_node)
builder.add_node("Reasoning", reasoning_node)
builder.add_node("Critique", critique_node)
builder.add_node("Report", report_writer_node)

# Define flow from one node to the next
builder.set_entry_point("ImageAnalysis")
builder.add_edge("ImageAnalysis", "Reasoning")
builder.add_edge("Reasoning", "Critique")
builder.add_edge("Critique", "Report")
builder.add_edge("Report", END)

# Compile the graph to executable form
medilens_graph = builder.compile()

# Orchestration function to run the pipeline asynchronously
async def run_medilens_pipeline(image_path: str, clinical_query: str = "", context: str = ""):
    initial_state = MediLensState({
        "image_path": image_path,
        "clinical_query": clinical_query,
        "retrieved_context": context
    })
    return await medilens_graph.ainvoke(initial_state)

# Example usage for testing the full pipeline
if __name__ == "__main__":
    async def main():
        print("🚀 Starting MediLens diagnostic pipeline...\n")
        result = await run_medilens_pipeline(
            image_path="data/images/sample_xray.jpg",
            clinical_query="Patient presents with shortness of breath",
            context="Pulmonary edema often appears with cardiomegaly and bilateral infiltrates."
        )
        print("\n✅ Pipeline complete.")
        print("\n🧾 Final Report:\n")
        print(result["final_report"])

    asyncio.run(main())
