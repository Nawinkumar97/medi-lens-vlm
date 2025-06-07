# graph/langgraph_orchestration.py (Async LangGraph Pipeline with Safe State Access)

import asyncio
from langgraph.graph import StateGraph, END
from agents.image_analyzer import ImageAnalyzerAgent
from agents.medical_reasoner import MedicalReasonerAgent
from agents.risk_critic import RiskCriticAgent
from agents.report_writer import ReportWriterAgent

# Define a basic state class for LangGraph
class MediLensState(dict):
    pass

# Node 1: Analyze the image using GPT-4V
async def image_analysis_node(state: MediLensState) -> MediLensState:
    print("🔍 Running Image Analyzer Agent...")
    analyzer = ImageAnalyzerAgent()
    image_path = state.get("image_path", "")
    query = state.get("clinical_query", "")
    result = analyzer.analyze_image(image_path, query)
    findings = analyzer.get_structured_findings(result)
    print("✅ Image findings extracted.")
    state["image_findings"] = findings
    return state

# Node 2: Perform diagnostic reasoning
async def reasoning_node(state: MediLensState) -> MediLensState:
    print("🧠 Running Medical Reasoner Agent...")
    reasoner = MedicalReasonerAgent()
    result = reasoner.reason_over_findings(state.get("image_findings", {}), state.get("retrieved_context", ""))
    reasoning = result.get("diagnostic_reasoning") if result.get("status") == "success" else result.get("error")
    print("📄 Reasoning:")
    print(reasoning)
    state["reasoning"] = reasoning
    return state

# Node 3: Risk critique of the reasoning
async def critique_node(state: MediLensState) -> MediLensState:
    print("⚠️ Running Risk Critic Agent...")
    critic = RiskCriticAgent()
    result = critic.critique_diagnosis(state.get("reasoning", ""), state.get("retrieved_context", ""))
    critique = result.get("critique") if result.get("status") == "success" else result.get("error")
    print("🔎 Critique:")
    print(critique)
    state["critique"] = critique
    return state

# Node 4: Compile the final medical report
async def report_writer_node(state: MediLensState) -> MediLensState:
    print("📝 Running Report Writer Agent...")
    writer = ReportWriterAgent()
    result = writer.compile_report(state.get("image_findings", {}), state.get("reasoning", ""), state.get("critique", ""))
    report = result.get("final_report") if result.get("status") == "success" else result.get("error")
    print("📑 Final Report Generated.")
    print(report)
    state["final_report"] = report
    return state

# Build the LangGraph flow
builder = StateGraph(MediLensState)

# Register nodes
builder.add_node("ImageAnalysis", image_analysis_node)
builder.add_node("Reasoning", reasoning_node)
builder.add_node("Critique", critique_node)
builder.add_node("Report", report_writer_node)

# Define execution flow
builder.set_entry_point("ImageAnalysis")
builder.add_edge("ImageAnalysis", "Reasoning")
builder.add_edge("Reasoning", "Critique")
builder.add_edge("Critique", "Report")
builder.add_edge("Report", END)

# Compile the graph to an executable
medilens_graph = builder.compile()

# Main pipeline function
async def run_medilens_pipeline(image_path: str, clinical_query: str = "", context: str = ""):
    initial_state = MediLensState({
        "image_path": image_path,
        "clinical_query": clinical_query,
        "retrieved_context": context
    })
    print("🧪 Initial State Keys:", list(initial_state.keys()))
    return await medilens_graph.ainvoke(initial_state)

# Optional test run
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
        print(result.get("final_report", "No report generated."))

    asyncio.run(main())
