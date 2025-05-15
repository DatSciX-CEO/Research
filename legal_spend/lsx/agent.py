# File: C:\DatSciX\Research\legal_spend\lsx\agent.py

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ADK Components
# Remove SessionMetadata from this import
from google.adk.sessions import InMemorySessionService, Session
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig

# --- Define Data Models (Pydantic Schemas) ---
class InvoiceData(BaseModel):
    """Represents the data extracted from an invoice."""
    invoice_id: str
    vendor_name: str
    total_amount: float
    line_items: List[Dict[str, Any]]
    raw_text: Optional[str] = None

class AssessmentResult(BaseModel):
    """Represents the outcome of assessing an invoice against guidelines."""
    invoice_id: str
    is_compliant: bool
    potential_issues: List[str]
    categorized_spend: Dict[str, float]

class AnalysisReport(BaseModel):
    """Represents the output of a spend analysis."""
    report_id: str
    summary: str
    trends: List[str]
    cost_saving_opportunities: List[str]

# --- Define Custom Tools (Simulated Examples) ---
def fetch_invoice_data_from_source(source_type: str, source_identifier: str) -> InvoiceData:
    print(f"TOOL EXECUTING: fetch_invoice_data_from_source(source_type='{source_type}', source_identifier='{source_identifier}')")
    if source_type == "db" and source_identifier == "inv-001":
        return InvoiceData(
            invoice_id="inv-001",
            vendor_name="Alpha Legal Services",
            total_amount=15000.75,
            line_items=[
                {"task_code": "L110", "description": "Initial Case Review", "hours": 10, "rate": 300, "total": 3000},
                {"task_code": "L120", "description": "Document Drafting - Pleadings", "hours": 25, "rate": 350, "total": 8750},
                {"task_code": "A101", "description": "Client Consultation", "hours": 5, "rate": 400, "total": 2000},
                {"task_code": "E101", "description": "Court Filing Fees", "total": 1250.75}
            ]
        )
    elif source_type == "pdf" and source_identifier == "invoice_xyz.pdf":
        return InvoiceData(
            invoice_id="inv-pdf-002",
            vendor_name="Beta Consulting Group",
            total_amount=5000.00,
            line_items=[{"description": "Strategic Legal Advice Session", "total": 5000.00}],
            raw_text="Simulated PDF content for invoice_xyz.pdf. Includes vendor Beta Consulting, total $5000..."
        )
    print(f"TOOL ERROR: Invalid data source specified: {source_type}, {source_identifier}")
    raise ValueError("Invalid data source specified for tool.")

fetch_invoice_tool = FunctionTool(func=fetch_invoice_data_from_source)

def check_billing_guidelines(invoice: InvoiceData, guidelines_kb_id: str) -> AssessmentResult:
    print(f"TOOL EXECUTING: check_billing_guidelines(invoice_id='{invoice.invoice_id}', guidelines_kb_id='{guidelines_kb_id}')")
    issues = []
    categorized_spend = {}
    is_compliant = True
    rate_threshold = 400 if guidelines_kb_id == "internal_policy_v2.1" else 600
    for item in invoice.line_items:
        if item.get("rate", 0) > rate_threshold:
            issues.append(f"High rate detected for task '{item.get('description', 'N/A')}': ${item.get('rate')} (Threshold: ${rate_threshold})")
            is_compliant = False
        category = item.get("task_code", "MISC_EXPENSE")[:4]
        categorized_spend[category] = categorized_spend.get(category, 0) + item.get("total", 0)
    if any("Unapproved Item" in item.get("description", "") for item in invoice.line_items):
        issues.append("Contains potentially unapproved line items.")
        is_compliant = False
    if invoice.total_amount > 20000 and guidelines_kb_id == "internal_policy_v2.1":
        issues.append(f"Invoice total amount ${invoice.total_amount} exceeds review threshold of $20,000 for internal policy.")
    return AssessmentResult(
        invoice_id=invoice.invoice_id,
        is_compliant=is_compliant,
        potential_issues=issues,
        categorized_spend=categorized_spend
    )

check_guidelines_tool = FunctionTool(func=check_billing_guidelines)

def generate_spend_trend_report(assessed_data_list: List[AssessmentResult], reporting_period: str) -> AnalysisReport:
    print(f"TOOL EXECUTING: generate_spend_trend_report(period='{reporting_period}', num_assessments={len(assessed_data_list)})")
    if not assessed_data_list:
        return AnalysisReport(
            report_id=f"report_{reporting_period.lower().replace(' ', '_')}_empty",
            summary="No data available for analysis.", trends=[], cost_saving_opportunities=[])
    total_spend = sum(sum(ar.categorized_spend.values()) for ar in assessed_data_list)
    num_non_compliant = sum(1 for ar in assessed_data_list if not ar.is_compliant)
    trends = [
        f"Total spend for {reporting_period}: ${total_spend:.2f}.",
        f"Number of invoices processed: {len(assessed_data_list)}.",
        f"Number of non-compliant invoices: {num_non_compliant}."
    ]
    opportunities = []
    if num_non_compliant > len(assessed_data_list) * 0.1:
        opportunities.append("High percentage of non-compliant invoices. Review common issues and update guidelines or vendor communication.")
    aggregated_categories: Dict[str, float] = {}
    for ar in assessed_data_list:
        for category, amount in ar.categorized_spend.items():
            aggregated_categories[category] = aggregated_categories.get(category, 0) + amount
    if aggregated_categories:
        sorted_categories = sorted(aggregated_categories.items(), key=lambda item: item[1], reverse=True)
        trends.append(f"Top 3 spend categories: {', '.join([f'{cat[0]} (${cat[1]:.2f})' for cat in sorted_categories[:3]])}")
        if sorted_categories and sorted_categories[0][1] > total_spend * 0.4:
            opportunities.append(f"Investigate high spend in top category: '{sorted_categories[0][0]}'. Explore alternatives or volume discounts.")
    return AnalysisReport(
        report_id=f"report_{reporting_period.lower().replace(' ', '_')}",
        summary=f"Spend analysis for {reporting_period}. Total spend: ${total_spend:.2f}. Non-compliant invoices: {num_non_compliant}.",
        trends=trends, cost_saving_opportunities=opportunities)

generate_analysis_tool = FunctionTool(func=generate_spend_trend_report)

# --- Define Agents ---
data_ingestion_agent = LlmAgent(
    name="data_ingestion_agent", 
    model="gemini-2.5-flash-preview-04-17",
    instruction="""Your primary role is to accurately fetch invoice data.
    You will be given a 'source_type' (e.g., 'db', 'pdf') and a 'source_identifier' (e.g., an invoice ID or a file path) as part of the user's request.
    Use the 'fetch_invoice_data_from_source' tool to get the invoice data.
    Ensure the output is the complete InvoiceData object returned by the tool.
    If the tool call fails, report the error clearly.""",
    description="Fetches and performs initial preprocessing of invoice data from various sources.",
    tools=[fetch_invoice_tool], 
    output_key="fetched_invoice_data")

spend_assessment_agent = LlmAgent(
    name="spend_assessment_agent", 
    model="gemini-2.5-flash-preview-04-17",
    instruction="""You are a meticulous legal spend auditor.
    You will receive invoice data under the key 'fetched_invoice_data' from the session state.
    You will also be provided with a 'guidelines_kb_id' (e.g., 'internal_policy_v2.1') in the session state to specify which set of billing rules to apply.
    Use the 'check_billing_guidelines' tool, passing the fetched invoice data and the guidelines_kb_id.
    Return the complete AssessmentResult object from the tool.
    If 'fetched_invoice_data' is missing or invalid, state that you cannot proceed without valid invoice data.""",
    description="Assesses invoice data against predefined legal billing guidelines and compliance rules.",
    tools=[check_guidelines_tool], 
    output_key="assessment_result")

data_analysis_agent = LlmAgent(
    name="data_analysis_agent", 
    model="gemini-2.5-flash-preview-04-17",
    instruction="""You are a sharp data analyst specializing in uncovering insights from legal spend data.
    You will receive a list of assessment results under the key 'list_of_assessment_results' from the session state.
    You will also be given a 'reporting_period' (e.g., "Q2 2025") in the session state.
    Use the 'generate_spend_trend_report' tool with this list and period to generate an analysis report.
    Return the complete AnalysisReport object.
    If 'list_of_assessment_results' is empty or missing, indicate that analysis cannot be performed.""",
    description="Performs deeper analysis on assessed spend data to identify trends, anomalies, and insights.",
    tools=[generate_analysis_tool], 
    output_key="final_analysis_report")

invoice_processing_pipeline = SequentialAgent(
    name="invoice_processing_workflow",
    sub_agents=[data_ingestion_agent, spend_assessment_agent,],
    description="A sequential workflow to ingest and assess a single invoice.")

class SystemCoordinator(LlmAgent):
    all_assessment_results: List[AssessmentResult] = Field(default_factory=list)
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    async def process_one_invoice(self, source_type: str, source_identifier: str, guidelines_kb_id: str, session: Session) -> Optional[AssessmentResult]:
        print(f"\n--- COORDINATOR: Starting processing for invoice: {source_identifier} ---")
        session.state["user_query_for_ingestion"] = f"Process invoice from {source_type} identified by '{source_identifier}'."
        session.state["guidelines_kb_id"] = guidelines_kb_id
        root_event_for_pipeline = Event(author="system_coordinator", content=session.state["user_query_for_ingestion"])
        if not session.run_config:
            if not InvocationContext.get_current_session_service():
                 InvocationContext.set_session_service(InMemorySessionService())
            session.run_config = RunConfig(session_service=InvocationContext.get_current_session_service())
        pipeline_invocation_ctx = InvocationContext(
            agent=invoice_processing_pipeline, root_event=root_event_for_pipeline,
            session=session, run_config=session.run_config)
        async for event in invoice_processing_pipeline.run_async(pipeline_invocation_ctx):
            print(f"PIPELINE EVENT ({event.author} in '{event.branch}'): {event.content.parts[0].text if event.content and event.content.parts else 'Action/State Event'}")
            if event.actions.state_delta:
                print(f"  STATE DELTA by {event.author}: {event.actions.state_delta}")
                session.state.update(event.actions.state_delta)
        assessment_data = session.state.get("assessment_result")
        if assessment_data:
            assessment_obj = AssessmentResult(**assessment_data)
            self.all_assessment_results.append(assessment_obj)
            print(f"COORDINATOR: Invoice '{assessment_obj.invoice_id}' processed. Compliant: {assessment_obj.is_compliant}. Issues: {len(assessment_obj.potential_issues)}")
            return assessment_obj
        else:
            print(f"COORDINATOR: No assessment result found in session state for {source_identifier}.")
            return None
    async def perform_batch_analysis(self, reporting_period: str, session: Session) -> Optional[AnalysisReport]:
        if not self.all_assessment_results:
            print("COORDINATOR: No assessment results available to analyze.")
            return None
        print(f"\n--- COORDINATOR: Starting batch analysis for period: {reporting_period} ---")
        session.state["list_of_assessment_results"] = [ar.model_dump() for ar in self.all_assessment_results]
        session.state["reporting_period"] = reporting_period
        root_event_for_analysis = Event(author="system_coordinator", content=f"Perform batch analysis for {reporting_period}.")
        if not session.run_config:
            if not InvocationContext.get_current_session_service():
                 InvocationContext.set_session_service(InMemorySessionService())
            session.run_config = RunConfig(session_service=InvocationContext.get_current_session_service())
        analysis_invocation_ctx = InvocationContext(
            agent=data_analysis_agent, root_event=root_event_for_analysis,
            session=session, run_config=session.run_config)
        async for event in data_analysis_agent.run_async(analysis_invocation_ctx):
            print(f"ANALYSIS EVENT ({event.author} in '{event.branch}'): {event.content.parts[0].text if event.content and event.content.parts else 'Action/State Event'}")
            if event.actions.state_delta:
                print(f"  STATE DELTA by {event.author}: {event.actions.state_delta}")
                session.state.update(event.actions.state_delta)
        report_data = session.state.get("final_analysis_report")
        if report_data:
            report_obj = AnalysisReport(**report_data)
            print(f"COORDINATOR: Analysis report '{report_obj.report_id}' generated for {reporting_period}.")
            return report_obj
        else:
            print(f"COORDINATOR: No analysis report found in session state for {reporting_period}.")
            return None

system_coordinator_agent = SystemCoordinator(
    name="legal_spend_master_coordinator", model="gemini-2.5-flash-preview-04-17",
    instruction="""You are the top-level orchestrator. Your primary functions are invoked programmatically.
    If given a natural language query, try to understand if the user wants to:
    1. Process a new invoice (ask for source_type, source_identifier, guidelines_kb_id).
    2. Run an analysis report (ask for reporting_period).
    3. Query existing results (this part is not fully implemented via LLM yet).
    For this demo, direct calls to your Python methods will be used.""",
    description="The main orchestrating agent for the legal spend AI system.", tools=[])

async def main_simulation():
    """Simulates the operation of the legal spend system."""
    session_service = InvocationContext.get_current_session_service()
    if not session_service:
        session_service = InMemorySessionService()
        InvocationContext.set_session_service(session_service)

    # Directly get a Session object from create_session()
    current_session: Session = await session_service.create_session()
    
    # Create and assign the RunConfig to the session
    shared_run_config = RunConfig(session_service=session_service)
    current_session.run_config = shared_run_config
    
    # Ensure state is initialized (though Session's default_factory for state should handle this)
    if current_session.state is None: # Or if it's not a dict, or to be absolutely sure it's an empty dict for the simulation
        current_session.state = {}

    print("Starting Legal Spend AI System Simulation...")
    await system_coordinator_agent.process_one_invoice(
        source_type="db", source_identifier="inv-001",
        guidelines_kb_id="internal_policy_v2.1", session=current_session)
    await system_coordinator_agent.process_one_invoice(
        source_type="pdf", source_identifier="invoice_xyz.pdf",
        guidelines_kb_id="external_counsel_guidelines_v1.0", session=current_session)
    analysis_report = await system_coordinator_agent.perform_batch_analysis(
        reporting_period="Q2 2025", session=current_session)
    if analysis_report:
        print("\n--- FINAL ANALYSIS REPORT (Retrieved from Coordinator) ---")
        print(f"Report ID: {analysis_report.report_id}")
        print(f"Summary: {analysis_report.summary}")
        print("Trends:")
        for trend in analysis_report.trends:
            print(f"  - {trend}")
        print("Cost Saving Opportunities:")
        for opp in analysis_report.cost_saving_opportunities:
            print(f"  - {opp}")
    print("\n--- Final Session State ---")
    print(current_session.state)
    print("\nSimulation Complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main_simulation())
    except RuntimeError as e:
        if "cannot run event loop while another loop is running" in str(e):
            print("Skipping main_simulation() as an event loop is already running (likely ADK server).")
        else:
            raise