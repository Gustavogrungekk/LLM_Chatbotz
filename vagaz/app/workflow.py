def build_workflow(self):
    # Define a state schema that excludes unhashable fields (like 'df')
    state_schema = {
        "input": str,
        "enriched_context": str,
        "date_info": str,
        "query": str,
        "insights": str,
        "visualization": str,
        "response": str,
        "error": str,
    }
    sg = StateGraph(state_schema)
    sg.add_node("enrich_context", self.state_enrich_context)
    sg.add_node("validate_context", self.state_validate_context)
    sg.add_node("extract_dates", self.state_extract_dates)
    sg.add_node("build_query", self.state_build_query)
    sg.add_node("execute_query", self.state_execute_query)
    sg.add_node("generate_insights", self.state_generate_insights)
    sg.add_node("generate_visualization", self.state_generate_visualization)
    sg.add_node("compose_response", self.state_compose_response)
    
    sg.set_entry_point("enrich_context")
    sg.add_edge("enrich_context", "validate_context")
    sg.add_edge("validate_context", "extract_dates")
    sg.add_edge("extract_dates", "build_query")
    sg.add_edge("build_query", "execute_query")
    sg.add_edge("execute_query", "generate_insights")
    sg.add_edge("generate_insights", "generate_visualization")
    sg.add_edge("generate_visualization", "compose_response")
    sg.add_edge("compose_response", END)
    
    self.workflow = sg.compile()