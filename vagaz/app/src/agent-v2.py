# app/src/advanced_agent.py

import yaml
import time
import awswrangler as wr
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END  # Make sure START/END are imported as needed

def get_llm():
    with open('config/llm_config.yaml', 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4-0125-preview"),
        temperature=llm_config.get("temperature", 0),
        seed=llm_config.get("seed", 1)
    )

# --- Sub-agents ---

class CuriosityAgent:
    def __init__(self, topic="Banco Itaú", prompt_template=None):
        self.topic = topic
        self.llm = get_llm()
        if prompt_template is None:
            self.prompt_template = (
                "Você é um especialista em curiosidades sobre {topic}. "
                "Forneça uma curiosidade interessante, exclusiva e concisa sobre {topic}."
            )
        else:
            self.prompt_template = prompt_template

    def get_curiosity(self) -> str:
        template = PromptTemplate(template=self.prompt_template, input_variables=["topic"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(topic=self.topic)

class ContextEnricher:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.llm = get_llm()
    
    def enrich(self, context: str) -> str:
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context)

class DateExtractor:
    def __init__(self, prompt: str, fixed_date_enabled: bool = False, fixed_date_value: str = None):
        self.prompt = prompt
        self.llm = get_llm()
        self.fixed_date_enabled = fixed_date_enabled
        self.fixed_date_value = fixed_date_value
    
    def extract(self, context: str) -> str:
        if self.fixed_date_enabled and self.fixed_date_value:
            return self.fixed_date_value
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context)

class QueryBuilder:
    def __init__(self, prompt: str, metadata: dict):
        self.prompt = prompt
        self.metadata = metadata
        self.llm = get_llm()
    
    def build(self, context: str, date_info: str) -> str:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "date_info", "metadata"]
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context, date_info=date_info, metadata=self.metadata)

class InsightsAgent:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.llm = get_llm()
    
    def generate(self, df: pd.DataFrame) -> str:
        data_str = df.to_csv(index=False)
        template = PromptTemplate(template=self.prompt, input_variables=["data"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(data=data_str)

class DataVizAgent:
    def __init__(self, prompt: str, config_path: str = "config/dataviz_config.yaml"):
        self.prompt = prompt
        with open(config_path, "r", encoding="utf-8") as f:
            self.viz_config = yaml.safe_load(f)
        self.library = "plotly"  # Default library
        self.llm = get_llm()
    
    def choose_plot_type(self, metadata: str, df: pd.DataFrame) -> str:
        available = list(self.viz_config.get("available_plots", {}).keys())
        prompt_template = (
            "Você é um especialista em visualização de dados. Com base nos metadados abaixo:\n\n"
            "{metadata}\n\n"
            "E considerando as colunas disponíveis no DataFrame: {columns}, "
            "escolha o tipo de gráfico mais adequado dentre as seguintes opções: {options}. "
            "Retorne apenas o nome do gráfico (exemplo: 'bar', 'line', 'scatter' ou 'pie')."
        )
        columns = ", ".join(df.columns)
        options = ", ".join(available)
        formatted_prompt = prompt_template.format(metadata=metadata, columns=columns, options=options)
        template = PromptTemplate(template=formatted_prompt, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=template)
        chosen_plot = chain.run()
        if chosen_plot.strip().lower() not in available:
            return self.viz_config.get("default", "bar")
        return chosen_plot.strip().lower()
    
    def plot(self, df: pd.DataFrame, metadata: str = "") -> str:
        import plotly.express as px
        plot_type = self.choose_plot_type(metadata, df)
        if self.library == "plotly":
            if plot_type == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualização do Potencial")
            elif plot_type == "line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Tendência ao longo do tempo")
            elif plot_type == "scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Correlação entre variáveis")
            elif plot_type == "pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Proporção por Categoria")
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualização do Potencial")
            return fig.to_html()
        else:
            import matplotlib.pyplot as plt
            if plot_type == "bar":
                fig, ax = plt.subplots()
                ax.bar(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Visualização do Potencial")
            elif plot_type == "line":
                fig, ax = plt.subplots()
                ax.plot(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Tendência ao longo do tempo")
            elif plot_type == "scatter":
                fig, ax = plt.subplots()
                ax.scatter(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Correlação entre variáveis")
            elif plot_type == "pie":
                fig, ax = plt.subplots()
                ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
                ax.set_title("Proporção por Categoria")
            else:
                fig, ax = plt.subplots()
                ax.bar(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Visualização do Potencial")
            return "Visualização gerada com matplotlib."

# --- AdvancedAgent using a TypedDict state schema ---

# Define the state schema using TypedDict
class AgentState(TypedDict):
    input: str
    enriched_context: str
    date_info: str
    query: str
    insights: str
    visualization: str
    response: str
    error: str
    # Note: 'df' is not included in the schema because it is unhashable.

class AdvancedAgent:
    def __init__(self):
        with open('data/metadata/metadata.yaml', 'r', encoding='utf-8') as meta_file:
            self.metadata = yaml.safe_load(meta_file)
        with open('data/prompts/prompts.yaml', 'r', encoding='utf-8') as prompts_file:
            self.prompts = yaml.safe_load(prompts_file)
        
        self.llm = get_llm()
        
        test_mode = self.prompts.get('test_mode', False)
        fixed_date_enabled = self.prompts.get('fixed_date_enabled', False) if test_mode else False
        fixed_date_value = self.prompts.get('fixed_date_value', None) if fixed_date_enabled else None

        self.context_enricher = ContextEnricher(self.prompts.get('context_enrichment'))
        self.date_extractor = DateExtractor(self.prompts.get('date_extraction'), fixed_date_enabled, fixed_date_value)
        self.query_builder = QueryBuilder(self.prompts.get('query_builder'), self.metadata)
        self.insights_agent = InsightsAgent(self.prompts.get('insights'))
        self.dataviz_agent = DataVizAgent(self.prompts.get('dataviz'))
        self.curiosity_agent = CuriosityAgent(topic="Banco Itaú")
        self.forbidden_topics = self.prompts.get('forbidden_topics', ['entertainment', 'politics'])
        
        self.build_workflow()

    def build_workflow(self):
        # Create the state graph with the state schema (TypedDict)
        sg = StateGraph(AgentState)
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

    # Each state node receives a state (as a dict conforming to AgentState) and returns a full dict update.
    def state_enrich_context(self, state: AgentState) -> dict:
        enriched = self.context_enricher.enrich(state["input"])
        # Return full state update:
        return {**state, "enriched_context": enriched}

    def state_validate_context(self, state: AgentState) -> dict:
        for topic in self.forbidden_topics:
            if topic.lower() in state["enriched_context"].lower():
                return {**state, "error": f"A consulta contém o tópico '{topic}', que está fora do escopo permitido. Por favor, reformule a pergunta."}
        return state

    def state_extract_dates(self, state: AgentState) -> dict:
        dates = self.date_extractor.extract(state["enriched_context"])
        return {**state, "date_info": dates}

    def state_build_query(self, state: AgentState) -> dict:
        query = self.query_builder.build(state["enriched_context"], state["date_info"])
        return {**state, "query": query}

    def state_execute_query(self, state: AgentState) -> dict:
        query = state["query"]
        max_attempts = 3
        attempt = 0
        timeout_minutes = 2
        start_time = time.time()
        df = None

        while attempt < max_attempts:
            try:
                df = wr.athena.read_sql_query(
                    sql=query,
                    database=self.metadata["table_config"]["database"],
                    workgroup=self.metadata["table_config"]["workgroup"]
                )
                break
            except Exception as e:
                attempt += 1
                elapsed = time.time() - start_time
                if elapsed > timeout_minutes * 60:
                    return {**state, "error": "A consulta demorou demais ou encontrou um erro."}
                curiosity = self.curiosity_agent.get_curiosity()
                print(f"Enquanto consultamos os dados, aqui vai uma curiosidade: {curiosity} | Tentativa: {attempt}")
                time.sleep(10)
        # Attach the DataFrame separately (not in state schema)
        state_update = {**state, "query": state["query"]}  # preserving current state
        state_update["df"] = df  # not part of schema but will be used downstream
        return state_update

    def state_generate_insights(self, state: AgentState) -> dict:
        df = state.get("df")
        insights = self.insights_agent.generate(df)
        return {**state, "insights": insights}

    def state_generate_visualization(self, state: AgentState) -> dict:
        df = state.get("df")
        viz = self.dataviz_agent.plot(df, metadata=str(self.metadata))
        return {**state, "visualization": viz}

    def state_compose_response(self, state: AgentState) -> dict:
        response = (
            "**Resposta do Agente:**\n\n"
            "📊 **Resultados da Análise**\n"
            f"{state.get('insights', '')}\n\n"
            "📈 **Visualização:**\n"
            f"{state.get('visualization', '')}\n\n"
            "🔧 **Query Executada:**\n"
            f"```sql\n{state.get('query', '')}\n```"
        )
        return {**state, "response": response}

    def run(self, input_data: dict) -> dict:
        context = input_data.get("context", "")
        initial_state: AgentState = {
            "input": context,
            "enriched_context": "",
            "date_info": "",
            "query": "",
            "insights": "",
            "visualization": "",
            "response": "",
            "error": ""
        }
        final_state = self.workflow.invoke(initial_state)
        if final_state.get("error"):
            return {"error": final_state["error"]}
        return {"response": final_state.get("response", "Nenhuma resposta gerada.")}

# Example execution of the advanced agent.
if __name__ == "__main__":
    agent = AdvancedAgent()
    result = agent.run({"context": "Quais foram os produtos que mais tiveram visto e clique nos canais FOLHA e TREVOS?"})
    print(result)
