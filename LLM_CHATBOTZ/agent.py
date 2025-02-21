import os
import sys
import yaml
import random
import time
import awswrangler as wr
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END  # Certifique-se de que esse módulo esteja instalado e configurado
from typing import TypedDict

#==== DELL: Carregar a chave da OpenAI
with open(r'C:\Users\mygam\OneDrive\Área de Trabalho\Scripts\solo-projects\mrbot\config\open_ia_key.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    api_key = config['key']

os.environ['OPENAI_API_KEY'] = api_key

# Função para carregar a configuração do LLM
def get_llm():
    with open('config/llm_config.yaml', 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4-0125-preview"),
        temperature=llm_config.get("temperature", 0),
        seed=llm_config.get("seed", 1)
    )

# Instancia o modelo uma única vez
global_llm = get_llm()

class RandomCuriosityAgent:
    def __init__(self):
        with open('config/curiosidades.yaml', "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.curiosidades = data.get("curiosidades", [])

    def get_curiosity(self) -> str:
        if not self.curiosidades:
            return "Sem curiosidades disponíveis."
        curiosidade = random.choice(self.curiosidades)
        return f"{curiosidade.get('titulo', '')}: {curiosidade.get('descricao', '')}"

# --- Sub-agents ---
class CuriosityAgent:
    def __init__(self, llm, topic="Banco Itaú", prompt_template=None):
        self.topic = topic
        self.llm = llm
        if prompt_template is None:
            self.prompt_template = (
                "Você é um especialista em curiosidades sobre {topic}. "
                "Forneça uma curiosidade interessante, exclusiva e concisa sobre {topic}."
            )
        else:
            self.prompt_template = prompt_template

    def get_curiosity(self) -> str:
        template = PromptTemplate(template=self.prompt_template, input_variables=["topic"])
        result = (template | self.llm).invoke({"topic": self.topic})
        return result

class ContextEnricher:
    def __init__(self, llm, prompt: str):
        self.prompt = prompt
        self.llm = llm
    
    def enrich(self, context: str) -> str:
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        result = (template | self.llm).invoke({"context": context})
        return result

class DateExtractor:
    def __init__(self, llm, prompt: str, fixed_date_enabled: bool = False, fixed_date_value: str = None):
        self.prompt = prompt
        self.llm = llm
        self.fixed_date_enabled = fixed_date_enabled
        self.fixed_date_value = fixed_date_value
    
    def extract(self, context: str) -> str:
        if self.fixed_date_enabled and self.fixed_date_value:
            return self.fixed_date_value
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        result = (template | self.llm).invoke({"context": context})
        return result

class QueryBuilder:
    def __init__(self, llm, prompt: str, metadata: dict):
        self.prompt = prompt
        self.metadata = metadata
        self.llm = llm
    
    def build(self, context: str, date_info: str) -> str:
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "date_info", "metadata"]
        )
        result = (template | self.llm).invoke({
            "context": context,
            "date_info": date_info,
            "metadata": self.metadata
        })
        return result

class InsightsAgent:
    def __init__(self, llm, prompt: str):
        self.prompt = prompt
        self.llm = llm
    
    def generate(self, df: pd.DataFrame) -> str:
        data_str = df.to_csv(index=False)
        template = PromptTemplate(template=self.prompt, input_variables=["data"])
        result = (template | self.llm).invoke({"data": data_str})
        return result

class DataVizAgent:
    def __init__(self, llm, prompt: str, config_path: str = "config/dataviz_config.yaml"):
        self.prompt = prompt
        with open(config_path, "r", encoding="utf-8") as f:
            self.viz_config = yaml.safe_load(f)
        self.library = "plotly"
        self.llm = llm
    
    def choose_plot_type(self, metadata: str, df: pd.DataFrame) -> str:
        if df is None or not hasattr(df, "columns"):
            return self.viz_config.get("default", "bar")
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
        chosen_plot = (template | self.llm).invoke({})
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

# --- AdvancedAgent usando um schema TypedDict (sem histórico) ---
class AgentState(TypedDict):
    input: str
    enriched_context: str
    date_info: str
    query: str
    insights: str
    visualization: str
    response: str
    error: str

class AdvancedAgent:
    def __init__(self):
        with open('data/metadata/metadata.yaml', 'r', encoding='utf-8') as meta_file:
            self.metadata = yaml.safe_load(meta_file)
        with open('data/prompts/prompts.yaml', 'r', encoding='utf-8') as prompts_file:
            self.prompts = yaml.safe_load(prompts_file)
        
        self.llm = global_llm

        test_mode = self.prompts.get('test_mode', False)
        fixed_date_enabled = self.prompts.get('fixed_date_enabled', False) if test_mode else False
        fixed_date_value = self.prompts.get('fixed_date_value', None) if fixed_date_enabled else None

        self.context_enricher = ContextEnricher(self.llm, self.prompts.get('context_enrichment'))
        self.date_extractor = DateExtractor(self.llm, self.prompts.get('date_extraction'), fixed_date_enabled, fixed_date_value)
        self.query_builder = QueryBuilder(self.llm, self.prompts.get('query_builder'), self.metadata)
        self.insights_agent = InsightsAgent(self.llm, self.prompts.get('insights'))
        self.dataviz_agent = DataVizAgent(self.llm, self.prompts.get('dataviz'))
        self.curiosity_agent = CuriosityAgent(self.llm, topic="Banco Itaú")
        self.random_curiosity_agent = RandomCuriosityAgent()
        
        self.forbidden_topics = self.prompts.get('forbidden_topics', ['entertainment', 'politics'])
        
        self.build_workflow()

    def build_workflow(self):
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

    def state_enrich_context(self, state: AgentState) -> dict:
        enriched = self.context_enricher.enrich(state["input"])
        return {**state, "enriched_context": enriched}

    def state_validate_context(self, state: AgentState) -> dict:
        enriched_context_str = str(state["enriched_context"])
        for topic in self.forbidden_topics:
            if topic.lower() in enriched_context_str.lower():
                return {**state, "error": f"A consulta contém o tópico '{topic}', que está fora do escopo permitido. Por favor, reformule a pergunta."}
        return state

    def state_extract_dates(self, state: AgentState) -> dict:
        dates = self.date_extractor.extract(state["enriched_context"])
        return {**state, "date_info": dates}

    def state_build_query(self, state: AgentState) -> dict:
        query_output = self.query_builder.build(state["enriched_context"], state["date_info"])
        if isinstance(query_output, str):
            query_sql = query_output
        elif hasattr(query_output, "content"):
            query_sql = query_output.content
        else:
            query_sql = ""
        print("Query gerada:", query_sql)
        return {**state, "query": query_sql}

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
                curiosidade_atual = self.random_curiosity_agent.get_curiosity()
                print(f"Enquanto consultamos os dados, aqui vai uma curiosidade: {curiosidade_atual} | Tentativa: {attempt}")
                time.sleep(10)
        state_update = {**state, "query": state["query"]}
        state_update["df"] = df
        return state_update

    def state_generate_insights(self, state: AgentState) -> dict:
        df = state.get("df")
        if df is None:
            return {**state, "error": "Não foi possível executar a query. Já notificamos nosso time. Por favor, tente novamente."}
        try:
            insights = self.insights_agent.generate(df)
        except Exception as e:
            return {**state, "error": "Ocorreu um erro ao gerar os insights. Já notificamos nosso time. Por favor, tente novamente."}
        return {**state, "insights": insights}

    def state_generate_visualization(self, state: AgentState) -> dict:
        df = state.get("df")
        if df is None:
            return {**state, "error": "A consulta não retornou dados para visualização. Já notificamos nosso time. Por favor, tente novamente."}
        try:
            viz = self.dataviz_agent.plot(df, metadata=str(self.metadata))
        except Exception as e:
            return {**state, "error": "Ocorreu um erro ao gerar a visualização. Já notificamos nosso time. Por favor, tente novamente."}
        return {**state, "visualization": viz}

    # Função atualizada: retorna somente a resposta final gerada (campo 'insights')
    def state_compose_response(self, state: AgentState) -> dict:
        response = state.get('insights', 'Nenhuma resposta gerada.')
        return {**state, "response": response}

    def run(self, input_data: dict) -> dict:
        new_input = "Pergunta: " + input_data.get("context", "")
        initial_state: AgentState = {
            "input": new_input,
            "enriched_context": "",
            "date_info": "",
            "query": "",
            "insights": "",
            "visualization": "",
            "response": "",
            "error": ""
        }
        final_state = self.workflow.invoke(initial_state)
        return {
            "response": final_state.get("response", "Nenhuma resposta gerada."),
            "query": final_state.get("query", ""),
            "insights": final_state.get("insights", ""),
            "graph": final_state.get("visualization", "")
        }

# Exemplo de execução:
if __name__ == "__main__":
    agent = AdvancedAgent()
    result = agent.run({"context": "Quais foram os produos com maior potencial visto e clique nos canais Email VAI e MCE?"})
    print(result)
