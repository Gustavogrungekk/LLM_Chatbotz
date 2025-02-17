import os
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import TypedDict, Sequence, Annotated, List, Dict, Optional
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# ====================
# Configurações Base
# ====================
class ConfigLoader:
    @staticmethod
    def load_table_config(table_name: str) -> dict:
        with open(Path(f'config/tables/{table_name}.yaml'), 'r') as f:
            return yaml.safe_load(f)['table_config']
    
    @staticmethod
    def load_personas() -> dict:
        with open(Path('config/prompts/personas.yaml'), 'r') as f:
            return yaml.safe_load(f)

# ====================
# Módulo de Prompts
# ====================
class PromptManager:
    def __init__(self):
        self.personas = ConfigLoader.load_personas()
    
    def get_prompt(self, persona: str, **kwargs) -> str:
        return self.personas[persona]['template'].format(**kwargs)

# ====================
# Gerenciador Athena
# ====================
class AthenaManager:
    def __init__(self, table_name: str):
        self.config = ConfigLoader.load_table_config(table_name)
        self.validator = QueryValidator(self.config)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        validation = self.validator.validate(query)
        if not validation['valid']:
            raise ValueError(validation['error'])
        
        try:
            return wr.athena.read_sql_query(
                sql=query + f" LIMIT {self.config['security']['maximum_rows']}",
                database=self.config['database'],
                workgroup=self.config['workgroup'],
                ctas_approach=True
            )
        except Exception as e:
            raise RuntimeError(f"Erro Athena: {str(e)}")

class QueryValidator:
    def __init__(self, config: dict):
        self.config = config
    
    def validate(self, query: str) -> dict:
        # Verificar operações proibidas
        if any(op in query.upper() for op in self.config['security']['forbidden_operations']):
            return {'valid': False, 'error': 'Operação proibida'}
        
        # Verificar partições obrigatórias
        required_partitions = ['year', 'month', 'canal']
        for partition in required_partitions:
            if not re.search(fr"{partition}\s*=\s*'[^']+'", query, re.IGNORECASE):
                return {'valid': False, 'error': f"Filtro {partition} ausente ou formato incorreto"}
        
        return {'valid': True}

# ====================
# Definição do Estado
# ====================
class AgentState(TypedDict):
    question: str
    filters: Optional[Dict[str, List[str]]]
    query: Optional[str]
    data: Optional[pd.DataFrame]
    response: Optional[str]
    error: Optional[str]

# ====================
# Núcleo do Agente
# ====================
class MrAgent:
    def __init__(self, table_name: str = 'crm_metadata'):
        self.table_name = table_name
        self.athena = AthenaManager(table_name)
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        self.prompts = PromptManager()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Adiciona nós ao fluxo de trabalho
        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)
        
        # Define o ponto de entrada
        workflow.set_entry_point("extract_filters")
        
        # Conecta os nós
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def extract_filters(self, state: AgentState) -> AgentState:
        try:
            prompt = self.prompts.get_prompt(
                "date_extractor",
                current_date=datetime.now().strftime('%Y-%m-%d')
            )
            filters = self.llm.invoke(prompt).content
            return {**state, "filters": eval(filters)}  # Converte string para dict
        except Exception as e:
            return {**state, "error": f"Erro extração datas: {str(e)}"}
    
    def generate_query(self, state: AgentState) -> AgentState:
        try:
            config = self.athena.config
            prompt = self.prompts.get_prompt(
                "query_generator",
                question=state['question'],
                table_name=config['name'],
                partitions=", ".join([p['name'] for p in config['partitions']]),
                ignore_values=str([col['ignore_values'] for col in config['columns'] if 'ignore_values' in col]),
                query_examples="\n".join([ex['sql'] for ex in config['query_examples']]),
                max_rows=config['security']['maximum_rows']
            )
            return {**state, "query": self.llm.invoke(prompt).content}
        except Exception as e:
            return {**state, "error": f"Erro geração query: {str(e)}"}
    
    def execute_query(self, state: AgentState) -> AgentState:
        try:
            if not state.get("query"):
                return {**state, "error": "Query não gerada corretamente"}
            
            df = self.athena.execute_query(state['query'])
            if df.empty:
                return {**state, "error": "Nenhum dado encontrado para a query"}
            return {**state, "data": df}
        except Exception as e:
            return {**state, "error": str(e)}
    
    def format_response(self, state: AgentState) -> AgentState:
        try:
            if not state.get("data"):
                return {**state, "error": "Dados não disponíveis para formatação"}
            
            prompt = self.prompts.get_prompt(
                "response_formatter",
                inter=state['data'].head(5).to_markdown(),
                question=state['question'],
                filters=state['filters']['filter_expression']
            )
            response = self.llm.invoke(prompt).content
            return {**state, "response": response}
        except Exception as e:
            return {**state, "error": f"Erro formatação: {str(e)}"}
    
    def run(self, question: str) -> str:
        # Inicializa o estado
        initial_state: AgentState = {
            "question": question,
            "filters": None,
            "query": None,
            "data": None,
            "response": None,
            "error": None
        }
        
        # Executa o fluxo de trabalho
        result = self.workflow.invoke(initial_state)
        
        # Retorna a resposta ou o erro
        if result.get("error"):
            return f"Erro: {result['error']}"
        return result.get("response", "Nenhum resultado encontrado.")

# ====================
# Exemplo de Uso
# ====================
if __name__ == "__main__":
    agent = MrAgent()
    
    # Consulta válida
    print(agent.run("Qual o potencial médio por produto no canal VAI nos últimos 2 meses?"))
    
    # Consulta inválida
    print(agent.run("DELETE FROM tabela WHERE year = 2023"))