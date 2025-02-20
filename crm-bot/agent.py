import os
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import TypedDict, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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
# Validador de Queries
# ====================
class QueryValidator:
    def __init__(self, config: dict):
        self.config = config

    def validate(self, query: str) -> dict:
        # Verificar operações proibidas
        forbidden_ops = self.config['security']['forbidden_operations']
        if any(op in query.upper() for op in forbidden_ops):
            return {'valid': False, 'error': 'Operação proibida'}
        
        # Verificar partições obrigatórias
        required_partitions = ['year', 'month', 'canal']
        for partition in required_partitions:
            pattern = fr"{partition}\s*=\s*(?:'[^']+'|\"[^\"]+\"|\d+)"
            if not re.search(pattern, query, re.IGNORECASE):
                return {'valid': False, 'error': f"Filtro obrigatório '{partition}' ausente ou com formato inválido"}
        
        return {'valid': True}

# ====================
# Definição do Estado
# ====================
class AgentState(TypedDict):
    question: str
    filters: Optional[Dict[str, List[str]]]
    raw_query: Optional[str]
    validated_query: Optional[str]
    data: Optional[pd.DataFrame]
    response: Optional[str]
    error: Optional[str]

# ====================
# Núcleo do Agente
# ====================
class MrAgent:
    def __init__(self, table_name: str = 'crm_metadata'):
        self.table_name = table_name
        self.config = ConfigLoader.load_table_config(table_name)
        self.validator = QueryValidator(self.config)
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        self.prompts = PromptManager()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("validate_query", self.validate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)
        
        workflow.set_entry_point("extract_filters")
        
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_conditional_edges(
            "generate_query",
            self.check_query_generated,
            {
                "valid": "validate_query",
                "invalid": END
            }
        )
        workflow.add_conditional_edges(
            "validate_query",
            self.check_query_valid,
            {
                "valid": "execute_query",
                "invalid": END
            }
        )
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
            print(f"Filtros extraídos:\n{filters}")  # DEBUG
            return {**state, "filters": eval(filters)}
        except Exception as e:
            return {**state, "error": f"Erro na extração de filtros: {str(e)}"}

    def generate_query(self, state: AgentState) -> AgentState:
        try:
            config = self.config
            prompt = self.prompts.get_prompt(
                "query_generator",
                question=state['question'],
                table_name=config['name'],
                partitions=", ".join([p['name'] for p in config['partitions']]),
                query_examples="\n".join([ex['sql'] for ex in config['query_examples']]),
                max_rows=config['security']['maximum_rows']
            )
            raw_query = self.llm.invoke(prompt).content
            print(f"Query bruta gerada:\n{raw_query}")  # DEBUG
            return {**state, "raw_query": raw_query}
        except Exception as e:
            return {**state, "error": f"Erro na geração da query: {str(e)}"}

    def check_query_generated(self, state: AgentState) -> str:
        return "valid" if state.get("raw_query") else "invalid"

    def validate_query(self, state: AgentState) -> AgentState:
        validation = self.validator.validate(state['raw_query'])
        if not validation['valid']:
            return {**state, "error": f"Query inválida: {validation['error']}"}
        print(f"Query validada com sucesso:\n{state['raw_query']}")  # DEBUG
        return {**state, "validated_query": state['raw_query']}

    def check_query_valid(self, state: AgentState) -> str:
        return "valid" if state.get("validated_query") else "invalid"

    def execute_query(self, state: AgentState) -> AgentState:
        try:
            df = wr.athena.read_sql_query(
                sql=state['validated_query'],
                database=self.config['database'],
                workgroup=self.config['workgroup'],
                ctas_approach=True
            )
            if df.empty:
                raise ValueError("Nenhum dado retornado pela query")
            print(f"Dados obtidos ({len(df)} linhas):\n{df.head(2)}")  # DEBUG
            return {**state, "data": df}
        except Exception as e:
            return {**state, "error": f"Erro na execução: {str(e)}"}

    def format_response(self, state: AgentState) -> AgentState:
        try:
            prompt = self.prompts.get_prompt(
                "response_formatter",
                inter=state['data'].head(5).to_markdown(),
                question=state['question'],
                filters=state['filters']['filter_expression']
            )
            response = self.llm.invoke(prompt).content
            return {**state, "response": response}
        except Exception as e:
            return {**state, "error": f"Erro na formatação: {str(e)}"}

    def run(self, question: str) -> str:
        initial_state: AgentState = {
            "question": question,
            "filters": None,
            "raw_query": None,
            "validated_query": None,
            "data": None,
            "response": None,
            "error": None
        }
        result = self.workflow.invoke(initial_state)
        return result.get("response") or f"Erro: {result.get('error', 'Erro desconhecido')}"

# Exemplo de uso
if __name__ == "__main__":
    agent = MrAgent()
    resposta = agent.run("Qual a taxa de conversão em janeiro de 2024?")
    print("\nResposta final:\n", resposta)