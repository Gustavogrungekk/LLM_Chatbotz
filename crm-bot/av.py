import os
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import TypedDict, Sequence, Annotated
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
        workflow = StateGraph(TypedDict)
        
        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)
        
        workflow.set_entry_point("extract_filters")
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def extract_filters(self, state: dict):
        try:
            prompt = self.prompts.get_prompt(
                "date_extractor",
                current_date=datetime.now().strftime('%Y-%m-%d')
            )
            filters = self.llm.invoke(prompt).content
            return {'filters': eval(filters)}  # Converte string para dict
        except Exception as e:
            return {'error': f"Erro extração datas: {str(e)}"}
    
    def generate_query(self, state: dict):
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
            return {'query': self.llm.invoke(prompt).content}
        except Exception as e:
            return {'error': f"Erro geração query: {str(e)}"}
    
    def execute_query(self, state: dict):
        try:
            df = self.athena.execute_query(state['query'])
            return {'data': df}
        except Exception as e:
            return {'error': str(e)}
    
    def format_response(self, state: dict):
        try:
            prompt = self.prompts.get_prompt(
                "response_formatter",
                inter=state['data'].head(5).to_markdown(),
                question=state['question'],
                filters=state['filters']['filter_expression']
            )
            response = self.llm.invoke(prompt).content
            return {'response': response}
        except Exception as e:
            return {'error': f"Erro formatação: {str(e)}"}
    
    def run(self, question: str):
        result = self.workflow.invoke({'question': question})
        return result.get('response', 'Erro ao processar solicitação')

# ====================
# Exemplo de Uso
# ====================
if __name__ == "__main__":
    agent = MrAgent()
    
    # Consulta válida
    print(agent.run("Qual o potencial médio por produto no canal VAI nos últimos 2 meses?"))
    
    # Consulta inválida
    print(agent.run("DELETE FROM tabela WHERE year = 2023"))
    
    
    
    new 
import os
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import TypedDict, Sequence, Annotated, Dict
from langchain_core.messages import AIMessage, BaseMessage
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
        if any(op in query.upper() for op in self.config['security']['forbidden_operations']):
            return {'valid': False, 'error': 'Operação proibida'}
        
        required_partitions = ['year', 'month', 'canal']
        for partition in required_partitions:
            if not re.search(fr"{partition}\s*=\s*'[^']+'", query, re.IGNORECASE):
                return {'valid': False, 'error': f"Filtro {partition} ausente ou formato incorreto"}
        
        return {'valid': True}

# ====================
# Definição do Estado (Workflow State)
# ====================
class WorkflowState(TypedDict):
    question: str
    filters: dict
    query: str
    data: pd.DataFrame
    response: str
    error: str

# ====================
# Núcleo do Agente (LangGraph)
# ====================
class MrAgent:
    def __init__(self, table_name: str = 'crm_metadata'):
        self.table_name = table_name
        self.athena = AthenaManager(table_name)
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        self.prompts = PromptManager()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(WorkflowState)  # Corrigindo o Schema

        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)

        workflow.set_entry_point("extract_filters")
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def extract_filters(self, state: WorkflowState) -> Dict:
        try:
            prompt = self.prompts.get_prompt(
                "date_extractor",
                current_date=datetime.now().strftime('%Y-%m-%d')
            )
            filters = self.llm.invoke(prompt).content
            return {'filters': eval(filters)}  # Converte string para dict
        except Exception as e:
            return {'error': f"Erro extração datas: {str(e)}"}

    def generate_query(self, state: WorkflowState) -> Dict:
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
            return {'query': self.llm.invoke(prompt).content}
        except Exception as e:
            return {'error': f"Erro geração query: {str(e)}"}

    def execute_query(self, state: WorkflowState) -> Dict:
        try:
            df = self.athena.execute_query(state['query'])
            return {'data': df}
        except Exception as e:
            return {'error': str(e)}

    def format_response(self, state: WorkflowState) -> Dict:
        try:
            prompt = self.prompts.get_prompt(
                "response_formatter",
                inter=state['data'].head(5).to_markdown(),
                question=state['question'],
                filters=state['filters']['filter_expression']
            )
            response = self.llm.invoke(prompt).content
            return {'response': response}
        except Exception as e:
            return {'error': f"Erro formatação: {str(e)}"}

    def run(self, question: str):
        result = self.workflow.invoke({'question': question})
        return result.get('response', 'Erro ao processar solicitação')

# ====================
# Exemplo de Uso
# ====================
if __name__ == "__main__":
    agent = MrAgent()
    
    print(agent.run("Qual foi o potencial para o canal VAI em janeiro de 2025?"))
    print(agent.run("DELETE FROM tabela WHERE year = 2023"))
    
    
    
    cxxxxx
    import os
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ====================
# Configurações Base
# ====================
class ConfigLoader:
    @staticmethod
    def load_table_config(table_name: str) -> dict:
        """Carrega a configuração da tabela a partir de um arquivo YAML."""
        with open(Path(f'config/tables/{table_name}.yaml'), 'r') as f:
            return yaml.safe_load(f)['table_config']
    
    @staticmethod
    def load_personas() -> dict:
        """Carrega os templates de prompts a partir de um arquivo YAML."""
        with open(Path('config/prompts/personas.yaml'), 'r') as f:
            return yaml.safe_load(f)

# ====================
# Módulo de Prompts
# ====================
class PromptManager:
    def __init__(self):
        self.personas = ConfigLoader.load_personas()
    
    def get_prompt(self, persona: str, **kwargs) -> str:
        """Retorna um prompt formatado com base em uma persona."""
        if persona not in self.personas:
            raise ValueError(f"Persona '{persona}' não encontrada")
        return self.personas[persona]['template'].format(**kwargs)

# ====================
# Gerenciador Athena
# ====================
class AthenaManager:
    def __init__(self, table_name: str):
        """Inicializa o gerenciador com as configurações da tabela."""
        self.config = ConfigLoader.load_table_config(table_name)
        self.validator = QueryValidator(self.config)

    def run_query(self, query: str) -> pd.DataFrame:
        """Executa uma query simples no Athena."""
        try:
            return wr.athena.read_sql_query(
                sql=query,
                database=self.config['database'],
                workgroup=self.config['workgroup'],
                ctas_approach=True
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao executar query no Athena: {str(e)}")

    def execute_safe_query(self, query: str) -> pd.DataFrame:
        """Valida e executa a query segura no Athena."""
        validation = self.validator.validate(query)
        if not validation['valid']:
            raise ValueError(f"Query inválida: {validation['error']}")
        
        query += f" LIMIT {self.config['security']['maximum_rows']}"
        return self.run_query(query)

class QueryValidator:
    def __init__(self, config: dict):
        """Inicializa o validador de queries com regras de segurança."""
        self.config = config
    
    def validate(self, query: str) -> dict:
        """Valida a query com base nas operações proibidas e partições obrigatórias."""
        if any(op in query.upper() for op in self.config['security']['forbidden_operations']):
            return {'valid': False, 'error': 'Operação proibida'}
        
        required_partitions = ['year', 'month', 'canal']
        for partition in required_partitions:
            if not re.search(fr"{partition}\s*=\s*'[^']+'", query, re.IGNORECASE):
                return {'valid': False, 'error': f"Filtro obrigatório: {partition}"}
        
        return {'valid': True}

# ====================
# Definição do Estado do Workflow
# ====================
class WorkflowState(TypedDict):
    question: str
    filters: dict
    query: str
    data: pd.DataFrame
    response: str
    error: str

# ====================
# Núcleo do Agente (LangGraph)
# ====================
class MrAgent:
    def __init__(self, table_name: str = 'crm_metadata'):
        self.athena = AthenaManager(table_name)
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        self.prompts = PromptManager()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Monta o fluxo de execução do agente usando LangGraph."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)

        workflow.set_entry_point("extract_filters")
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def extract_filters(self, state: WorkflowState) -> Dict:
        """Extrai filtros de data ou canal usando LLM."""
        try:
            prompt = self.prompts.get_prompt(
                "date_extractor",
                current_date=datetime.now().strftime('%Y-%m-%d')
            )
            filters = self.llm.invoke(prompt).content
            return {'filters': eval(filters)}  # Converte string para dict
        except Exception as e:
            return {'error': f"Erro ao extrair filtros: {str(e)}"}

    def generate_query(self, state: WorkflowState) -> Dict:
        """Gera uma query SQL com base na pergunta e filtros usando LLM."""
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
            query = self.llm.invoke(prompt).content
            return {'query': query}
        except Exception as e:
            return {'error': f"Erro ao gerar query: {str(e)}"}

    def execute_query(self, state: WorkflowState) -> Dict:
        """Executa a query no Athena."""
        try:
            df = self.athena.execute_safe_query(state['query'])
            return {'data': df}
        except Exception as e:
            return {'error': f"Erro ao executar query: {str(e)}"}

    def format_response(self, state: WorkflowState) -> Dict:
        """Formata a resposta final usando LLM."""
        try:
            prompt = self.prompts.get_prompt(
                "response_formatter",
                inter=state['data'].head(5).to_markdown(),
                question=state['question'],
                filters=state['filters'].get('filter_expression', '')
            )
            response = self.llm.invoke(prompt).content
            return {'response': response}
        except Exception as e:
            return {'error': f"Erro ao formatar resposta: {str(e)}"}

    def run(self, question: str):
        """Executa o workflow completo com base na pergunta."""
        result = self.workflow.invoke({'question': question})
        if 'error' in result:
            return f"Erro: {result['error']}"
        return result.get('response', 'Erro desconhecido')

# ====================
# Exemplo de Uso
# ====================
if __name__ == "__main__":
    agent = MrAgent()

    # Pergunta válida
    print(agent.run("Qual o potencial médio para o canal VAI em janeiro de 2025?"))

    # Pergunta inválida
    print(agent.run("DELETE FROM tabela WHERE year = 2023"))
    
    
    
v3
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
            df = self.athena.execute_query(state['query'])
            return {**state, "data": df}
        except Exception as e:
            return {**state, "error": str(e)}
    
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
    
    