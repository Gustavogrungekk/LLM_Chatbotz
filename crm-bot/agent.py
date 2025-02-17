# ====================
# Python Standard Libraries
# ====================
import os
from pathlib import Path

# ====================
# Data Manipulation & Time Libraries
# ====================
import yaml
import re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ====================
# Typing Libraries
# ====================
from typing import TypedDict, List, Dict

# ====================
# LangChain Libraries
# ====================
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ====================
# Data Visualization Libraries
# ====================
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# ====================
# Image Processing Libraries
# ====================
from io import BytesIO
import base64

# ============================================================
# INÍCIO DO CÓDIGO - AGENTE DE CONSULTAS E VISUALIZAÇÃO-CRM
# ============================================================
# Este código gerencia a execução de consultas SQL no AWS Athena,
# processa os resultados, e gera visualizações com Plotly, Seaborn e Matplotlib.
# 
# A sequência de etapas inclui:
# 1. Extração de filtros de datas
# 2. Geração de queries SQL
# 3. Execução de queries no Athena
# 4. Formatação e visualização dos resultados
# ============================================================


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
            if not re.search(fr"{partition}\s*=[\s'\d]+", query, re.IGNORECASE):
                return {'valid': False, 'error': f"Filtro {partition} ausente"}
        
        return {'valid': True}

# ====================
# Data Viz Expert
# ====================
class DataVizExpert:
    def __init__(self):
        pass
    
    def create_bar_chart(self, df, x_col, y_col, title="Gráfico de Barras"):
        """Cria um gráfico de barras com Plotly."""
        fig = px.bar(df, x=x_col, y=y_col, title=title)
        return self._encode_image(fig)

    def create_line_chart(self, df, x_col, y_col, title="Gráfico de Linhas"):
        """Cria um gráfico de linhas com Plotly."""
        fig = px.line(df, x=x_col, y=y_col, title=title)
        return self._encode_image(fig)

    def create_histogram(self, df, column, title="Histograma"):
        """Cria um histograma com Seaborn."""
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, color='blue')
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequência')
        return self._encode_image(plt)

    def _encode_image(self, figure):
        """Codifica a figura gerada para base64 (para exibição em um chat)."""
        img_stream = BytesIO()
        figure.write_image(img_stream, format='png')  # Pode ser .png, .jpeg, etc.
        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
        return img_base64

# ====================
# Núcleo do Agente
# ====================
class MrAgent:
    def __init__(self, table_name: str = 'crm_metadata'):
        self.table_name = table_name
        self.athena = AthenaManager(table_name)
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        self.prompts = PromptManager()
        self.data_viz = DataVizExpert()  # Adicionando o especialista em visualização
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(TypedDict)
        
        workflow.add_node("extract_filters", self.extract_filters)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("format_response", self.format_response)
        workflow.add_node("generate_visualization", self.generate_visualization)  # Nova etapa
        
        workflow.set_entry_point("extract_filters")
        workflow.add_edge("extract_filters", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", "generate_visualization")  # Conectando
        workflow.add_edge("generate_visualization", END)
        
        return workflow.compile()
    
    # Nova função para gerar visualizações
    def generate_visualization(self, state: dict):
        try:
            df = state['data']
            # Exemplo de gráfico gerado (pode ser personalizado de acordo com a pergunta)
            chart_img = self.data_viz.create_bar_chart(df, x_col='category', y_col='total_sales', title="Total de Vendas por Categoria")
            return {'chart': chart_img}
        except Exception as e:
            return {'error': f"Erro na geração do gráfico: {str(e)}"}
    
    def extract_filters(self, state: dict):
        try:
            prompt = self.prompts.get_prompt(
                "date_extractor",
                current_date=datetime.now().strftime('%Y-%m-%d'),
                last_month=(datetime.now() - relativedelta(months=1)).strftime('%Y-%m')
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
    
    # # Consulta válida
    # print(agent.run("Qual o potencial médio por produto no canal VAI nos últimos 2 meses?"))
    
    # # Consulta inválida
    # print(agent.run("DELETE FROM tabela WHERE year = 2023"))