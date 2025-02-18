import json
import pandas as pd
import awswrangler as wr
import yaml
from typing import TypedDict, Annotated, Sequence, List
import operator
from pathlib import Path

# Self-defined functions
from util_functions import get_last_chains

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langgraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

# Date and time handling
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load table metadata
with open('table_metadata.yaml') as f:
    TABLE_METADATA = yaml.safe_load(f)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[list], operator.add]
    inter: pd.DataFrame 
    question: str 
    memory: str 
    date_filter: list 
    attempts_count: int 
    agent: str 
    metadados: str 
    table_desc: str 
    additional_filters: str 
    query: str 

class AthenaQueryTool:
    def __init__(self):
        self.metadata = TABLE_METADATA
        self._last_ref = None
        
    def get_query_guidelines(self):
        return "\n".join(self.metadata['query_guidelines'])
    
    def get_column_context(self):
        return "\n".join([f"{col['name']} ({col['type']}): {col['description']}" 
                         for col in self.metadata['columns']])
    
    def get_partition_filters(self, date_filter):
        if not date_filter or date_filter[0] == '0000-00-00':
            return ""
            
        years_months = set()
        for date_str in date_filter:
            if pd.isnull(pd.to_datetime(date_str, errors='coerce')):
                continue
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            years_months.add((str(dt.year), f"{dt.month:02d}"))
        
        partition_filters = []
        for year, month in years_months:
            partition_filters.append(f"(year = '{year}' AND month = '{month}')")
        
        return " AND ".join(partition_filters) if partition_filters else ""

    def generate_sql_query(self, question: str, date_filter: list, additional_filters: str = "") -> str:
        base_query = f"SELECT * FROM {self.metadata['table_config']['name']} "
        
        # Add WHERE clause with partitions
        where_clauses = []
        partition_filter = self.get_partition_filters(date_filter)
        if partition_filter:
            where_clauses.append(partition_filter)
        
        # Add additional filters
        if additional_filters:
            where_clauses.append(additional_filters)
        
        # Add security filters
        where_clauses.append(" AND ".join([
            f"{col['name']} NOT IN ({','.join(map(repr, col['ignore_values']))}"
            for col in self.metadata['columns'] if col['ignore_values']
        ]))
        
        if where_clauses:
            base_query += "WHERE " + " AND ".join(where_clauses)
        
        # Add security limits
        base_query += f"\nLIMIT {self.metadata['table_config']['security']['maximum_rows']}"
        
        return base_query

class MrAgent:
    def __init__(self):
        self.athena_tool = AthenaQueryTool()
        self.init_prompts()
        self.init_models()
        self.build_workflow()

    def init_prompts(self):
        # Date prompt remains similar but adjusted for Athena partitions
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", """Como analista de dados brasileiro especialista em AWS Athena, extraia informações de data para partições.
             Sempre retorne datas no formato 'YYYY-MM-DD'. Use filtros de partição year/month."""),
            MessagesPlaceholder(variable_name="memory"),
            ("user", '{question}')
        ])

        # Updated prompt for SQL generation
        self.mr_camp_prompt_str = f"""
        Como engenheiro de dados especializado em AWS Athena, gere queries SQL seguindo estas regras:
        {self.athena_tool.get_query_guidelines()}
        
        Colunas disponíveis:
        {self.athena_tool.get_column_context()}
        
        Diretrizes:
        - Use sempre filtros de partição year/month
        - Formate valores de data como strings
        - Use COUNT(DISTINCT CASE WHEN) para métricas binárias
        - Limite resultados a {TABLE_METADATA['table_config']['security']['maximum_rows']} linhas
        
        Exemplos válidos:
        {chr(10).join([ex['sql'] for ex in TABLE_METADATA['query_examples']]}
        """

        self.mr_camp_output = ChatPromptTemplate.from_messages([
            ("system", self.mr_camp_prompt_str),
            MessagesPlaceholder(variable_name="messages", n_messages=1)
        ])

    def init_models(self):
        self.tools = [convert_to_openai_tool(self.athena_tool.generate_sql_query)]
        self.tool_executor = ToolExecutor([self.athena_tool.generate_sql_query])
        
        self.model_mr_camp = (
            self.mr_camp_output 
            | ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
            .bind_tools(self.tools, tool_choice="generate_sql_query")
        )

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.set_entry_point("generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", END)
        self.app = workflow.compile()

    def generate_query(self, state):
        response = self.model_mr_camp.invoke(state)
        return {"messages": [response], "query": response.tool_calls[0]['args']['question']}

    def execute_query(self, state):
        try:
            query = state['query']
            df = wr.athena.read_sql_query(
                sql=query,
                database=TABLE_METADATA['table_config']['database'],
                workgroup=TABLE_METADATA['table_config']['workgroup'],
                ctas_approach=True
            )
            return {
                "messages": [AIMessage(content=f"Resultado da query:\n{df.head().to_markdown()}")],
                "inter": df
            }
        except Exception as e:
            error_msg = f"Erro na query: {str(e)}"
            return {"messages": [AIMessage(content=error_msg)]}

    def run(self, context):
        inputs = {
            "messages": [HumanMessage(content=context['messages'][-1]["content"])],
            "question": context['messages'][-1]["content"],
            "memory": context['messages'][:-1],
            "attempts_count": 0
        }
        
        result = self.app.invoke(inputs)
        return result['messages'][-1].content, result.get('inter', pd.DataFrame())

# Athena query execution function (separate for reuse)
def run_query(query: str):
    return wr.athena.read_sql_query(
        sql=query,
        database=TABLE_METADATA['table_config']['database'],
        workgroup=TABLE_METADATA['table_config']['workgroup'],
        ctas_approach=True
    )
