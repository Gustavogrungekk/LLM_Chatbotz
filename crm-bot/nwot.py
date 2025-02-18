from typing import TypedDict, Sequence, Annotated
from datetime import datetime
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.utils.function_calling import convert_to_openai_tool
import awswrangler as wr
from dateutil.relativedelta import relativedelta

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
    query: str  # Novo campo para armazenar a query Athena

class AthenaTool:
    def __init__(self):
        self.database = 'database_w1'
        self.workgroup = 'tbl_coeres_painel_anomes_v1_gold'
        self.metadata = self.get_metadata()

    def run_query(self, query: str) -> pd.DataFrame:
        try:
            if not self.validate_query(query):
                raise ValueError("Query contém operações perigosas")
            
            inicio = datetime.now()
            df = wr.athena.read_sql_query(
                sql=query,
                database=self.database,
                workgroup=self.workgroup,
                ctas_approach=True
            )
            print(f"TEMPO EXEC ATHENA: {datetime.now() - inicio}")
            return df
        except Exception as e:
            print(f"Erro no Athena: {str(e)}")
            return pd.DataFrame()

    def get_metadata(self) -> str:
        return """
        [METADADOS DA TABELA]
        Colunas disponíveis:
        - id_campanha: Identificador único (STRING)
        - data_execucao: Data de execução (DATE)
        - segmento: Segmento de clientes (STRING)
        - taxa_resposta: Taxa de resposta (%)
        - canal: Canal de comunicação (EMAIL/SMS/PUSH)
        - status: Status da campanha
        - safra: Período de referência (YYYYMM)
        """

    def validate_query(self, query: str) -> bool:
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
        return not any(cmd in query.upper() for cmd in forbidden)


class MrAgent:
    def __init__(self):
        self.athena_tool = AthenaTool()

        # Create a wrapper function for the AthenaTool's run_query method
        self.athena_tool_func = self.create_tool_function(self.athena_tool.run_query)

        # Configuração inicial dos prompts
        self._setup_prompts()
        self._setup_models()
        self._setup_tools()
        self.build_workflow()

    def create_tool_function(self, original_func):
        """
        This wraps the AthenaTool's `run_query` to match the expected signature.
        """
        def tool_function(inputs: dict) -> dict:
            query = inputs.get("query")
            if query:
                result = original_func(query)
                return {"output": result}  # Return the result in the expected format
            return {"output": pd.DataFrame()}  # Fallback empty dataframe if no query is found
        
        return tool_function

    def _setup_prompts(self):
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", """Como especialista em datas, extraia intervalos relevantes seguindo:
             - Formato Athena DATE 'YYYY-MM-DD'
             - Use BETWEEN para intervalos
             - Última data disponível: {last_ref}
             Exemplos: 
             * Último mês → WHERE data_execucao BETWEEN DATE '2024-05-01' AND DATE '2024-05-31'
             * Semana passada → WHERE data_execucao BETWEEN DATE '2024-06-10' AND DATE '2024-06-16'""")
        ])

        self.mr_camp_prompt_str = ChatPromptTemplate.from_messages([
            ("system", f"""Você é um gerador de queries SQL para Athena. Regras:
            1. Use sempre a tabela 'tbl_coeres_painel_anomes_v1_gold'
            2. Colunas textuais em MAIÚSCULAS com UPPER()
            3. Filtre datas usando: {self.athena_tool.metadata}
            4. Formato de data: DATE 'YYYY-MM-DD'
            5. Limite resultados com LIMIT 1000
            6. Inclua safra quando relevante
            
            Exemplos:
            - "Quantos clientes no RJ?" → SELECT COUNT(*) FROM tabela WHERE UPPER(estado) = 'RJ'
            - "Média de taxas por segmento" → SELECT segmento, AVG(taxa_resposta) FROM tabela GROUP BY segmento""")
        ])

        self.resposta_prompt_desc = ChatPromptTemplate.from_messages([
            ("system", """Valide e formate respostas considerando:
             - Dados do Athena
             - Formato Markdown
             - Inclua período das datas
             - Destaque métricas principais""")
        ])

    def _setup_models(self):
        self.model_mr_camp = self.mr_camp_prompt_str | ChatOpenAI(model="gpt-4", temperature=0)
        self.resposta_model = self.resposta_prompt_desc | ChatOpenAI(model="gpt-4", temperature=0)

    def _setup_tools(self):
        # Use the wrapped AthenaTool function
        self.tools = [convert_to_openai_tool(self.athena_tool_func)]
        self.tool_node = ToolNode(self.tools)

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("date_extraction", self.extract_dates)
        workflow.add_node("generate_query", self.generate_athena_query)
        workflow.add_node("execute_query", self.execute_athena_query)
        workflow.add_node("format_response", self.format_response)

        workflow.set_entry_point("date_extraction")
        
        workflow.add_edge("date_extraction", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        workflow.add_edge("format_response", END)

        self.app = workflow.compile()

    def extract_dates(self, state: AgentState):
        question = state['question']
        last_ref = datetime.now().strftime('%Y-%m-%d')
        
        date_chain = self.date_prompt.partial(last_ref=last_ref) | ChatOpenAI(model="gpt-4")
        date_filter = date_chain.invoke({"question": question}).content
        
        return {"date_filter": date_filter}

    def generate_athena_query(self, state: AgentState):
        question = state['question']
        date_filter = state.get('date_filter', '')
        
        full_prompt = f"""
        Pergunta: {question}
        Filtro de datas: {date_filter}
        Metadados: {self.athena_tool.metadata}
        
        Gere uma query SQL válida para o Athena seguindo:
        - Use apenas colunas existentes
        - Formate datas corretamente
        - Adicione LIMIT 1000
        """
        
        response = self.model_mr_camp.invoke(full_prompt)
        return {"query": response.content}

    def execute_athena_query(self, state: AgentState):
        query = state['query']
        
        try:
            df = self.athena_tool.run_query(query)
            return {"inter": df, "messages": [f"Query executada com sucesso. {len(df)} linhas retornadas."]}

        except Exception as e:
            error_msg = f"Erro na query: {str(e)}"
            return {"messages": [error_msg], "inter": pd.DataFrame()}

    def format_response(self, state: AgentState):
        df = state.get('inter', pd.DataFrame())
        question = state['question']
        
        if not df.empty:
            markdown_table = df.head(10).to_markdown()
            response = f"""
            **Resposta para**: {question}
            **Período**: {state.get('date_filter', 'N/A')}
            **Dados**:
            {markdown_table}
            """
        else:
            response = "Não foi possível obter dados para esta consulta."
        
        return {"messages": [AIMessage(content=response)]}

    def run(self, question: str):
        inputs = {
            "question": question,
            "messages": [],
            "actions": [],
            "attempts_count": 0
        }
        
        final_response = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                if key == "format_response":
                    final_response = value['messages'][0].content
        
        return final_response

# Exemplo de uso
if __name__ == "__main__":
    agent = MrAgent()
    resposta = agent.run("Qual a taxa média de resposta por segmento no último mês?")
    print(resposta)
