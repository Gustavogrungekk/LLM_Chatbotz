# Basics
import json
import pandas as pd
import awswrangler as wr
import yaml
from datetime import datetime
from dateutil.relativedelta import relativedelta
import operator
from typing import TypedDict, Annotated, Sequence, List

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Langgraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Rag tools
from rag_tools.pandas_tools import PandasTool
from rag_tools.documents_tools import DocumentTool
from rag_tools.date_tool import date_tool, DateToolDesc
from rag_tools.more_info_tool import ask_more_info

# Self-defined functions
from util_functions import get_last_chains

# Define o AgentState para o fluxo
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[List], operator.add]
    inter: pd.DataFrame
    memory: str
    date_filter: List
    attempts_count: int
    agent: str
    metadados: str
    table_desc: str
    additional_filters: str

# ================================
# ATHENA TOOL - UTILIZANDO OS METADADOS DO YAML
# ================================
class AthenaTool():
    def __init__(self, metadata_path='table_metadata.yaml'):
        with open(metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
            
    def get_table_description(self):
        return f"""
Tabela: {self.metadata['table_config']['name']}
Partições obrigatórias: {[p['name'] for p in self.metadata['table_config']['partitions']]}
Colunas principais: {[c['name'] for c in self.metadata['columns']]}
"""
        
    def get_query_guidelines(self):
        return "\n".join(self.metadata['query_guidelines'])
    
    def get_latest_date(self):
        """
        Obtém a data mais recente a partir da tabela no Athena,
        consultando os valores máximos dos campos de partição 'year' e 'month'.
        """
        try:
            query_year = f"SELECT MAX(CAST(year AS INT)) as max_year FROM {self.metadata['table_config']['name']}"
            df_year = wr.athena.read_sql_query(
                sql=query_year, 
                database=self.metadata['table_config']['database'],
                workgroup=self.metadata['table_config']['workgroup'],
                ctas_approach=False
            )
            max_year = df_year.iloc[0]['max_year']
            query_month = f"SELECT MAX(CAST(month AS INT)) as max_month FROM {self.metadata['table_config']['name']} WHERE year = '{max_year}'"
            df_month = wr.athena.read_sql_query(
                sql=query_month, 
                database=self.metadata['table_config']['database'],
                workgroup=self.metadata['table_config']['workgroup'],
                ctas_approach=False
            )
            max_month = df_month.iloc[0]['max_month']
            return {'year': str(max_year), 'month': str(max_month).zfill(2)}
        except Exception as e:
            print("Erro ao obter a data mais recente:", e)
            # Em caso de erro, retorna um valor padrão (evite hardcode em produção)
            return {'year': '2025', 'month': '01'}

# ================================
# AGENTE PRINCIPAL: MrAgent
# ================================
class MrAgent():
    def __init__(self):
        # Inicializa o agente e as ferramentas
        self.athena_tool = AthenaTool()
        self.athenas_time = []  # Para armazenar os tempos de execução das queries no Athena
        self.init_model()
        self.build_workflow()
    
    # ========== PROMPT PARA GERAÇÃO DE QUERIES NO ATHENA ==========
    sql_gen_prompt_str = """
Como especialista em AWS Athena e análise de dados bancários, sua função é criar queries SQL eficientes usando estas diretrizes:

**Metadados da Tabela:**
{table_metadata}

**Diretrizes Obrigatórias:**
{query_guidelines}

**Instruções:**
1. Sempre incluir filtros de partição (year, month, canal) como strings.
2. NUNCA utilize SELECT *; especifique explicitamente as colunas necessárias: cnpj, produto, abastecido, potencial, visto, clique, atuado, disponivel, canal, metrica_engajamento.
3. Priorize consultas performáticas com LIMIT e filtros adequados.
4. Formate datas como strings: 'YYYY' para year, 'MM' para month.

Pergunta do usuário: {question}
Filtros aplicados: {filters}

Retorne a query SQL pronta para execução, com comentários e explicações, para fins de homologação.
    """
    
    # ========== PROMPT ORIGINAL PARA A MÁQUINA DE RESULTADOS ==========
    mr_camp_prompt_str = """
Como engenheiro de dados brasileiro, especializado em análise de dados bancários de engajamento e CRM (Customer Relationship Management) usando a linguagem de programação Python, seu papel é responder exclusivamente a perguntas sobre a Máquina de Resultados, 
um conjunto de dados utilizado para acompanhar o desempenho de campanhas e ações de CRM.
Você tem acesso ao dataframe 'df' com informações sobre:
{table_desc}
Baseando-se na descrição das colunas e nos metadados:
{metadados}
Responda à pergunta do usuário de forma clara, concisa e detalhada.
    """
    
    def init_model(self):
        # Configurar modelo para geração de queries SQL para Athena
        self.sql_gen_prompt = ChatPromptTemplate.from_template(self.sql_gen_prompt_str).partial(
            table_metadata=self.athena_tool.get_table_description(),
            query_guidelines=self.athena_tool.get_query_guidelines()
        )
        self.sql_gen_model = (self.sql_gen_prompt 
                              | ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
                              | StrOutputParser())
        
        # Configurar o modelo para a Máquina de Resultados (preservando prompt original)
        self.mr_camp_prompt = ChatPromptTemplate.from_template(self.mr_camp_prompt_str).partial(
            table_desc=self.athena_tool.get_table_description(),
            metadados="Consulte os metadados do YAML para detalhes completos sobre cada coluna."
        )
        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)
        
        # Inicialização das ferramentas originais
        pdt = PandasTool()
        self.pdt = pdt
        tool_evaluate_pandas_chain = pdt.evaluate_pandas_chain
        
        dt = DocumentTool()
        self.dt = dt
        
        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = tools  # Lista das ferramentas
        
        # ========== CONFIGURAÇÃO DO EXTRATOR DE DATAS ==========
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Extraia períodos no formato Athena:
- Datas como strings 'YYYY'/'MM'
- Se não houver data na pergunta, use a última data disponível conforme a consulta.
- Priorize os últimos 3 meses se não especificado.
            """)
        ])
        self.date_prompt = self.date_prompt.partial(latest_date=self.athena_tool.get_latest_date())
        self.date_extractor = (self.date_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
                               | StrOutputParser())
        
        # ========== CONFIGURAÇÃO DOS PROMPTS DE SUGESTÃO E RESPOSTA ==========
        self.suges_pergunta_prompt_desc = """
Você é um assistente de IA especializado em melhorar a clareza e a completude das perguntas dos usuários.
Sua tarefa é analisar a pergunta original e identificar se há informações faltantes ou ambíguas.
Considere que o dataframe possui as seguintes informações:
{table_desc}
Metadados: {metadados}
Se a pergunta estiver clara, confirme o entendimento. Caso contrário, peça mais informações.
        """
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.suges_pergunta_prompt_desc),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "(question)")
        ])
        
        self.resposta_prompt_desc = """
Você é um analista de dados especializado em dados bancários e engajamento do cliente.
Sua função é verificar se a resposta técnica contém todas as informações necessárias para responder à pergunta do usuário.
Use exclusivamente os dados do dataframe que contém:
{table_desc}
Metadados: {metadados}
Se as informações forem suficientes, formate a resposta em markdown, incluindo o período de datas.
Caso contrário, utilize a ferramenta 'ask_more_info' para solicitar mais dados.
        """
        self.resposta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.resposta_prompt_desc),
            MessagesPlaceholder(variable_name="memory"),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.model_enrich_mr_camp = self.suges_pergunta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1) \
            .bind_tools([ask_more_info], parallel_tool_calls=False)
    
    # ========== EXECUÇÃO DA QUERY GERADA (nó mr_camp_action) ==========
    def call_tool(self, state):
        # Gera a query SQL utilizando o modelo configurado e os filtros extraídos
        query = self.sql_gen_model.invoke({
            "question": state['question'],
            "filters": state['date_filter']
        })
        # Imprime a query com comentários e explicações para homologação
        print(f"\n=== QUERY GERADA PARA HOMOLOGAÇÃO ===\n{query}\n")
        # Executa a query no AWS Athena
        df = self.run_query(query)
        response = df.head().to_markdown()
        return {"messages": [ToolMessage(content=response, name="SQL_Generation")]}
    
    def run_query(self, query: str):
        inicio = datetime.now()
        df = wr.athena.read_sql_query(
            sql=query,
            database=self.athena_tool.metadata['table_config']['database'],
            workgroup=self.athena_tool.metadata['table_config']['workgroup'],
            ctas_approach=False
        )
        self.athenas_time.append(datetime.now() - inicio)
        print(f"TEMPO EXEC ATHENA: {datetime.now() - inicio}")
        return df
    
    # ========== EXTRAÇÃO DE DATAS ==========
    def get_date_filter(self, question: str) -> dict:
        return self.date_extractor.invoke({"question": question})
    
    # ========== FLUXO PRINCIPAL ==========
    def run(self, context, verbose: bool = True):
        print("Streamlit session state:")
        print(context)
        print(type(context))
        
        query = context['messages'][-1]["content"]
        memory = context['messages'][:-1]
        
        # Obter filtros de data a partir da pergunta
        date_filter = self.get_date_filter(query)
        
        inputs = {"messages": [HumanMessage(content=query)],
                  "actions": ["<BEGIN>"],
                  "question": query,
                  "memory": memory,
                  "date_filter": date_filter,
                  "attempts_count": 0}
        
        try:
            current_action = []
            inter_list = []
            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):
                print(output)
                for key, value in output.items():
                    if key.endswith("agent") and verbose:
                        print(f"Agent {key} working...")
                    elif key.endswith("_action") and verbose:
                        if value["messages"][0].name == "view_pandas_dataframes":
                            print("Current action: viewing dataframes")
                        else:
                            if "actions" in value.keys():
                                print(f"Current action: {value['actions']}")
                                print(f"Current output: {value['inter']}")
                    elif key == "date_extraction" and verbose:
                        print(value["date_filter"])
                        print("Date filter for the current question:")
                    elif key == "sugest_pergunta" and verbose:
                        print("Prompt engineering response:")
                        print(value["messages"])
                    elif key == "add_count" and verbose:
                        print("Adding attempt count:")
                        print(value["attempts_count"])
                    elif key == "resposta" and verbose:
                        print("Verificando resposta:")
                        print(value["messages"])
                    elif verbose:
                        print("Finishing up...")
                        print(f"Final output: {value.get('inter', '')}")
                        print(f"Final action chain: {' -> '.join(value.get('actions', []))} -> <END>")
                    
                    if "actions" in value.keys():
                        current_action.append("->".join(value["actions"]) if isinstance(value["actions"], list) else value["actions"])
                    messages = value.get('messages', None)
                    if 'inter' in value and value['inter'] is not None:
                        inter_list.append(value['inter'])
                    print("---")
                final_action = current_action[-1] if current_action else ""
                agent_response = messages[-1].content if messages else ""
                final_table = inter_list[-1] if inter_list else []
                final_message = agent_response.replace('<END>', '').replace('<BEGIN>', '')
        except Exception as e:
            print("Houve um erro no processo:")
            print(e)
            final_message = "Encontramos um problema processando sua pergunta. Tente novamente, com outra abordagem."
            final_action = ''
            final_table = ''
        return final_message, final_action, final_table
    
    def should_ask(self, state):
        print(f"QUANTIDADE DE TENTATIVAS: {state['attempts_count']}")
        last_message = state['messages'][-1]
        if (("An exception occured" in last_message['content']) and (state['attempts_count'] >= 2)) or (state['attempts_count'] >= 4):
            return "ask"
        else:
            print(f"Última mensagem: {last_message['content']}")
            return "not_ask"
    
    def add_count(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if not last_message.get('tool_calls'):
            return {"attempts_count": state['attempts_count']}
        else:
            if last_message['additional_kwargs']['tool_calls'][0]['function']['name'] != 'view_pandas_dataframes':
                qtd_passos = state['attempts_count'] + 1
                return {"attempts_count": qtd_passos}
        return {"attempts_count": state['attempts_count']}
    
    def need_info(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informações:"):
            return "more_info"
        return "ok"
    
    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("resposta", self.call_resposta)
        workflow.set_entry_point("date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "mr_camp_agent")
        workflow.add_edge("mr_camp_agent", "add_count")
        workflow.add_edge("add_count", "mr_camp_action")
        workflow.add_conditional_edges(
            "mr_camp_action",
            self.should_ask,
            {"ask": "sugest_pergunta", "not_ask": "resposta"}
        )
        workflow.add_conditional_edges(
            "resposta",
            self.need_info,
            {"more_info": "mr_camp_enrich_agent", "ok": "END"}
        )
        workflow.add_edge("sugest_pergunta", "END")
        self.app = workflow.build()  # Usando build() em vez de compile()
    
    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}
    
    def call_model_mr_camp_enrich(self, state):
        response = self.model_enrich_mr_camp.invoke(state)
        return {"messages": [response]}
    
    def call_model_mr_camp(self, state):
        response = self.model_mr_camp.invoke(state)
        return {"messages": [response]}
    
    def call_sugest_pergunta(self, state):
        sugestao = self.sugest_model.invoke(state)
        return {"messages": [sugestao]}
    
    def call_resposta(self, state):
        resposta = self.resposta_model.invoke(state)
        print("RESPOSTA AQUI -->", resposta)
        if not resposta.tool_calls:
            return {"messages": [resposta]}
        else:
            resposta = "Mais informações:"
            resposta = AIMessage(resposta)
            return {"messages": [resposta]}

# ================================
# FUNÇÃO EXTERNA RENOMEADA PARA EVITAR CONFLITO
# ================================
def execute_tool_chain(self, state):
    messages = state['messages']
    last_message = messages[-1]
    output_dict = {"messages": []}
    for idx, tool_call in enumerate(last_message.additional_kwargs['tool_calls']):
        tool_input = last_message.additional_kwargs['tool_calls'][idx]['function']['arguments']
        tool_input_dict = json.loads(tool_input)
        if last_message.additional_kwargs['tool_calls'][idx]['function']['name'] == 'evaluate_pandas_chain':
            tool_input_dict['inter'] = state['inter']
            tool_input_dict['date_filter'] = state['date_filter']
            tool_input_dict['agent'] = state['agent']
        action = ToolInvocation(
            tool=last_message.additional_kwargs['tool_calls'][idx]['function']['name'],
            tool_input=tool_input_dict
        )
    response, attempted_action, inter = self.tool_executor.invoke(action)
    if "An exception occurred:" in str(response):
        error_info = f"""
You have previously performed the actions:
{state['actions']}

Current action:
{attempted_action}

Result.head(10):
{response}

You must correct your approach and continue until you can answer the question:
{state['question']}

Continue the chain with the following format: action_i -> action_i+1... -> <END>
        """
        print(error_info)
        function_message = ToolMessage(
            content=str(error_info),
            name=action.tool,
            tool_call_id=tool_call["id"]
        )
        output_dict["messages"].append(function_message)
    else:
        success_info = f"""
You have previously performed the actions:
{state['actions']}

Current action:
{attempted_action}

Result.head(50):
{response}

You must continue until you can answer the question:
{state['question']}

Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
        """
        print(success_info)
        function_message = ToolMessage(
            content=str(success_info),
            name=action.tool,
            tool_call_id=tool_call["id"]
        )
        output_dict["messages"].append(function_message)
        output_dict["actions"] = [attempted_action]
        output_dict["inter"] = inter
        print("TOOL OUTPUT")
        print(output_dict)
        return output_dict

# ================================
# Exemplo de Uso
# ================================
if __name__ == "__main__":
    agent = MrAgent()
    pergunta = "QUAL O POTENCIAL DO CANAL VAI E MCE NOS ÚLTIMOS 2 MESES?"
    resultado = agent.run({"messages": [{"content": pergunta}]})
    print("Resultado Final:")
    print(resultado)
