# Basics
import yaml
import re
import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence, List
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import awswrangler as wr

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableParallel
from langchain.agents import Tool

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

# Definição do estado do agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[str], operator.add]
    inter: pd.DataFrame
    memory: Annotated[Sequence[BaseMessage], operator.add]
    date_filter: List
    attempts_count: int
    agent: str
    metadados: str
    table_desc: str
    additional_filters: str
    question: str

class MrAgent():
    def __init__(self):
        # Reading the .yamls
        with open("metadata-v2.yaml", "r") as f:
            self.metadata = yaml.safe_load(f)

        # ############## PROMPT EXTRAÇÃO DE DATAS ##############
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", """Como analista de dados brasileiro especialista em python, sua função é extrair as informações relativas a data.
            
            Você está na data: {last_ref}
            
            Sempre forneça a data como um código pd.date_range() com o argumento freq 'ME' e no formato 'YYYY-mm-dd'.
            Caso exista apenas a informação mês, retorne start='0000-mm-01' e end='0000-mm-dd'.
            Caso exista apenas a informação ano, retorne todo o intervalo do ano.
            Caso não exista informação de data, retorne pd.date_range(start='0000-00-00', end='0000-00-00', freq='ME').
            Caso a pergunta contenha a expressão "mês a mês" ou "referência", retorne pd.date_range(start='3333-00-00', end='3333-00-00', freq='ME').
            Caso a pergunta contenha "último(s) mês(es)", retorne os últimos meses de acordo com a pergunta.
            
            Nunca forneça intervalos de data maiores que fevereiro de 2025."""),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{question}")
        ])

        # ############## PROMPT ENRIQUECIMENTO PERGUNTA ##############
        self.enrich_mr_camp_str = """Como engenheiro de prompt, sua função é reescrever e detalhar a pergunta de forma que um modelo de LLM consiga responder.
        
        Considere que você tem acesso a seguinte tabela para enriquecer a resposta:
        {table_description_mr}
        {column_context_mr}
        
        Pergunta do usuário: {question}
        
        Reescreva de forma sucinta a pergunta indicando quais filtros são necessários realizar para respondê-la.
        Atente-se à pergunta! Não infira nada além do que está nela.
        
        Caso a pergunta contenha algum conceito que não está nos metadados, redija a pergunta de forma a dizer que não consegue responder.
        
        Considere que a pergunta possui o seguinte filtro na coluna 'safra': {date_filter}"""
        
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.enrich_mr_camp_str),
            MessagesPlaceholder(variable_name="memory"),
            MessagesPlaceholder(variable_name="messages")
        ])

        # ############## PROMPT MÁQUINA DE RESULTADOS ##############
        self.mr_camp_prompt_str = """Como engenheiro de dados brasileiro, especializado em análise de dados bancários de engajamento e CRM usando Python, seu papel é responder exclusivamente a perguntas sobre a Máquina de Resultados.
        
        Você tem acesso ao dataframe 'df' com informações sobre:
        {table_description_mr}
        
        Baseando-se nas descrições das colunas disponíveis:
        {column_context_mr}
        
        Identifique quais colunas estão diretamente relacionadas com a pergunta. Desenvolva e execute uma sequência de comandos utilizando a ferramenta 'evaluate_pandas_chain' da seguinte maneira:
        
        <BEGIN> -> action1 -> action2 -> action3 -> <END>.
        
        Observações:
        - Use str.contains() para procurar valores em colunas string.
        - Valores string estão em CAPSLOCK.
        - Caso a pergunta contenha conceitos não presentes nos metadados, não infira resposta.
        - Retorne tabelas em markdown quando solicitado."""
        
        self.mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.mr_camp_prompt_str),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Initialize models and tools
        self.init_model()

    def init_model(self):
        # Initialize tools
        pdt = PandasTool()
        self.pdt = pdt
        tool_evaluate_pandas_chain = pdt.evaluate_pandas_chain

        dt = DocumentTool()
        self.dt = dt

        # Configure tools
        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Configure prompts with partials
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(
            table_description_mr=pdt.get_qstring_mr_camp(),
            column_context_mr=dt.get_col_context_mr_camp()
        )

        # Initialize models
        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)
        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain")

        # Date handling
        refs = pdt.get_refs()
        if not refs:
            refs = [int(datetime.now().strftime("%Y%m"))]
        max_ref = max(refs)
        last_ref = (datetime.strptime(str(max_ref), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref, datas_disponiveis=refs)

        # Build workflow
        self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("resposta", self.call_resposta)

        workflow.set_entry_point("query_generation")
        workflow.add_edge("query_generation", "date_extraction")
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
            {"more_info": "mr_camp_enrich_agent", "ok": END}
        )
        workflow.add_edge("sugest_pergunta", END)
        self.app = workflow.compile()

    def run(self, context, verbose=True):
        try:
            query = context['messages'][-1]["content"]
            memory = context['messages'][:-1]

            inputs = {
                "messages": [HumanMessage(content=query)],
                "actions": ["<BEGIN>"],
                "inter": pd.DataFrame(),
                "memory": memory,
                "date_filter": [],
                "attempts_count": 0,
                "agent": "",
                "metadados": "",
                "table_desc": "",
                "additional_filters": "",
                "question": query
            }

            for output in self.app.stream(inputs, {"recursion_limit": 100}):
                for key, value in output.items():
                    if verbose:
                        print(f"Step: {key} | Output: {value}")

            final_state = self.app.invoke(inputs)
            return final_state.get("messages", ["No response generated"])[-1].content, "", pd.DataFrame()

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return "Encontramos um problema processando sua pergunta. Tente novamente com outra abordagem.", "", pd.DataFrame()

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

    # FLAG DE MUDANÇA: Novo método de geração de query e obtenção do dataframe via Athena
    def call_query_generator(self, state):
        question = state["question"]
        metadata_str = json.dumps(self.metadata, indent=2)
        forbidden_operations = ", ".join(self.metadata["table_config"]["security"]["forbidden_operations"])
        maximum_rows = self.metadata["table_config"]["security"]["maximum_rows"]
        query_guidelines = "\n".join(self.metadata.get("query_guidelines", []))

        _ = self.query_generation_prompt.partial(
            forbidden_operations=forbidden_operations,
            maximum_rows=maximum_rows,
            metadata=metadata_str,
            query_guidelines=query_guidelines,
            question=question
        )
        response = self.model_query_generator.invoke({
            "forbidden_operations": forbidden_operations,
            "maximum_rows": maximum_rows,
            "metadata": metadata_str,
            "query_guidelines": query_guidelines,
            "question": question
        })
        generated_query = response.content.strip()
        print("Query Gerada para Homologação (antes do ajuste de data):")
        print(generated_query)
        
        # Se o LLM não incluiu nenhuma condição de data, injeta o filtro padrão
        query_lower = generated_query.lower()
        if "year" not in query_lower and "month" not in query_lower:
            # Obter a última referência disponível (supondo que get_refs() retorne valores no formato YYYYMM)
            max_ref = max(self.pdt.get_refs())
            default_year = str(max_ref)[:4]
            default_month = str(max_ref)[4:].zfill(2)
            date_filter = f"year = '{default_year}' AND month = '{default_month}'"
            # Se houver cláusula WHERE, adiciona com AND; senão, cria uma cláusula WHERE
            if "where" in query_lower:
                generated_query = generated_query.rstrip(";") + f" AND {date_filter};"
            else:
                generated_query = generated_query.rstrip(";") + f" WHERE {date_filter};"
            print("Filtro de data injetado automaticamente:")
            print(date_filter)
        
        print("Query Final para Homologação:")
        print(generated_query)
        
        df = self.run_query(generated_query)
        if 'safra' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['safra'] = df['year'].astype(str) + df['month'].astype(str).str.zfill(2)
            print("Coluna 'safra' criada a partir de 'year' e 'month'.")
        self.pdt.df = df
        return {"generated_query": generated_query, "df": df}

    def build_workflow(self):
        # FLAG DE MUDANÇA: Novo nó "query_generation" adicionado ao workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("resposta", self.call_resposta)
        workflow.set_entry_point("query_generation")
        workflow.add_edge("query_generation", "date_extraction")
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
        self.app = workflow.compile()

    # Método para execução das ferramentas
    def call_tool(self, state):
        messages = state['messages']
        last_message = messages[-1]
        output_dict = {}
        output_dict["messages"] = []
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
            output_dict["actions"].append(attempted_action)
            output_dict["inter"] = inter
            print("TOOL OUTPUT")
            print(output_dict)
            return output_dict

    def call_model_mr_camp_enrich(self, state):
        response = self.model_enrich_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_model_mr_camp(self, state):
        response = self.model_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}

    def call_sugest_pergunta(self, state):
        sugestao = self.sugest_model.invoke(state)
        return {"messages": [sugestao]}

    def call_resposta(self, state):
        resposta = self.resposta_model.invoke(state)
        print("RESPOSTA AQUIIIIIII -->", resposta)
        if not resposta.tool_calls:
            return {"messages": [resposta]}
        else:
            resposta = "Mais informações:"
            resposta = AIMessage(resposta)
            return {"messages": [resposta]}

    # FLAG DE MUDANÇA: Método para executar a query no Athena
    def run_query(self, query: str):
        inicio = datetime.now()
        df = wr.athena.read_sql_query(
            sql=query,  
            database='database_db_compartilhado_consumer_crmcoecampanhaspj',
            workgroup='analytics-workspace-v3',
            ctas_approach=False
        )
        if not hasattr(self, 'athenas_time'):
            self.athenas_time = []
        self.athenas_time.append(datetime.now() - inicio)
        print(f"TEMPO EXEC ATHENA: {datetime.now() - inicio}")
        return df
