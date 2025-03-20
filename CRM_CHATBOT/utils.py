# Main code
import boto3 
import json 
import operator 
import random 
import os
import re 
import sys 
import time 
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import Annotated, List, Sequence, TypedDict
import awswrangler as wr
import openai
import pandas as pd 
import yaml 

from langchain.agents import Tool 
from langchain.chains import LLMChain 
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage, HumanMessage, ToolMessage
from langchain_core.output_parses.openai_tools import JsonOutputKeyToolsParser 
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel 
from langchain_core.utils.function_calling import converto_to_openai_function, convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Local modules 
from rag_tools.date_tool import DateToolDesc, date_tool 
from rag_tools.document_tools import DocumentTool 
from rag_tools.more_info_tool import ask_more_info 
from rag_tools.pandas_tools import run_query, PandasTool 
from util_functions import get_last_chains 
from rag_tools.prompts import (
    DATE_PROMPT_DESC,
    ENRICH_MR_CAMP_DSC,
    MR_CAMP_PROMPT_STR_DSC,
    SUGESTAO_PERGUNTA_PROMPT_DSC,
    RESP_PROMPT_DSC,
    QUERY_GENERATION_PROMPT_DSC
)


os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# Definicao do Estado do agente 
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[List], operator.add]
    inter: pd.Dataframe 
    memory: Annotated[Sequence[BaseModel], operator.add]
    date_filter: List 
    attempts_count: int
    agent: str 
    metadados: str 
    table_dsc: str
    additional_filters: str 
    generated_query: str
    question: str 

# Definicao do Agente
class MrAgent():
    def __init__(self):

        # Metadata Extraction 
        with open('metadata.yaml', 'r', encoding='utf-8') as file:
            self.metadata = yaml.safe_load(file)

        # LLM configuration
        with open('llm_config.yaml', 'r', encoding='utf-8') as file:
            self.llm_config = yaml.safe_load(file)

        # Initialize LLM with configuration 
        self.llm - ChatOpenAI(
            model = self.llm_config['model'],
            temperature = self.llm_config['temperature'],
            seed = self.llm_config['seed'],
        )

        # Prompt Date Extraction 
        self.date_prompt = ChatPromptTemplate(
            [
                ("system", DATE_PROMPT_DESC),
                MessagesPlaceholder(variable_name='memory'),
                ("user", "{question}")
            ]
        )

        # Prompt for Enrichment
        self.enrichment_prompt = ChatPromptTemplate(
            [
                ("system", ENRICH_MR_CAMP_DSC),
                MessagesPlaceholder(variable_name='memory'),
                MessagesPlaceholder(variable_name='messages')
            ]
        )

        # Prompt for MR Campaign
        self.mr_camp_prompt = ChatPromptTemplate(
            [
                ("system", MR_CAMP_PROMPT_STR_DSC),
                MessagesPlaceholder(variable_name='memory')
            ]
        )

        # Prompt for Suggestion
        self.sugestion_prompt = ChatPromptTemplate(
            [
                ("system", SUGESTAO_PERGUNTA_PROMPT_DSC),
                MessagesPlaceholder(variable_name='memory'),
                ("user", "{question}") 
            ]
        )

        # Prompt for Response
        self.response_prompt = ChatPromptTemplate(
            [
                ("system", RESP_PROMPT_DSC),
                MessagesPlaceholder(variable_name='memory'),
                MessagesPlaceholder(variable_name='messages')
            ]
        )

        # Prompt for Query Generation
        self.query_generation_prompt = ChatPromptTemplate(
            [
                ("system", QUERY_GENERATION_PROMPT_DSC),
                ("user", "{question}")
                # MessagesPlaceholder(variable_name='memory')
            ]
        )

    def init_model(self):

        # Inicializa os modelos e ferramentas 
        pdt = PandasTool()
        dt = DocumentTool()
        tool_run_query = run_query

        # Configura as ferramentas que serao usadas 
        tools = [tool_run_query]
        self.tools_executor = ToolExecutor(tools)

        # converte as ferramentas para o formato do OpenAI
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # agente enriquecedor da maquina de resultados campanha 
        self.enrichment_prompt = self.enrichment_prompt.partial(table_description=pdt.get_qstring_mr_camp())

        self.enrichment_prompt = self.enrichment_prompt | ChatOpenAI(
            model = self.llm_config['model'],
            temperature = self.llm_config['temperature'],
            seed = self.llm_config['seed'],
        )

        # agente de maquina de resultados campanha
        self.mr_camp_prompt = self.mr_camp_prompt.partial(table_description=pdt.get_qstring_mr_camp())

        self.mr_camp_prompt = self.mr_camp_prompt | ChatOpenAI(
            model = self.llm_config['model'],
            temperature = self.llm_config['temperature'],
            seed = self.llm_config['seed'],
        )

        # Define date tool 
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), "%Y%m") + relativedelta(months=-1)).strftime("%Y/%m/%d")
        dates = pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = ChatOpenAI(
            model = self.llm_config['model'],
            temperature = self.llm_config['temperature'],
            seed = self.llm_config['seed'],
        ).bind_tools([DateToolDesc], tool_choice='DateToolsDesc')

        partial_mode = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolsDesc') | (lambda x: x[0]['pandas_str'])
        self.date_extractor = RunnableParallel(pandas_str=partial_mode, refs_list=lambda x: pdt.get_refs()) | date_tool

        # inclusao do verificador de respostas 
        self.response_prompt = self.response_prompt.partial(table_description=pdt.get_qstring_mr_camp())
        self.resposse_prompt = self.response_prompt.partial(metadados=dt.get_col_cotext_mr_camp())
        self.resposta_model = self.resposta_prompt | ChatOpenAI(
            model = self.llm_config['model'],
            temperature = self.llm_config['temperature'],
            seed = self.llm_config['seed'],
        ).bind_tools([ask_more_info], tool_choice='run_query')

        # construção do workflow 
        self.build_workflow()

    def run(self, context, verbose: bool=True):
        print('streamlit session state:')
        print(context)
        print(type(context))

        last_message = context['last_message'][-1]
        query = last_message['context']
        memory = context['memory'][-1]

        inputs = {
            "messages": [HumanMessage(content=query)],
            "actions": ["<BEGIN>"],
            "question": query,
            "memory": memory,
            "attempt_count": 0
        }

        context['question'] = query

        try:
            current_action = []
            inter_list = []

            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):

                for idx, (key, value) in enumerate(output.items()):
                    print(f"DEBUG - Key: {key}, Value: {value}")

                    if "question" not in value:
                        value["question"] = query  # Ensure question is present in state
                        print(f"ERROR - question MISSING in state at step: {key} -> {value}")

                    if key.endswith("agent") and verbose:
                        print(f"Agent {key} working...")

                    elif key.endswith("_action") and verbose:
                        print(f"Current action: {value.get('actions', [])}")
                        print(f"Current output value: {value.get('inter', '')}")

                    elif key == "date_extraction" and verbose:
                        if not value.get("date_filter", None):
                            print("Date filter for the current question not found")

                    elif key == "sugest_pergunta" and verbose:
                        print("Prompt engineering response:")
                        print(value.get("messages", []))

                    elif key == "add_count" and verbose:
                        print("Adding attempt count:")
                        print(value.get("attempts_count", 0))

                    elif key == "resposta" and verbose:
                        print("Verificando resposta:")
                        print(value.get("messages", []))

                    elif verbose:
                        print("Finishing up...")
                        print(f"Final output: {value.get('inter', '')}")
                        print(f"Final action chain: {' -> '.join(map(str, value.get('actions', [])))} -> <END>")

                    # Ensure actions and messages are stored correctly
                    if "actions" in value.keys():
                        current_action.append(" -> ".join(value["actions"][:-1]).replace("<BEGIN> -> ", "").replace("import pandas as pd;", ""))

                    if "inter" in value:
                        inter = value.get("inter", None)
                        if inter is not None:
                            inter_list.append(inter)

                final_action = current_action[-1] if current_action else ""
                final_table = inter_list[-1] if inter_list else ""
                final_message = value.get("messages", [])[-1].context if "messages" in value else ""

        except Exception as e:
            print(f"ERROR - {e}")
            final_action = ""
            final_table = ""
            final_message = ""

        return final_message, final_action, final_table
    
    def should_ask(self, state):
        print(f"QUANTIDADE DE TENTATIVAS: {state['attempts_count']}")
        last_message = state["messages"][-1]
        if ("An exception occurred" in last_message.content and (state["attempts_count"] >= 2)) or (state["attempts_count"] >= 4):
            return "ask"
        else:
            print(f"Última mensagem: {last_message.content}")
            return "not_ask"

    def add_count(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls"):
            return {"attempts_count": state["attempts_count"]}
        else:
            if last_message.additional_kwargs["tool_calls"][0]["function"]["name"] != "view_pandas_dataframes":
                qtd_passos = state["attempts_count"] + 1
                return {"attempts_count": qtd_passos}
        return {"attempts_count": state["attempts_count"]}

    def need_info(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informações:"):
            return "none_info"
        return "ok"
        
    def initialize_context_classifier(self):
        """Initialize the context classifier for out-of-scope detection"""
        # Define prompt for context classification
        self.context_classification_prompt = ChatPromptTemplate([
            ("system", """Você é um classificador especializado que determina se uma pergunta está dentro do escopo de análise de dados de campanhas de marketing.
            
            Escopo válido:
            - Perguntas sobre métricas de campanhas
            - Análise de desempenho de campanhas 
            - Comparações entre diferentes períodos
            - Tendências de resultados
            - Perguntas introdutórias ou de saudação (como "olá", "oi", "quem é você", etc.)
            
            Escopo inválido:
            - Perguntas pessoais complexas
            - Solicitações não relacionadas a dados
            - Instruções para executar ações externas
            - Conteúdo ofensivo ou prejudicial
            
            Retorne apenas "dentro_do_escopo" ou "fora_do_escopo" como resposta."""), 
            ("user", "{question}")
        ])
        
        # Create the classifier model
        self.context_classifier = self.context_classification_prompt | ChatOpenAI(
            model=self.llm_config['model'],
            temperature=0.1,
            seed=self.llm_config['seed'],
        )
    
    def is_out_of_scope(self, question):
        """Check if a question is out of scope"""
        try:
            response = self.context_classifier.invoke({"question": question})
            is_out = "fora_do_escopo" in response.content.lower()
            is_conversational = "dentro_do_escopo" in response.content.lower() and any(
                keyword in response.content.lower() for keyword in ["saudação", "introdução"]
            )
            print(f"Context classification: {response.content} (Out of scope: {is_out}, Conversational: {is_conversational})")
            return "conversational" if is_conversational else is_out
        except Exception as e:
            print(f"Error in scope detection: {e}")
            return False
    
    def call_date_extractor(self, state):
        # Check if question is out of scope
        scope_check = self.is_out_of_scope(state["question"])
        
        # Handle conversational questions
        if scope_check == "conversational":
            intro_response = self.get_conversational_response(state["question"])
            return {"messages": [intro_response], "date_filter": []}
        
        # Handle other out of scope questions
        elif scope_check:
            out_of_scope_message = AIMessage(content="""Desculpe, mas esta pergunta parece estar fora do escopo da análise de dados de campanhas. 
            Posso responder perguntas relacionadas a métricas, desempenho, comparações ou tendências de campanhas de marketing.""")
            return {"messages": [out_of_scope_message], "date_filter": []}
            
        # Continue with normal date extraction if in scope
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}
    
    def get_conversational_response(self, question):
        """Generate appropriate response for conversational questions"""
        intro_prompt = ChatPromptTemplate([
            ("system", """Você é um assistente de análise de dados de campanhas de marketing chamado MR Bot.
            Responda de forma amigável e profissional a perguntas introdutórias ou saudações.
            Mantenha a resposta breve e informe que você pode ajudar com análises de campanhas de marketing."""), 
            ("user", "{question}")
        ])
        
        intro_llm = ChatOpenAI(
            model=self.llm_config['model'],
            temperature=0.7,
            seed=self.llm_config['seed'],
        )
        
        intro_chain = intro_prompt | intro_llm
        response = intro_chain.invoke({"question": question})
        return AIMessage(content=response.content)
    
    def call_query_generator(self, state):
        print("PRINTANDO STATE", state["question"])
        question = state["messages"][-1].content
        
        # Check again if question is out of scope (in case date extraction was bypassed)
        scope_check = self.is_out_of_scope(question)
        
        # Handle conversational questions
        if scope_check == "conversational":
            intro_response = self.get_conversational_response(question)
            return {"messages": [intro_response]}
        
        # Handle other out of scope questions
        elif scope_check:
            out_of_scope_message = AIMessage(content="""Desculpe, mas esta pergunta parece estar fora do escopo da análise de dados de campanhas. 
            Posso responder perguntas relacionadas a métricas, desempenho, comparações ou tendências de campanhas de marketing.""")
            return {"messages": [out_of_scope_message]}
        
        # Get metadata
        metadata_str = json.dumps(self.metadata, indent=2)
        forbidden_operations = ", ".join(self.metadata["table_config"]["security"]["forbidden_operations"])
        maximum_rows = self.metadata["table_config"]["security"]["maximum_rows"]
        query_guidelines = "\n".join(self.metadata.get("query_guidelines", []))

        # Select relevant examples for this question
        relevant_examples = self.select_relevant_examples(question)
        
        # Format examples for the query generator
        query_examples = "\n".join(
            [f"[{i+1}] {example['description']}:\n{example['sql']}" for i, example in enumerate(relevant_examples)]
        )

        response = self.model_query_generator.invoke({
            "forbidden_operations": forbidden_operations,
            "maximum_rows": maximum_rows,
            "metadata": metadata_str,
            "query_guidelines": query_guidelines,
            "question": question,
            "query_examples": query_examples,
            "messages": state["messages"]
        })

        return {"messages": [response]}

    def build_workflow(self):
        # Initialize the context classifier
        self.initialize_context_classifier()
        
        # Create the state graph
        workflow = StateGraph(AgentState)

        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_emrich_agent", self.call_model_mr_camp_emrich)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("add_count", self.add_count)
        # workflow.add_edge("query_agent", "add_count")
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("resposta", self.call_resposta)

        # Define END node
        workflow.add_node("END", lambda state: state)

        workflow.set_entry_point("date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_emrich_agent")
        workflow.add_edge("mr_camp_emrich_agent", "query_generation")
        workflow.add_edge("query_generation", "mr_camp_action")
        workflow.add_edge("mr_camp_action", "add_count")
        workflow.add_edge("add_count", "resposta")
        workflow.add_edge("resposta", "END")

        workflow.add_conditional_edges(
            "add_count",
            self.should_ask,
            {
                "ask": "sugest_pergunta",
                "not_ask": "resposta"
            }
        )

        workflow.add_conditional_edges(
            "resposta",
            self.need_info,
            {
                "more_info": "mr_camp_emrich_agent",
                "ok": "END"
            }
        )

        self.app = workflow.compile()

    # Metodo para execucao de ferramentas 
    def call_tool(self, state):
        messages = state['messages']
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]

        output_dict = {}

        output_dict["messages"] = []

        for idx, tool_call in enumerate(last_message.additional_kwargs['tool_calls']):
            # tool_call = last_message.additional_kwargs['tool_calls'][0]
            tool_input = last_message.additional_kwargs['tool_calls'][idx]['function']['arguments']
            tool_input_dict = json.loads(tool_input)

            # if the tool is to evaluate chain the chain
            if last_message.additional_kwargs['tool_calls'][idx]['function']['name'] == 'run_query':
                # We construct a ToolInvocation from the function_call
                action = ToolInvocation(
                    tool=last_message.additional_kwargs['tool_calls'][idx]['function']['name'], 
                    tool_input=tool_input_dict,
                )
                # We call the tool_executor and get back a response
                query, df = self.tool_executor.invoke(action)
                response = df.to_string()

                success_info = f"""
                você criou o código:
                {query}

                E essa foi a tabela resultante:
                {response}

                Você deve responder a seguinte pergunta:
                {state['question']}
                """
                print(success_info)
                # We use the response to create a FunctionMessage
                function_message = ToolMessage(
                    content=str(success_info),name=action.tool, tool_call_id=tool_call['id']
                )

                # We return a listm because this will get added to the existing list
                output_dict["messages"].append(function_message)
                output_dict['actions'] = [query]
                output_dict['df'] = df

        print('TOOL OUTPUT')
        print(output_dict)

        return output_dict

    def call_model_mr_camp_emrich(self, state):
        response = self.model_emrich_mr_camp.invoke(state)
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
        print("RESPOSTA AQUILILLILI -->", resposta)
        if not resposta.tool_calls:
            return {"messages": [resposta]}
        else:
            resposta = "Mais informações:"
            resposta = AIMessage(resposta)
            return {"messages": [resposta]}
