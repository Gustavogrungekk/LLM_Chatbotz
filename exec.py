import boto3
import json
import operator
import os
import random
import re
import sys
import time
from datetime import datetime  # Fixed typo here (removed extra 'a')
from dateutil.relativedelta import relativedelta
from typing import Annotated, List, Sequence, TypedDict
# import plotly.graph_objects as go
from IPython.display import Markdown, display
# import matplotlib.pyplot as plt

import awswrangler as wr
import openai
import pandas as pd
import yaml
from datetime import datetime, timedelta, date

from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from rag_tools.date_tool import DateToolDesc, date_tool
from rag_tools.documents_tools import DocumentTool
from rag_tools.more_info_tool import ask_more_info
from rag_tools.pandas_tools import run_query, PandasTool


from util_functions import get_last_chains
# from utils.auth.openai_key import VaultSecretReader
print('Current location;',os.getcwd())

# Prompts 
# sys.path.append("/app")
from rag_tools.prompts import (
    DATE_PROMPT_DSC,
    ENRICH_MR_CAMP_DSC,
    MR_CAMP_PROMPT_STR_DSC,
    SUGESTAO_PERGUNTA_PROMPT_DSC,
    RESP_PROMPT_DSC,
    QUERY_GENERATION_PROMPT_DSC,
    QUERY_VALIDATOR_PROMPT_DSC,  # Added query validator prompt
    QUERY_ROUTER_PROMPT_DSC,  # Added query router prompt
    SCOPE_VALIDATOR_PROMPT_DSC  # Add scope validator prompt import
)


os.environ["REQUESTS_CA_BUNDLE"] = 'ca_bundle.crt'

# Apos pegar a chave deleta a variavle de ambiente por conta de conflitos.
del os.environ["REQUESTS_CA_BUNDLE"]

# llm = ChatOpenAI(model="gpt-4o", temperature=0, seed=1)

# Definição do estado do agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[List], operator.add]
    inter: pd.DataFrame
    memory: Annotated[Sequence[BaseMessage], operator.add]
    date_filter: List
    attempts_count: int
    agent: str
    metadados: str
    table_desc: str
    additional_filters: str
    generated_query: str
    question: str
    validation_attempts: int  # Track query validation attempts
    query_is_valid: bool  # Flag to track if query is valid
    in_scope: bool  # Flag to track if query is within scope
    scope_reason: str  # Reason why a query is out of scope


class MrAgent():
    def __init__(self):

        # === METADATA EXTRACTION ===
        with open("src/data/metadata/metadata.yaml", "r", encoding='utf-8') as f:
            self.metadata = yaml.safe_load(f)     

        # === LLM CONFIGURATION ===
        with open("src/config/llm_config.yaml", "r", encoding='utf-8') as f:
            self.llm_config = yaml.safe_load(f)

        # Initialize LLM with configuration from llm_config.yaml
        self.llm = ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.llm_config["seed"]
        )

        # === PROMPT DATA EXTRACTION ===
        self.date_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DATE_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                ("user", "{question}")
            ]
        )

        # ==== PROMPT ENRIQUECIMENTO MÁQUINA DE RESULTADOS CAMPANHA ====
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ENRICH_MR_CAMP_DSC),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        # === PROMPT MÁQUINA DE RESULTADOS CAMPANHA ===
        self.mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MR_CAMP_PROMPT_STR_DSC),
                MessagesPlaceholder(variable_name="messages")  # Ensure correct usage
            ]
            )
        
        # === PROMPT SUGESTÃO DE PERGUNTA ===
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUGESTAO_PERGUNTA_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                ("user", "{question}")
            ]
        )
        
        # === PROMPT RESPOSTA ===
        self.resposta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESP_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        # === PROMPT GERAÇÃO DE QUERY ===
        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_GENERATION_PROMPT_DSC),
                ("user", "{question}"),
                # MessagesPlaceholder(variable_name="messages", n_messages=1)
            ]
        )    

        # === PROMPT VALIDADOR DE QUERY ATHENA ===
        self.query_validator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_VALIDATOR_PROMPT_DSC),
                ("user", "Valide a seguinte query Athena: {generated_query}"),
                MessagesPlaceholder(variable_name="messages", n_messages=1)
            ]
        )
        
        # === PROMPT ROUTER DE QUERY ===
        self.query_router_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_ROUTER_PROMPT_DSC),
                ("user", "{question}")
            ]
        )

        # === PROMPT SCOPE VALIDATOR ===
        self.scope_validator_prompt = ChatPromptTemplate.from_messages([
            ("system", SCOPE_VALIDATOR_PROMPT_DSC),
            ("user", "{question}")
        ])

    def init_model(self):
        # Inicializa o modelo e ferramentas

        # Inicializa a ferramenta Pandas
        pdt = PandasTool()
        
        # Inicializa a ferramenta Document
        dt = DocumentTool()
        tool_run_query = run_query

        # Configura as ferramentas que serão usadas
        tools = [tool_run_query]
        self.tool_executor = ToolExecutor(tools)

        # Converte as ferramentas para OpenAI
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Agente enriquecedor da Máquina de Resultados Campanha
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        # self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(column_context_mr=dt.get_col_context_mr_camp())

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | self.llm

        # Agente Máquina de Resultados Campanha
        self.mr_camp_prompt = self.mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        # self.mr_camp_prompt = self.mr_camp_prompt.partial(column_context_mr=dt.get_col_context_mr_camp())

        self.model_mr_camp = self.mr_camp_prompt | self.llm
        # .bind_tools(
        #     self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain")

        # Define date Tool
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")

        # Get all dates ref of the dataframe
        dates = pdt.get_refs()

        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = self.llm.bind_tools([DateToolDesc], tool_choice='DateToolDesc')

        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc')  | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | date_tool

        # Inclusão do modelo para verificação da pergunta
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.sugest_model = self.suges_pergunta_prompt | self.llm

        # Inclusão do verificador de resposta
        self.resposta_prompt = self.resposta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.resposta_prompt = self.resposta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.resposta_model = (self.resposta_prompt | self.llm.bind_tools([ask_more_info], parallel_tool_calls=False))

        # Inclusão do verificador do gerador de queries
        self.model_query_generator = self.llm.bind_tools(tools, parallel_tool_calls=False, tool_choice='run_query')
        
        # Inclusão do validador de queries Athena
        self.model_query_validator = self.query_validator_prompt | self.llm
        
        # Inclusão do router de queries
        self.model_query_router = self.query_router_prompt | self.llm

        # Inclusão do scope validator
        self.model_scope_validator = self.scope_validator_prompt | self.llm

        # Construção do workflow
        self.build_workflow()

    def run(self, context, verbose: bool = True):
        print("Streamlit session state:")
        print(context)
        print(type(context))
    
        last_message = context["messages"][-1]
        query = last_message["content"]
        memory = context["messages"][:-1]
    
        # Print the initial input state for debugging
        inputs = {
            "messages": [HumanMessage(content=query)],
            "actions": ["<BEGIN>"],
            "question": query,  
            "memory": memory,
            "attempts_count": 0,
        }
        context['question'] = query
        print("DEBUG - Inputs before starting workflow:", inputs)
    
        try:
            current_action = []
            inter_list = []

            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):
                print("DEBUG - Workflow Output:", output)  # Debug print

                for idx, (key, value) in enumerate(output.items()):
                    print(f"DEBUG - Key: {key}, Value: {value}")

                    if "question" not in value:
                        value["question"] = query  # Ensure question is present in state
                        print(f"ERROR - 'question' MISSING in state at step: {key} -> {value}")

                    if key.endswith("agent") and verbose:
                        print(f"Agent {key} working...")
                    elif key.endswith("_action") and verbose:
                        print(f"Current action: `{value.get('actions', [])}`")
                        print(f"Current output: {value.get('inter', '')}")
                    elif key == "date_extraction" and verbose:
                        print(value.get("date_filter", "No date filter found"))
                        print("Date filter for the current question:")
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
                        current_action.append("->".join(value["actions"][-1]).replace("<BEGIN> -> ", "").replace("import pandas as pd;", ""))
                    
                    if "inter" in value:
                        inter = value.get("inter", None)
                        if inter is not None:
                            inter_list.append(inter)

                final_action = current_action[-1] if current_action else ""
                final_table = inter_list[-1] if inter_list else []
                final_message = value.get("messages", [""])[-1].content if "messages" in value else ""

        except Exception as e:
            print("Houve um erro no processo:")
            print(e)
            final_message = "Encontramos um problema processando sua pergunta. Tente novamente, com outra abordagem."
            final_action = ""
            final_table = ""
    
        # return self.process_response(final_message), final_action, final_table
        return final_message, final_action, final_table

    def should_ask(self, state):
        print(f"QUANTIDADE DE TENTATIVAS: {state['attempts_count']}")
        last_message = state['messages'][-1]
        if (("An exception occured" in last_message.content) and (state['attempts_count'] >= 2)) or (state['attempts_count'] >= 4):
            return "ask"
        else:
            print(f"Última mensagem: {last_message.content}")
            return "not_ask"

    def add_count(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls'):
            return {"attempts_count": state['attempts_count']}
        else:
            if last_message.additional_kwargs['tool_calls'][0]['function']['name'] != 'view_pandas_dataframes':
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
        print('PRINTANDO STATE', state['question'])
        question = state["messages"][-1].content
        metadata_str = json.dumps(self.metadata, indent=2)
        forbidden_operations = ", ".join(self.metadata["table_config"]["security"]["forbidden_operations"])
        maximum_rows = self.metadata["table_config"]["security"]["maximum_rows"]
        query_guidelines = "\n".join(self.metadata.get("query_guidelines", []))
        
        # Add query examples
        query_examples = "\n".join(
            [f"{example['description']}:\n{example['sql']}" for example in self.metadata["table_config"]["query_examples"]]
        )

        _ = self.query_generation_prompt.partial(
            forbidden_operations=forbidden_operations,
            maximum_rows=maximum_rows,
            metadata=metadata_str,
            query_guidelines=query_guidelines,
            question=question,
            query_examples=query_examples,
        )
        response = self.model_query_generator.invoke({
            "forbidden_operations": forbidden_operations,
            "maximum_rows": maximum_rows,
            "metadata": metadata_str,
            "query_guidelines": query_guidelines,
            "question": question,
            "query_examples": query_examples,
            "messages": state['messages']
        })
       
        return {"messages": [response]}

    # Método para validar queries Athena
    def call_query_validator(self, state):
        generated_query = state.get("generated_query", "")
        if not generated_query and state["messages"] and hasattr(state["messages"][-1], "tool_calls"):
            # Extract query from tool call if available
            tool_call = state["messages"][-1].additional_kwargs['tool_calls'][0]
            if tool_call['function']['name'] == 'run_query':
                tool_args = json.loads(tool_call['function']['arguments'])
                generated_query = tool_args.get("query", "")
        
        # Validate the query
        response = self.model_query_validator.invoke({
            "generated_query": generated_query,
            "messages": state["messages"]
        })
        
        is_valid = "válida" in response.content.lower() and "erro" not in response.content.lower()
        
        # Increment validation attempts
        validation_attempts = state.get("validation_attempts", 0) + 1
        
        return {
            "messages": [response],
            "query_is_valid": is_valid,
            "validation_attempts": validation_attempts,
            "generated_query": generated_query
        }

    # Método para o router de queries
    def call_query_router(self, state):
        # Get question and query examples from metadata
        question = state["question"]
        query_examples = self.metadata["table_config"]["query_examples"]
        
        response = self.model_query_router.invoke({
            "question": question,
            "query_examples": json.dumps(query_examples, indent=2)
        })
        
        # Extract matched query from response if available
        matched_example = None
        for example in query_examples:
            if example["description"].lower() in response.content.lower():
                matched_example = example
                break
                
        if matched_example:
            # If we found a matching example, use its SQL directly
            return {
                "messages": [HumanMessage(content=f"Usando consulta pré-definida para: {matched_example['description']}")],
                "generated_query": matched_example["sql"]
            }
        else:
            # Otherwise, continue with normal flow
            return {"messages": [HumanMessage(content="Nenhuma consulta pré-definida encontrada.")]}    

    # Check if we should continue with validation or give up
    def check_validation_status(self, state):
        if state.get("query_is_valid", False):
            return "valid"
        elif state.get("validation_attempts", 0) >= 3:
            return "failed"
        else:
            return "retry"
    
    # Method to handle validation failure
    def handle_validation_failure(self, state):
        failure_message = AIMessage(content="""
        Desculpe, não consegui gerar uma consulta SQL válida para o Athena após várias tentativas. 
        Por favor, reformule sua pergunta ou forneça detalhes adicionais para que eu possa lhe ajudar melhor.
        """)
        return {"messages": [failure_message]}

    # Método para validar se a pergunta está dentro do escopo
    def validate_scope(self, state):
        question = state["messages"][-1].content if "messages" in state else state.get("question", "")
        
        # Define temas e contextos dentro do escopo
        scope_context = {
            "permitted_topics": [
                "CRM do Banco Itaú", 
                "campanhas de marketing bancário",
                "métricas de performance de campanhas",
                "resultados de campanhas",
                "análise de dados de CRM",
                "segmentação de clientes bancários",
                "performance de produtos bancários",
                "ofertas e promoções bancárias"
            ]
        }
        
        response = self.model_scope_validator.invoke({
            "question": question,
            "permitted_topics": ", ".join(scope_context["permitted_topics"])
        })
        
        # Verificar se a resposta indica que está dentro do escopo
        in_scope = "dentro do escopo" in response.content.lower()
        reason = response.content
        
        return {
            "messages": [response] if not in_scope else state.get("messages", []),
            "in_scope": in_scope,
            "scope_reason": reason
        }
    
    # Método para lidar com perguntas fora do escopo
    def handle_out_of_scope(self, state):
        out_of_scope_message = AIMessage(content="""
        Desculpe, esta pergunta está fora do escopo deste assistente. 
        
        Estou programado para responder apenas perguntas relacionadas ao CRM do Banco Itaú, 
        campanhas de marketing bancário, e métricas de performance de campanhas.
        
        Por favor, reformule sua pergunta dentro deste contexto ou entre em contato com outro canal 
        de atendimento para este tipo de consulta.
        """)
        return {"messages": [out_of_scope_message]}

    # Verifica se uma pergunta está dentro ou fora do escopo
    def check_scope(self, state):
        if state.get("in_scope", True):
            return "in_scope"
        else:
            return "out_of_scope"

    def build_workflow(self):
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes for the existing workflow
        workflow.add_node("scope_validator", self.validate_scope)
        workflow.add_node("out_of_scope", self.handle_out_of_scope)
        workflow.add_node("query_router", self.call_query_router)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("query_validator", self.call_query_validator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("resposta", self.call_resposta)
        workflow.add_node("validation_failure", self.handle_validation_failure)

        # Define END node
        workflow.add_node("END", lambda state: state) 

        # Define entry point for the workflow
        workflow.set_entry_point("scope_validator")
        
        # Add conditional edge to check scope
        workflow.add_conditional_edges(
            "scope_validator",
            self.check_scope,
            {
                "in_scope": "date_extraction",
                "out_of_scope": "out_of_scope"
            }
        )

        workflow.add_edge("out_of_scope", "END")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "query_router")
        workflow.add_edge("query_router", "query_generation")
        workflow.add_edge("query_generation", "query_validator")
        workflow.add_edge("query_validator", "mr_camp_action", condition=lambda state: state.get("query_is_valid", False))
        workflow.add_edge("mr_camp_action", "add_count")        
        workflow.add_edge("add_count", "resposta")        
        workflow.add_edge("resposta", "END")
        workflow.add_edge("validation_failure", "resposta")

        # Add conditional edges for query validation
        workflow.add_conditional_edges(
            "query_validator",
            self.check_validation_status,
            {
                "valid": "mr_camp_action",
                "retry": "query_generation",
                "failed": "validation_failure"
            }
        )

        # Existing conditional edges
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
                "more_info": "mr_camp_enrich_agent",
                "ok": END
            }
        )
        
        # Compile the workflow
        self.app = workflow.compile()

    # Método para execução das ferramentas
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
                    # tool_input_dict['inter'] = state['df']
                    # tool_input_dict['date_filter'] = state['date_filter']
                    # tool_input_dict['agent'] = state['agent']
    
                    # We construct an ToolInvocation from the function_call
                    action = ToolInvocation(
                        tool=last_message.additional_kwargs['tool_calls'][idx]['function']['name'],
                        tool_input=tool_input_dict,
                    )
                    # We call the tool_executor and get back a response
                    # response, attempted_action, inter = self.tool_executor.invoke(action)
                    query, df = self.tool_executor.invoke(action)
                    response = df.to_string()

                    success_info = f"""
                    Você criou o código: 
                    {query}

                    E essa foi a tabela resultante:
                    {response}

                    Você deve responder a seguinte pergunta:
                    {state['question']}
                    """
                    print(success_info)

                    # We use the response to create a FunctionMessage
                    function_message = ToolMessage(
                    content=str(success_info), name=action.tool, tool_call_id=tool_call["id"]
                    )
                    
                    # We return a list, because this will get added to the existing list
                    output_dict["messages"].append(function_message)
                    output_dict["actions"] = [query]
                    output_dict["df"] = df
                        
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
