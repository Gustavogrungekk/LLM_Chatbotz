import boto3 
import json 
import operator 
import os 
import random 
import re 
import sys 
import time 
from datetime import datetime, timedelta 
from dateutil.relativedelta import relativedelta
from typing import Annotated, List, Sequence, TypedDict 
# import plotly 
from IPython.display import display
# import pyplot 

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
from langchain_core.output_parsers import JsonOutputKeyToolsParser
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
# 
print('current location', os.getcwd())

# Prompts
# sys.path.append(os.path.join(os.getcwd(), 'prompts'))
from rag_tools.prompts import (
    DATE_PROMPT_DSC,
    ENRICH_MR_CAMP_DSC,
    MR_CAMP_PROMPT_STR_DSC,
    SUGESTAO_PERGUNTA_PROMPT_DSC,
    RESP_PROMPT_DSC,
    QUERY_GENERATION_PROMPT_DSC
    )


os.environ['REQUESTS_CA_BUNDLE'] = 'ca_bundle.crt'

# Apos pegar a chave deleta a variable de ambiente por dconta de conflitos 
del os.environ['REQUESTS_CA_BUNDLE']

# llm 

# Definicicao do estado do agente 
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


class MrAgent():
    def __init__(self):

        # Metadata
        with open('src/data/metadata/metadata.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)

        # LLM Config
        with open('src/config/llm_config.yaml', 'r') as f:
            self.llm_config = yaml.safe_load(f)

        # initialize LLM with configuration
        self.llm = ChatOpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            seed=self.llm_config['seed'],
        )

        # Prompt date extraction 
        self.date_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DATE_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                ("user", "{question}")
            ]
        )

        # PROMPT ENRIQUECIMENTO
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ENRICH_MR_CAMP_DSC),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        # PROMPT MAQUINA DE RESULTADOS
        self.mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MR_CAMP_PROMPT_STR_DSC),
                MessagesPlaceholder(variable_name="messages")  # Ensure correct usage
            ]
        )

        # PROMPT SUGESTAO DE PERGUNTAS
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUGESTAO_PERGUNTA_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                ("user", "{question}")      
            ]
        )

        # PROMPT RESPOSTA 
        self.resposta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESP_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        # PROMPT QUERY GENERATION
        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_GENERATION_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

    def init_model(self):
        # Inicializa o modelo

        # inicializa a ferramenta do pandas
        pdt = PandasTool()

        # Inicializa a ferramenta Document
        dt = DocumentTool()
        tool_run_query = run_query()

        # Configura as ferramentas  que serao usadas 
        tools = [tool_run_query]
        self.tool_executor = ToolExecutor(tools)

        # Converte as ferramentas para o formato do OpenAI
        self.tools = [convert_to_openai_tool(tool) for tool in tools]

        # Agente enriquecedor maquina de resultados campanha 
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        # 

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | self.llm

        # Agente maquina de resuldaos campanha 
        self.mr_camp_prompt = self.mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        #

        self.model_mr_camp = self.mr_camp_prompt | self.llm
        #
        # #
        # 
        # Defube date tool 
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), '%Y%m') + relativedelta(months=1)).strftime('%Y/%m/%d')

        # Get akk dates ref of the dataframe 
        dates = pdt.get_refs()

        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = self.date_prompt | self.llm.bind_tools([DateToolDesc], tool_choice='DateToolDesc')

        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]['pandas_str'])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | date_tool 

        # Inclusão do modelo para verificação da pergunta
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.sugest_model = self.suges_pergunta_prompt | self.llm

        # Inclusão do verificador de resposta
        self.resposta_prompt = self.resposta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.resposta_prompt = self.resposta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.resposta_model = self.resposta_prompt | self.llm.bind_tools([ask_more_info], parallel_tool_calls=False)

        # Inclusão do verificador do gerador de queries
        self.model_query_generator = self.query_generation_prompt | self.llm.bind_tools(tools, parallel_tool_calls=False, tool_choice='run_query')

        # Construção do workflow
        self.build_workflow()

    def run(self, context, verbose: bool=True):
        print('Steamlit Session state:')
        print(context)
        print(type(context))

        last_message = context['messages'][-1]
        query = last_message['content']
        memory = context['memory'][:-1]

        # Print the initial input state for debbuging
        inputs = {
            "messages": [HumanMessage(content=query)],
            "actions": ["<BEGIN>"],
            "question": query,
            "memory": memory,
            "attempts_count": 0,
        }
        context['question'] = query
        print('DEBUG - INPUTS BEFORE STARTING WORKFLOW:', inputs)

        try:
            current_action = []
            inter_list = []

            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):
                print('DEBUG - WORKFLOW OUTPUT:', output)

                for idx, (key, value) in enumerate(output.items()):
                    print('DEBUG - OUTPUT:', key, value)

                    if "question" not in value:
                        value["question"] = query
                        print("ERROR question missing in value")
                    
                    if key.endswith("agent") and verbose:
                        print(f"Agent ({key}) working...")
                    elif key.endswith("_action") and verbose:
                        print(f"Current action: {value.get('actions', [])}")
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
                        print(f"Final action chain: {' -> '.join(str(value.get('actions', [])))} -> <END>")

                    # Ensure actions and messages are stored correctly
                    if "actions" in value.keys():
                        current_action.append(" -> ".join(value["actions"][-1]).replace("<BEGIN> -> ", "").replace("import pandas as pd;", ""))

                    if "inter" in value:
                        inter = value.get("inter", None)
                        if inter is not None:
                            inter_list.append(inter)

                final_action = current_action[-1] if current_action else ""
                final_table = inter_list[-1] if inter_list else []
                final_message = value.get("messages", [""])[-1].get("content", "") if "messages" in value else ""

        except Exception as e:
            print("Houve um erro no processo:")
            print(e)
            final_message = "Encontramos um problema processando sua pergunta. Tente novamente, com outra abordagem."
            final_action = ""
            final_table = ""

        # 
        return final_message, final_action, final_table
    
    def should_ask(self, state):
        print("Quantidade de tetativas: ", {state['attempts_count']})
        last_message = state['messages'][-1]
        if (("An exception occurred" in last_message.content) and (state['attempts_count'] >= 2) or state['attempts_count'] >= 4):
            return 'ask'
        else:
            print("Ultima mensagem: ", last_message.content)
            return "not_ask"
        
    def add_count(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls'):
            return {"Attempts_count": state['attempts_count']}
        else:
            if last_message.additional_kwargs['tool_calls'][0]['function']['name'] != 'view_pandas_dataframe':
                qtd_passos = state['attempts_count'] + 1
                return {"Attempts_count": qtd_passos}
        return {"Attempts_count": state['attempts_count']}

    def need_info(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informacoes:"):
            return 'more_info'
        return 'ok'

    # FLAG DE MUDANÇA: Novo método de geração de query e obtenção do dataframe via Athena
    def call_query_generator(self, state):
        print("PRINTANDO STATE:", state["question"])
        question = state["messages"][-1].content
        metadata_str = json.dumps(self.metadata, indent=2)
        forbidden_operations = ", ".join(self.metadata["table_config"]["security"]["forbidden_operations"])
        maximum_rows = self.metadata["table_config"]["security"]["maximum_rows"]
        query_guidelines = "\n".join(self.metadata.get("query_guidelines", []))

        # Add query examples
        query_examples = "\n".join(
            [f"[example['description']]:\n{example['sql']}" for example in self.metadata["table_config"]["query_examples"]]
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
            "messages": state["messages"],
        })

        return {"messages": [response]}
    
    def build_workflow(self):
        # Create the state graph
        workflow = StateGraph(AgentState)

        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("add_count", self.add_count)
        # workflow.add_edge("query_agent", "add_count")
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("resposta", self.call_resposta)

        # Define END node
        workflow.add_node("END", lambda state: state)

        workflow.set_entry_point("date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "query_generation")
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
                "more_info": "mr_camp_enrich_agent","ok": "END"
            }
        )

        # Compile the workflow
        self.app = workflow.compile()

    # Metodo para execução das ferramentas
    def call_tool(self, state):
        messages = state['messages']

        # 
        #
        last_message = messages[-1]

        output_dict = {}
        output_dict['messages'] = []

        for idx, tool_call in enumerate(last_message.additional_kwargs['tool_calls']):

            # tool_call 
            tool_input = last_message.additional_kwargs['tool_calls'][idx]['function']['arguments']
            tool_input_dict = json.loads(tool_input)


            # if the tool s to evaluate chain the chain 
            if last_message.additional_kwargs['tool_calls'][idx]['function']['name'] == 'run_query':
                #
                #
                #

                # We construct an toolinvocation from the function
                action = ToolInvocation(
                    tool = last_message.additional_kwargs['tool_calls'][idx]['function']['name'],
                        tool_input = tool_input_dict,
                )
                # w call the tool_executor and get back a response
                # response, attempted_action, inter = self.tool_executor.invoke(action)
                query, df = self.tool_executor.invoke(action)
                response = df.to_string()

                success_info = f"""
                Voce criou o codigo:
                {query}
        
                E Essa foi a tabela resultante:
                {response}
                
                Voce deve responder a seguinte pergunta
                {state['question']}
                """
                print(success_info)

                # We use the response to create a FunctionMessage
                function_message = ToolMessage(
                    content=str(success_info), name=action.tool, tool_call_id=["id"]
                )

                # We return a list, because it will get added to the existing list 
                output_dict['messages'].append(function_message)
                output_dict['actions'] = [query]
                output_dict['df'] = df

            print("OUTPUT TOOL:")
            print(output_dict)

            return output_dict
    
    def call_model_mr_camp_envich(self, state):
        response = self.model_envich_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}

    def call_suggest_pergunta(self, state):
        sugestao = self.sugest_model.invoke(state)
        return {"messages": [sugestao]}  # Fixed typo in variable name

    def call_resposta(self, state):
        resposta = self.resposta_model.invoke(state)
        print("RESPOSTA AQUI -->", resposta)  # Fixed typo in print message
        
        # Assuming resposta_tool_calls should be checking something in resposta
        if not hasattr(resposta, 'tool_calls') or not resposta.tool_calls:
            return {"messages": [resposta]}
        else:
            resposta_text = "Mais informações necessárias:"  # Fixed typo and improved variable name
            resposta = AIMessage(content=resposta_text)  # Assuming AIMessage is imported/defined
            return {"messages": [resposta]}
