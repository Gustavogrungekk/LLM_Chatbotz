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
from typing import Annotated, List, Sequence, TypeDict 
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


os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# Apos pegar a chave deleta a variable de ambiente por dconta de conflitos 
del os.environ['REQUESTS_CA_BUNDLE']

# llm 

# Definicicao do estado do agente 
class AgentState(TypeDict):
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
        with open('config.yaml', 'r') as f:
            self.metadata = yaml.safe_load(f)

        # LLM Config
        with open('llm_config.yaml', 'r') as f:
            self.llm_config = yaml.safe_load(f)

        # initialize LLM with configuration
        self.llm = ChatOpenAI(
            temperature=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            seed=self.metadata['seed'],
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
        self.sugestao_pergunta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUGESTAO_PERGUNTA_PROMPT_DSC),
                MessagesPlaceholder(variable_name="memory"),
                ("user", "{question}")      
            ]
        )

        # PROMPT RESPOSTA 
        self.resp_prompt = ChatPromptTemplate.from_messages(
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

        # Inicializa o modelo de classificação de entrada
        self.input_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um especialista em classificar consultas de usuários. Determine se a entrada é uma saudação ou uma solicitação de dados. Responda apenas com 'saudacao' ou 'requisicao_dados'."),
            ("user", "{question}")
        ])
        self.input_classifier_model = self.input_classifier_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=0.0,
            seed=self.metadata["seed"]
        )

        # Inicializa o modelo de saudação
        self.greeting_prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente de IA amigável. Responda à saudação do usuário de maneira profissional e cordial em português brasileiro."),
            ("user", "{question}")
        ])
        self.greeting_model = self.greeting_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.metadata["seed"]
        )

        # Inicializa o modelo de exemplos de consultas
        self.query_examples_prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um especialista em SQL. Com base na pergunta do usuário e nos exemplos de consulta disponíveis, selecione os exemplos mais relevantes para ajudar a gerar uma consulta eficaz."),
            MessagesPlaceholder(variable_name="memory"),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Encontre exemplos de consulta relevantes para: {question}")
        ])
        self.query_examples_model = self.query_examples_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.metadata["seed"]
        )

        # Inicializa o modelo de validação SQL
        self.sql_validation_prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um especialista em SQL para AWS Athena. Valide a consulta SQL fornecida quanto a erros de sintaxe e melhores práticas."),
            ("user", "Valide a seguinte consulta para AWS Athena: {query}")
        ])
        self.sql_validation_model = self.sql_validation_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=0.0,
            seed=self.metadata["seed"]
        ).bind_tools([
            {
                "type": "function",
                "function": {
                    "name": "validate_sql",
                    "description": "Valida a sintaxe SQL para AWS Athena",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_valid": {
                                "type": "boolean",
                                "description": "Se o SQL é válido ou não"
                            },
                            "feedback": {
                                "type": "string",
                                "description": "Feedback sobre a consulta SQL em português"
                            }
                        },
                        "required": ["is_valid", "feedback"]
                    }
                }
            }
        ], tool_choice="validate_sql")
        
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

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | ChatOpenAI(
            temperature=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            seed=self.metadata['seed'],
        )

        # Agente maquina de resuldaos campanha 
        self.mr_camp_prompt = self.mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        #

        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(
            temperature=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            seed=self.metadata['seed'],
        )
        #
        # #
        # 
        # Defube date tool 
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), '%Y%m') + timedelta(months=1)).strftime('%Y-%m-%d')

        # Get akk dates ref of the dataframe 
        dates = pdt.get_dates()

        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = self.date_prompt | ChatOpenAI(
            temperature=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            seed=self.metadata['seed'],
        ).bind_tools([DateToolDesc], tool_choice='DateToolDesc')

        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]['pandas_str'])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | date_tool 

        # Inclusão do modelo para verificação da pergunta
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.llm_config["seed"]
        )

        # Inclusão do verificador de resposta
        self.resposta_prompt = self.resposta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.resposta_prompt = self.resposta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.resposta_model = self.resposta_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.llm_config["seed"]
        ).bind_tools([ask_more_info], parallel_tool_calls=False)

        # Inclusão do verificador do gerador de queries
        self.model_query_generator = self.query_generation_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            seed=self.llm_config["seed"]
        ).bind_tools(tools, parallel_tool_calls=False, tool_choice='run_query')

        # Construção do workflow
        self.build_workflow()

    # Método para classificar a intenção do usuário usando LLM
    def call_input_classifier(self, state):
        """
        Utiliza o modelo de linguagem para classificar a intenção do usuário.
        """
        question = state["question"]
        response = self.input_classifier_model.invoke({"question": question})
        return {"messages": [response]}

    # Método para determinar se a solicitação é saudação ou requisição de dados
    def is_valid_request(self, state):
        """
        Utiliza o LLM para classificar a intenção do usuário como saudação ou requisição de dados.
        """
        last_message = state['messages'][-1]
        content = last_message.content.lower()
        
        # Preparar o prompt para classificação
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em classificar mensagens dos usuários.
            Analise a mensagem e determine se é apenas uma saudação ou se é uma requisição de dados.
            
            Exemplos de saudações: olá, bom dia, oi, como vai, tudo bem
            Exemplos de requisição de dados: me mostre, quero saber sobre, qual foi, quantos, analise, gere um relatório
            
            Classifique APENAS como "saudacao" ou "requisicao_dados". Responda com uma única palavra."""),
            ("user", content)
        ])
        
        # Usar o modelo para classificar
        classifier = classification_prompt | ChatOpenAI(
            model=self.llm_config["model"],
            temperature=0.0,
            seed=self.metadata["seed"]
        )
        
        # Obter a classificação do modelo
        result = classifier.invoke({})
        classification = result.content.lower().strip()
        
        print(f"Classificação da mensagem: '{classification}'")
        
        # Retornar com base na classificação do modelo
        if "saudacao" in classification:
            return "greeting"
        else:
            return "data_request"

    # Método para processar saudações
    def greeting_agent(self, state):
        """
        Processa saudações utilizando um modelo de linguagem especializado em português-BR.
        """
        question = state["question"]
        response = self.greeting_model.invoke({"question": question})
        return {"messages": [response]}

    # Método para buscar exemplos de consulta relevantes
    def query_examples(self, state):
        """
        Utiliza LLM para encontrar exemplos de consulta relevantes para a pergunta do usuário.
        """
        question = state["question"]
        
        # Obter exemplos de consulta dos metadados
        examples = self.metadata["table_config"]["query_examples"]
        formatted_examples = json.dumps(examples, indent=2)
        
        # Invocar modelo para encontrar exemplos relevantes
        result = self.query_examples_model.invoke({
            "question": question,
            "messages": state["messages"],
            "memory": state.get("memory", []),
            "examples": formatted_examples
        })
        
        # Adicionar os exemplos selecionados ao estado para geração de consulta
        return {
            "messages": state["messages"],
            "selected_examples": result.content
        }

    # Método para validar a sintaxe SQL
    def sql_validation_agent(self, state):
        """
        Valida a sintaxe SQL da consulta gerada utilizando um modelo especializado.
        """
        # Extrair a consulta SQL da última mensagem
        last_message = state["messages"][-1]
        
        # Tentar encontrar SQL no conteúdo da mensagem
        query = ""
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("function", {}).get("name") == "run_query":
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args.get("query", "")
                    break
        
        if not query:
            # Se não encontrarmos uma consulta, tentar extraí-la do conteúdo
            content = last_message.content
            sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
            if sql_match:
                query = sql_match.group(1)
        
        # Se ainda não tivermos uma consulta, retornar um erro
        if not query:
            return {
                "messages": state["messages"] + [AIMessage(content="Não foi possível encontrar uma consulta SQL válida para validação.")],
                "sql_valid": False,
                "sql_feedback": "Consulta SQL não encontrada."
            }
        
        # Validar a consulta
        response = self.sql_validation_model.invoke({"query": query})
        
        # Extrair resultado da validação
        validation_result = {}
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.get("function", {}).get("name") == "validate_sql":
                    validation_result = json.loads(tool_call["function"]["arguments"])
                    break
        
        # Retornar o resultado da validação
        return {
            "messages": state["messages"] + [AIMessage(content=f"Validação SQL: {validation_result.get('feedback', 'Sem feedback')}")],
            "sql_valid": validation_result.get("is_valid", False),
            "sql_feedback": validation_result.get("feedback", "")
        }

    # Método para verificar se o SQL é válido
    def is_sql_valid(self, state):
        """
        Verifica se a consulta SQL foi validada com sucesso.
        """
        return "valid" if state.get("sql_valid", False) else "invalid"

    def build_workflow(self):
        # Criar o grafo de estado
        workflow = StateGraph(AgentState)

        # Adicionar nós
        workflow.add_node("input_classifier", self.call_input_classifier)
        workflow.add_node("greeting_agent", self.greeting_agent)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("query_examples", self.query_examples)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("sql_validation", self.sql_validation_agent)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("resposta", self.call_resposta)

        # Definir nó END
        workflow.add_node("END", lambda state: state)

        # Definir ponto de entrada
        workflow.set_entry_point("input_classifier")

        # Adicionar arestas condicionais para classificador de entrada
        workflow.add_conditional_edges(
            "input_classifier",
            self.is_valid_request,
            {
                "greeting": "greeting_agent",
                "data_request": "date_extraction"
            }
        )

        # Adicionar aresta para greeting_agent até END
        workflow.add_edge("greeting_agent", "END")

        # Adicionar arestas para fluxo de requisição de dados
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "query_examples")
        workflow.add_edge("query_examples", "query_generation")
        workflow.add_edge("query_generation", "sql_validation")

        # Adicionar arestas condicionais para validação SQL
        workflow.add_conditional_edges(
            "sql_validation",
            self.is_sql_valid,
            {
                "valid": "mr_camp_action",
                "invalid": "query_generation"  # Retorna para query_generation com feedback
            }
        )

        # Adicionar arestas restantes
        workflow.add_edge("mr_camp_action", "add_count")

        # Adicionar arestas condicionais para add_count
        workflow.add_conditional_edges(
            "add_count",
            self.should_ask,
            {
                "ask": "sugest_pergunta",
                "not_ask": "resposta"
            }
        )

        # Adicionar arestas condicionais para resposta
        workflow.add_conditional_edges(
            "resposta",
            self.need_info,
            {
                "more_info": "mr_camp_enrich_agent",
                "ok": "END"
            }
        )

        # Compilar o workflow
        self.app = workflow.compile()

    # Método para execução das ferramentas - corrigindo o nome do método que estava com typo
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
            if last_message.additional_kwargs['tool_calls'][idx]['function']['name'] == 'run_query'
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
    
    def call_model_mr_camp_enrich(self, state):
        response = self.model_enrich_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}

    def call_sugest_pergunta(self, state):
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
