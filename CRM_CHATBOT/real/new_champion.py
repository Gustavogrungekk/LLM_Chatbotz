import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence, List
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import awswrangler as wr  # FLAG: Import necessário para execução de queries no Athena

import logging
logging.basicConfig(level=logging.INFO)

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, MessagePlaceholder
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
    actions: Annotated[Sequence[List], operator.add]
    inter: pd.DataFrame
    memory: str
    date_filter: List
    attempts_count: int
    agent: str
    metadados: str
    table_desc: str
    additional_filters: str

class MrAgent():
    def __init__(self):
        # Inicialização dos metadados (supondo que foram carregados via YAML)
        self.metadata = {
            "table_config": {
                "name": "gold",
                "database": "database_w1",
                "workgroup": "workspace",
                "partitions": [
                    {"name": "year", "type": "string", "description": "Ano de referência"},
                    {"name": "month", "type": "string", "description": "Mês de referência (01-12)"},
                    {"name": "canal", "type": "string", "description": "Canal de comunicação"}
                ],
                "security": {
                    "forbidden_operations": ["DELETE", "INSERT", "ALTER", "DROP", "TRUNCATE", "UPDATE"],
                    "maximum_rows": 100000
                },
                "query_examples": [
                    {
                        "description": "Consulta básica com filtro de partições",
                        "sql": "SELECT produto, AVG(potencial) as media_potencial FROM gold WHERE year = '2024' AND month = '5' AND canal = 'VAI' GROUP BY produto LIMIT 100"
                    }
                ]
            },
            "columns": [
                {"name": "cnpj", "type": "bigint", "description": "Campo único para identificar o cliente que recebeu alguma campanha", "ignore_values": None},
                {"name": "produto", "type": "string", "description": "Tipo de produto financeiro", "ignore_values": None},
                {"name": "abastecido", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente foi abastecido!", "ignore_values": None},
                {"name": "potencial", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente foi potencial!", "ignore_values": None},
                {"name": "visto", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente visualizou a campanha!", "ignore_values": None},
                {"name": "clique", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente clicou em uma campanha!", "ignore_values": None},
                {"name": "atuado", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente atuou na campanha!", "ignore_values": None},
                {"name": "disponivel", "type": "int", "description": "Campo numérico podendo ser 0 ou 1 para identificar se um cliente estava disponível para a campanha!", "ignore_values": None},
                {"name": "canal", "type": "string", "description": "Campo para identificar em qual canal foi disparada a campanha", "ignore_values": None},
                {"name": "metrica_engajamento", "type": "double", "description": "Índice de engajamento do cliente", "ignore_values": None}
            ],
            "query_guidelines": [
                "Sempre usar partições year/month/canal como strings",
                "Usar COUNT(DISTINCT CASE WHEN) para métricas binárias",
                "Comparar conversões entre estágios do funil",
                "Validar formato das datas (YYYY para year, MM para month)"
            ]
        }

        # --- Prompt: Extração de Datas
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Como analista de dados brasileiro especialista em python, sua função é extrair as informações relativas a data. "
                "Você está na data: {last_ref}. Sempre forneça a data como um código pd.date_range() com o argumento freq 'ME' e no formato 'YYYY-mm-dd'. "
                "Caso exista apenas a informação mês, retorne start-'0000-mm-01' e end-'0000-mm-dd'. "
                "Caso exista apenas a informação ano, retorne todo o intervalo do ano. "
                "Caso não exista informação de data, retorne pd.date_range(start='0000-00-00', end='0000-00-00', freq='ME'). "
                "Caso a pergunta contenha a expressão \"mês a mês\" ou \"referência\", retorne pd.date_range(start='3333-00-00', end='3333-00-00', freq='ME'). "
                "Caso a pergunta contenha \"último(s) mês(es)\", retorne os últimos meses de acordo com a pergunta. "
                "Nunca forneça intervalos de data maiores que fevereiro de 2025."
            )),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "(question)")
        ])

        # --- Prompt: Enriquecimento da Pergunta da Máquina de Resultados Campanha
        self.enrich_mr_camp_str = (
            "Como engenheiro de prompt, sua função é reescrever e detalhar a pergunta de forma que um modelo de LLM consiga responder. "
            "Considere que você tem acesso à seguinte tabela para enriquecer a resposta: {table_description_mr} {column_context_mr}. "
            "Pergunta do usuário: {question}. "
            "Reescreva de forma sucinta a pergunta indicando quais filtros são necessários para respondê-la. "
            "Atente-se à pergunta! Não infira nada além do que está nela. "
            "Caso a pergunta contenha algum conceito que não está nos metadados, informe que não é possível responder. "
            "Considere que a pergunta possui o seguinte filtro na coluna 'safra': {date_filter}"
        )
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.enrich_mr_camp_str),
            ("user", "{question}")
        ])

        # --- Prompt: Máquina de Resultados Campanha
        self.mr_camp_prompt_str = (
            "Como engenheiro de dados brasileiro, especializado em análise de dados bancários de engajamento e CRM usando Python, "
            "seu papel é responder exclusivamente a perguntas sobre a Máquina de Resultados, um conjunto de dados utilizado para acompanhar o desempenho de campanhas e ações de CRM. "
            "Você tem acesso ao dataframe 'df' com informações sobre: {table_description_mr}. "
            "Baseando-se nas descrições das colunas disponíveis no CSV a seguir: {column_context_mr}. "
            "Identifique quais colunas estão diretamente relacionadas com a pergunta feita no chat. "
            "Depois, desenvolva e execute uma sequência de comandos utilizando a ferramenta 'evaluate_pandas_chain', "
            "estruturando-os da seguinte forma: <BEGIN> -> action1 -> action2 -> action3 -> <END>. "
            "Atente-se: use str.contains() para procurar valores em colunas do tipo string (todos em CAPSLOCK). "
            "Caso a pergunta contenha algum conceito ausente nos metadados, não infira uma resposta. "
            "Retorne uma tabela em markdown quando solicitado."
        )
        self.mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.mr_camp_prompt_str),
            ("user", "{question}")
        ])

        # --- Prompt: Verificação/Melhoria da Pergunta
        self.suges_pergunta_prompt_desc = (
            "Você é um assistente de IA especializado em melhorar a clareza e a completude das perguntas dos usuários. "
            "Analise a pergunta original para identificar se há informações faltantes ou ambíguas. "
            "O dataframe 'df' possui as colunas: {metadados} e a tabela é: {table_desc}. "
            "Se a pergunta estiver clara, confirme o entendimento; caso contrário, solicite mais detalhes."
        )
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.suges_pergunta_prompt_desc),
            ("user", "{question}")
        ])

        # --- Prompt: Análise de Resposta
        self.resposta_prompt_desc = (
            "Você é um analista de dados brasileiro especializado em dados bancários e engajamento do cliente. "
            "Verifique se as respostas fornecidas contêm todas as informações necessárias para responder à pergunta do usuário. "
            "O dataframe 'df' tem as colunas com o seguinte contexto: {metadados}. "
            "Requisição do usuário: {question}. "
            "Se as informações forem suficientes, valide e organize a resposta de forma clara; caso contrário, identifique as lacunas e solicite mais informações usando 'ask_more_info'."
        )
        self.resposta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.resposta_prompt_desc),
            ("user", "{question}")
        ])

    def init_model(self):
        # Inicializa os modelos e ferramentas

        pdt = PandasTool()
        self.pdt = pdt
        if hasattr(pdt, "get_qstring_mr_camp"):
            table_description = pdt.get_qstring_mr_camp()
        else:
            table_description = "Descrição padrão da tabela"

        dt = DocumentTool()
        self.dt = dt
        if hasattr(dt, "get_col_context_mr_camp"):
            column_context = dt.get_col_context_mr_camp()
        else:
            column_context = "Contexto padrão das colunas"

        tools = [pdt.evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Configura o prompt de enriquecimento da pergunta
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(
            table_description_mr=table_description
        )
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(
            column_context_mr=column_context
        )
        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # Configura o prompt da Máquina de Resultados Campanha
        self.mr_camp_prompt = self.mr_camp_prompt.partial(
            table_description_mr=table_description
        )
        self.mr_camp_prompt = self.mr_camp_prompt.partial(
            column_context_mr=column_context
        )
        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)\
            .bind_tools(self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain")

        # Configura a ferramenta de data
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        dates = pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)
        date_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)\
            .bind_tools([DateToolDesc], tool_choice='DateToolDesc')
        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # Configura os prompts para verificação da pergunta e resposta
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(
            table_desc=table_description
        )
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(
            metadados=column_context
        )
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.resposta_prompt = self.resposta_prompt.partial(
            table_desc=table_description
        )
        self.resposta_prompt = self.resposta_prompt.partial(
            metadados=column_context
        )
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)\
            .bind_tools([ask_more_info], parallel_tool_calls=False)

        # Configura o prompt para geração de query SQL
        self.query_generation_prompt_str = (
            "Você é um engenheiro de dados especializado em gerar queries SQL para o Amazon Athena. "
            "Baseando-se na pergunta do usuário e nos metadados da tabela, gere uma query SQL que atenda aos seguintes requisitos: "
            "- Utilize as partições year, month e canal como strings. "
            "- Não permita operações: {forbidden_operations}. "
            "- Limite o retorno a no máximo {maximum_rows} linhas. "
            "- Utilize os metadados abaixo para orientar a consulta: {metadata} "
            "Diretrizes adicionais: {query_guidelines} "
            "Pergunta do Usuário: {question} "
            "Apenas retorne a query SQL final, formatada para ser executada no Athena."
        )
        self.query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.query_generation_prompt_str),
            ("user", "{question}")
        ])
        self.model_query_generator = self.query_generation_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.build_workflow()

    def run(self, context, verbose: bool = True):
        logging.info(f"Streamlit session state: {context}")
        query = context['messages'][-1]["content"]
        memory = context['messages'][:-1]

        # Estado inicial com "question" e "messages"
        inputs = {
            "messages": [HumanMessage(content=query)],
            "actions": ["<BEGIN>"],
            "question": query,
            "memory": memory,
            "attempts_count": 0
        }

        try:
            current_action = []
            inter_list = []
            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):
                logging.info(f"Output: {output}")
                for idx, (key, value) in enumerate(output.items()):
                    if key.endswith("agent") and verbose:
                        logging.info(f"Agent {key} working...")
                    elif key.endswith("_action") and verbose:
                        if value["messages"][0].name == "view_pandas_dataframes":
                            logging.info("Current action: `viewing dataframes`")
                        else:
                            if "actions" in value:
                                logging.info(f"Current action: {value['actions']}")
                                logging.info(f"Current output: {value['inter']}")
                    elif key == "date_extraction" and verbose:
                        logging.info(f"Date filter: {value['date_filter']}")
                    elif key == "sugest_pergunta" and verbose:
                        logging.info(f"Prompt engineering response: {value['messages']}")
                    elif key == "add_count" and verbose:
                        logging.info(f"Attempt count: {value['attempts_count']}")
                    elif key == "resposta" and verbose:
                        logging.info(f"Verifying response: {value['messages']}")
                    elif verbose:
                        logging.info(f"Final output: {value['inter']}")
                        logging.info(f"Final action chain: {' -> '.join(value['actions'])} -> <END>")
                    if "actions" in value:
                        current_action.append("->".join(value["actions"][-1]).replace("<BEGIN> -> ", "").replace("import pandas as pd;", ""))
                    messages = value.get('messages', [])
                    if 'inter' in value:
                        inter = value.get('inter', None)
                        if inter is not None:
                            inter_list.append(inter)
                    logging.info(f"INTER LIST: {inter_list}")
                final_action = current_action[-1] if current_action else ""
                agent_response = messages[-1].content if messages else "No response generated."
                final_table = inter_list[-1] if inter_list else []
                final_message = agent_response.replace('<END>', '').replace('<BEGIN>', '')
        except Exception as e:
            logging.error(f"Error in process: {e}")
            final_message = "Encontramos um problema processando sua pergunta. Tente novamente, com outra abordagem."
            final_action = ''
            final_table = ''
        return final_message, final_action, final_table

    def should_ask(self, state):
        logging.info(f"Attempt count: {state['attempts_count']}")
        last_message = state['messages'][-1]
        if (("An exception occurred" in last_message['content']) and (state['attempts_count'] >= 2)) or (state['attempts_count'] >= 4):
            return "ask"
        logging.info(f"Last message: {last_message['content']}")
        return "not_ask"

    def add_count(self, state):
        messages = state.get('messages', [])
        if not messages:
            return {"attempts_count": state['attempts_count']}
        last_message = messages[-1]
        if "tool_calls" in last_message.additional_kwargs:
            if last_message.additional_kwargs["tool_calls"][0]['function']['name'] != 'view_pandas_dataframes':
                return {"attempts_count": state['attempts_count'] + 1}
        return {"attempts_count": state['attempts_count']}

    def need_info(self, state):
        messages = state.get('messages', [])
        if not messages:
            return "ok"
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informações:"):
            return "more_info"
        return "ok"

    def call_query_generator(self, state):
        question = state.get("question")
        if not question:
            raise ValueError("State is missing the 'question' key")
        logging.info(f"Extracted question: {question}")
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
        logging.info(f"Query generated (before date adjustment): {generated_query}")
        query_lower = generated_query.lower()
        if "year =" not in query_lower and "month =" not in query_lower:
            max_ref = max(self.pdt.get_refs())
            default_year = str(max_ref)[:4]
            default_month = str(max_ref)[4:].zfill(2)
            date_filter = f"year = '{default_year}' AND month = '{default_month}'"
            if "where" in query_lower:
                generated_query = generated_query.rstrip(";") + f" AND {date_filter};"
            else:
                generated_query = generated_query.rstrip(";") + f" WHERE {date_filter};"
            logging.info(f"Date filter injected automatically: {date_filter}")
        logging.info(f"Final Query: {generated_query}")
        df = self.run_query(generated_query)
        if 'safra' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['safra'] = df['year'].astype(str) + df['month'].astype(str).zfill(2)
            logging.info("Column 'safra' created from 'year' and 'month'.")
        self.pdt.df = df
        new_state = dict(state)
        new_state.update({"generated_query": generated_query, "df": df})
        return new_state

    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        new_state = dict(state)
        new_state.update({"date_filter": date_list})
        return new_state

    def call_sugest_pergunta(self, state):
        sugestao = self.sugest_model.invoke(state)
        new_state = dict(state)
        new_state.update({"messages": [sugestao]})
        return new_state

    def call_model_mr_camp_enrich(self, state):
        response = self.model_enrich_mr_camp.invoke(state)
        new_state = dict(state)
        new_state.update({"messages": [response]})
        return new_state

    def call_model_mr_camp(self, state):
        response = self.model_mr_camp.invoke(state)
        new_state = dict(state)
        new_state.update({"messages": [response]})
        return new_state

    def call_tool(self, state):
        messages = state.get('messages', [])
        if not messages:
            return state
        last_message = messages[-1]
        output_dict = {"messages": [], "actions": []}
        if "tool_calls" in last_message.additional_kwargs:
            for tool_call in last_message.additional_kwargs["tool_calls"]:
                tool_input = tool_call['function']['arguments']
                tool_input_dict = json.loads(tool_input)
                if tool_call['function']['name'] == 'evaluate_pandas_chain':
                    tool_input_dict['inter'] = state.get('inter')
                    tool_input_dict['date_filter'] = state.get('date_filter')
                    tool_input_dict['agent'] = state.get('agent')
                action = ToolInvocation(
                    tool=tool_call['function']['name'],
                    tool_input=tool_input_dict
                )
                response, attempted_action, inter = self.tool_executor.invoke(action)
                if "An exception occurred:" in str(response):
                    error_info = (
                        f"You have previously performed the actions: {state.get('actions')}\n"
                        f"Current action: {attempted_action}\n"
                        f"Result.head(10): {response}\n"
                        f"You must correct your approach and continue until you can answer the question: {state.get('question')}\n"
                        f"Continue the chain with the following format: action_i -> action_i+1... -> <END>"
                    )
                    logging.error(error_info)
                    function_message = ToolMessage(
                        content=error_info,
                        name=action.tool,
                        tool_call_id=tool_call.get("id")
                    )
                    output_dict["messages"].append(function_message)
                else:
                    success_info = (
                        f"You have previously performed the actions: {state.get('actions')}\n"
                        f"Current action: {attempted_action}\n"
                        f"Result.head(50): {response}\n"
                        f"You must continue until you can answer the question: {state.get('question')}\n"
                        f"Continue the chain with the following format: action_i -> action_i+1 ... -> <END>"
                    )
                    logging.info(success_info)
                    function_message = ToolMessage(
                        content=success_info,
                        name=action.tool,
                        tool_call_id=tool_call.get("id")
                    )
                    output_dict["messages"].append(function_message)
                    output_dict["actions"].append(attempted_action)
                    output_dict["inter"] = inter
                    logging.info(f"Tool output: {output_dict}")
        new_state = dict(state)
        new_state.update(output_dict)
        return new_state

    def call_resposta(self, state):
        resposta = self.resposta_model.invoke(state)
        logging.info(f"Response: {resposta}")
        if not resposta.tool_calls:
            new_state = dict(state)
            new_state.update({"messages": [resposta]})
            return new_state
        else:
            fallback = "Mais informações:"
            fallback = AIMessage(fallback)
            new_state = dict(state)
            new_state.update({"messages": [fallback]})
            return new_state

    def run_query(self, query: str):
        inicio = datetime.now()
        df = wr.athena.read_sql_query(
            sql=query,
            database='database_w1',
            workgroup='workspace',
            ctas_approach=False
        )
        if not hasattr(self, 'athenas_time'):
            self.athenas_time = []
        self.athenas_time.append(datetime.now() - inicio)
        logging.info(f"TEMPO EXEC ATHENA: {datetime.now() - inicio}")
        return df

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
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