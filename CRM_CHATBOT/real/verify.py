# agent.py

import json
import pandas as pd
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import awswrangler as wr  # Necessário para execução de queries no Athena
import yaml                # Para carregar os arquivos YAML
import logging

# Configura o logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importa os prompts definidos em prompts.py
from prompts import (
    DATE_PROMPT,
    ENRICH_MR_CAMP_PROMPT,
    MR_CAMP_PROMPT,
    SUGES_PERGUNTA_PROMPT,
    RESPOSTA_PROMPT,
    QUERY_GENERATION_PROMPT,
    GRAPH_GENERATION_PROMPT
)

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
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


class MrAgent():
    """
    Agente que integra queries dinâmicas, LLMs e visualização,
    operando sobre dados reais via Athena.
    """
    def __init__(self):
        # Carrega os metadados a partir do arquivo YAML
        with open("metadata-v2.yaml", "r") as f:
            self.metadata = yaml.safe_load(f)
        
        # Carrega a configuração da LLM a partir do arquivo llm_config.yaml
        with open("llm_config.yaml", "r") as f:
            self.llm_config = yaml.safe_load(f)
        
        # FLAG DE MUDANÇA: Switchkey para definir a biblioteca de plotagem ("plotly" ou "matplotlib")
        self.plot_library = "plotly"  # Pode ser alterada para "matplotlib" conforme necessário

        # Inicializa os prompts e os modelos
        self.init_prompts()
        self.init_model()

    def init_prompts(self):
        """Inicializa os templates de prompt a partir dos textos importados."""
        self.date_prompt = ChatPromptTemplate.from_messages(
            ("system", DATE_PROMPT),
            (MessagesPlaceholder(variable_name="memory"), "user", "(question)")
        )
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ENRICH_MR_CAMP_PROMPT),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.mr_camp_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MR_CAMP_PROMPT),
                MessagesPlaceholder(variable_name="messages", n_message=1)
            ]
        )
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages(
            ("system", SUGES_PERGUNTA_PROMPT),
            (MessagesPlaceholder(variable_name="memory"), "(question)")
        )
        self.resposta_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPOSTA_PROMPT),
                MessagesPlaceholder(variable_name="memory"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [("system", QUERY_GENERATION_PROMPT)]
        )
        self.graph_generation_prompt = ChatPromptTemplate.from_messages(
            [("system", GRAPH_GENERATION_PROMPT)]
        )

    def init_model(self):
        """Inicializa os modelos LLM e ferramentas, usando a configuração definida em llm_config.yaml."""
        # Inicializa a ferramenta Pandas
        self.pdt = PandasTool()
        tool_evaluate_pandas_chain = self.pdt.evaluate_pandas_chain

        # Inicializa a ferramenta Document
        self.dt = DocumentTool()

        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Recupera os parâmetros da LLM a partir do YAML de configuração
        model = self.llm_config["model"]
        temperature = self.llm_config["temperature"]
        seed = self.llm_config["seed"]

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt.partial(
            table_description_mr=self.pdt.get_qstring_mr_camp(),
            column_context_mr=self.dt.get_col_context_mr_camp()
        ) | ChatOpenAI(model=model, temperature=temperature, seed=seed)

        self.model_mr_camp = self.mr_camp_prompt.partial(
            table_description_mr=self.pdt.get_qstring_mr_camp(),
            column_context_mr=self.dt.get_col_context_mr_camp()
        ) | ChatOpenAI(model=model, temperature=temperature, seed=seed).bind_tools(
            self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain"
        )

        last_ref = (datetime.strptime(str(max(self.pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        dates = self.pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = ChatOpenAI(model=model, temperature=temperature, seed=seed).bind_tools([DateToolDesc], tool_choice='DateToolDesc')
        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: self.pdt.get_refs()) | ChatOpenAI(model=model, temperature=temperature, seed=seed)

        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(
            table_desc=self.pdt.get_qstring_mr_camp(),
            metadados=self.dt.get_col_context_mr_camp()
        )
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(model=model, temperature=temperature, seed=seed)

        self.resposta_prompt = self.resposta_prompt.partial(
            table_desc=self.pdt.get_qstring_mr_camp(),
            metadados=self.dt.get_col_context_mr_camp()
        )
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model=model, temperature=temperature, seed=seed).bind_tools(
            [ask_more_info], parallel_tool_calls=False
        )

        self.model_query_generator = self.query_generation_prompt | ChatOpenAI(model=model, temperature=temperature, seed=seed)
        self.model_graph_generator = self.graph_generation_prompt | ChatOpenAI(model=model, temperature=temperature, seed=seed)

        self.build_workflow()

    def run(self, context, verbose: bool = True):
        """
        Executa o workflow do agente com base no contexto fornecido.

        Parâmetros:
            context (dict): Dicionário contendo as mensagens do usuário.
            verbose (bool): Se True, exibe logs detalhados.

        Retorna:
            tuple: (mensagem_final, ação_final, tabela_final)
        """
        logger.info("Iniciando execução do agente com contexto: %s", context)
        query = context['messages'][-1]["content"]
        memory = context['messages'][:-1]

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
                logger.debug("Saída do stream: %s", output)
                for key, value in output.items():
                    if key.endswith("agent") and verbose:
                        logger.info("Agent %s trabalhando...", key)
                    elif key.endswith("_action") and verbose:
                        if value["messages"][0].name == "view_pandas_dataframes":
                            logger.info("Ação atual: visualizando dataframes")
                        else:
                            if "actions" in value:
                                logger.info("Ação atual: %s", value['actions'])
                                logger.info("Saída atual: %s", value['inter'])
                    elif key == "date_extraction" and verbose:
                        logger.info("Filtro de data: %s", value["date_filter"])
                    elif key == "sugest_pergunta" and verbose:
                        logger.info("Resposta de engenharia de prompt: %s", value["messages"])
                    elif key == "add_count" and verbose:
                        logger.info("Incrementando contador de tentativas: %s", value["attempts_count"])
                    elif key == "resposta" and verbose:
                        logger.info("Verificando resposta: %s", value["messages"])
                    elif verbose:
                        logger.info("Saída final: %s", value['inter'])
                        logger.info("Cadeia de ações final: %s -> <END>", " -> ".join(value['actions']))
                    
                    if "actions" in value:
                        current_action.append(" -> ".join(value["actions"][-1]).replace("<BEGIN> -> ", ""))
                    messages = value.get('messages', None)
                    if 'inter' in value and value.get('inter') is not None:
                        inter_list.append(value['inter'])
                
            final_action = current_action[-1] if current_action else ""
            agent_response = messages[-1].content
            final_table = inter_list[-1] if inter_list else []
            final_message = agent_response.replace('<END>', '').replace('<BEGIN>', "")

        except Exception as e:
            logger.error("Erro durante a execução do agente: %s", e, exc_info=True)
            final_message = "Encontramos um problema processando sua pergunta. Tente novamente, com outra abordagem."
            final_action = ''
            final_table = ''

        return final_message, final_action, final_table

    def should_ask(self, state):
        """
        Verifica se é necessário solicitar mais informações.
        """
        logger.info("Contador de tentativas: %s", state['attempts_count'])
        last_message = state['messages'][-1]
        if (("An exception occured" in last_message['content']) and (state['attempts_count'] >= 2)) or (state['attempts_count'] >= 4):
            return "ask"
        else:
            logger.info("Última mensagem: %s", last_message['content'])
            return "not_ask"

    def add_count(self, state):
        """
        Incrementa o contador de tentativas, se necessário.
        """
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
        """
        Verifica se é necessário solicitar mais informações.
        """
        messages = state['messages']
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informações:"):
            return "more_info"
        return "ok"

    def call_query_generator(self, state):
        """
        Gera a query SQL com base na pergunta do usuário e nos metadados,
        injeta um filtro de data padrão caso nenhum seja especificado,
        executa a query via Athena e atualiza o dataframe.
        """
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
        logger.info("Query Gerada para Homologação (antes do ajuste de data): %s", generated_query)
        
        # Injeta filtro de data se não houver condições explícitas
        query_lower = generated_query.lower()
        if "year" not in query_lower and "month" not in query_lower:
            max_ref = max(self.pdt.get_refs())
            default_year = str(max_ref)[:4]
            default_month = str(max_ref)[4:].zfill(2)
            date_filter = f"year = '{default_year}' AND month = '{default_month}'"
            if "where" in query_lower:
                generated_query = generated_query.rstrip(";") + f" AND {date_filter};"
            else:
                generated_query = generated_query.rstrip(";") + f" WHERE {date_filter};"
            logger.info("Filtro de data injetado automaticamente: %s", date_filter)
        
        logger.info("Query Final para Homologação: %s", generated_query)
        
        df = self.run_query(generated_query)
        if 'safra' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['safra'] = df['year'].astype(str) + df['month'].astype(str).str.zfill(2)
            logger.info("Coluna 'safra' criada a partir de 'year' e 'month'.")
        self.pdt.df = df
        return {"generated_query": generated_query, "df": df}

    def call_graph_generation(self, state):
        """
        Gera o código Python para a criação de um gráfico utilizando a biblioteca definida na switchkey.
        O código é gerado pelo LLM.
        """
        question = state["question"]
        _ = self.graph_generation_prompt.partial(
            library=self.plot_library,
            question=question
        )
        response = self.model_graph_generator.invoke({
            "library": self.plot_library,
            "question": question
        })
        graph_code = response.content.strip()
        logger.info("Código do gráfico gerado para Homologação: %s", graph_code)
        # OBS.: A execução do código gerado deve ser feita com segurança.
        return {"graph_code": graph_code}

    def build_workflow(self):
        """
        Constrói o workflow do agente com nós para geração de query, extração de datas,
        enriquecimento da pergunta, execução de ações, e geração opcional de gráficos.
        """
        workflow = StateGraph(dict)
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("resposta", self.call_resposta)
        workflow.add_node("graph_generation", self.call_graph_generation)
        workflow.set_entry_point("query_generation")
        workflow.add_edge("query_generation", "date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "mr_camp_agent")
        workflow.add_edge("mr_camp_agent", "add_count")
        workflow.add_edge("add_count", "mr_camp_action")
        
        def route_based_on_question(state):
            q = state["question"].lower()
            if "grafico" in q or "chart" in q or "plot" in q:
                return "graph_generation"
            else:
                return "resposta"
        workflow.add_conditional_edges("mr_camp_action", route_based_on_question, {"graph_generation": "graph_generation", "resposta": "resposta"})
        workflow.add_conditional_edges("resposta", self.need_info, {"more_info": "mr_camp_enrich_agent", "ok": "END"})
        workflow.add_edge("sugest_pergunta", "END")
        self.app = workflow.compile()

    def call_tool(self, state):
        """
        Invoca a ferramenta (tool) e retorna o resultado da execução.
        """
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
            logger.error(error_info)
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
            logger.info(success_info)
            function_message = ToolMessage(
                content=str(success_info),
                name=action.tool,
                tool_call_id=tool_call["id"]
            )
            output_dict["messages"].append(function_message)
            output_dict["actions"].append(attempted_action)
            output_dict["inter"] = inter
            logger.info("TOOL OUTPUT: %s", output_dict)
            return output_dict

    def call_model_mr_camp_enrich(self, state):
        """
        Invoca o modelo de enriquecimento para processar a pergunta do usuário.
        """
        response = self.model_enrich_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_model_mr_camp(self, state):
        """
        Invoca o modelo MR camp para gerar a cadeia de ações.
        """
        response = self.model_mr_camp.invoke(state)
        return {"messages": [response]}

    def call_date_extractor(self, state):
        """
        Invoca o modelo de extração de datas para gerar o filtro temporal.
        """
        date_list = self.date_extractor.invoke(state)
        return {"date_filter": date_list}

    def call_sugest_pergunta(self, state):
        """
        Invoca o modelo de sugestão de pergunta para refinar a consulta.
        """
        sugestao = self.sugest_model.invoke(state)
        return {"messages": [sugestao]}

    def call_resposta(self, state):
        """
        Invoca o modelo de resposta para gerar a resposta final ao usuário.
        """
        resposta = self.resposta_model.invoke(state)
        logger.info("RESPOSTA AQUIIIIIII --> %s", resposta)
        if not resposta.tool_calls:
            return {"messages": [resposta]}
        else:
            resposta = "Mais informações:"
            resposta = AIMessage(resposta)
            return {"messages": [resposta]}

    def run_query(self, query: str):
        inicio = datetime.now()
        df = wr.athena.read_sql_query(
            sql=query,
            database='',
            workgroup='',
            ctas_approach=False
        )
        if not hasattr(self, 'athenas_time'):
            self.athenas_time = []
        self.athenas_time.append(datetime.now() - inicio)
        logger.info("TEMPO EXEC ATHENA: %s", datetime.now() - inicio)
        return df
