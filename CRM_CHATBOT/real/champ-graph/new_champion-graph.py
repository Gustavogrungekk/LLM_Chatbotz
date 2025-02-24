# agent.py

import json
import pandas as pd
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import awswrangler as wr  # Necessário para execução de queries no Athena
import yaml                # Para carregar o arquivo YAML de metadados

# Importando os prompts definidos em prompts.py
from prompts import (
    DATE_PROMPT,
    ENRICH_MR_CAMP_PROMPT,
    MR_CAMP_PROMPT,
    SUGES_PERGUNTA_PROMPT,
    RESPOSTA_PROMPT,
    QUERY_GENERATION_PROMPT,
    GRAPH_GENERATION_PROMPT  # FLAG DE MUDANÇA: novo prompt para gráficos
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
    def __init__(self):
        # Carrega os metadados a partir do arquivo YAML
        with open("metadata-v2.yaml", "r") as f:
            self.metadata = yaml.safe_load(f)
        
        # FLAG DE MUDANÇA: Switchkey para definir a biblioteca de plotagem ("plotly" ou "matplotlib")
        self.plot_library = "plotly"  # Pode ser alterada para "matplotlib" conforme necessário

        # Inicializa os prompts e os modelos
        self.init_prompts()
        self.init_model()

    def init_prompts(self):
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

        # FLAG DE MUDANÇA: Novo prompt para geração de gráficos
        self.graph_generation_prompt = ChatPromptTemplate.from_messages(
            [("system", GRAPH_GENERATION_PROMPT)]
        )

    def init_model(self):
        self.pdt = PandasTool()
        tool_evaluate_pandas_chain = self.pdt.evaluate_pandas_chain

        self.dt = DocumentTool()

        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = [convert_to_openai_tool(t) for t in tools]

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt.partial(
            table_description_mr=self.pdt.get_qstring_mr_camp(),
            column_context_mr=self.dt.get_col_context_mr_camp()
        ) | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.model_mr_camp = self.mr_camp_prompt.partial(
            table_description_mr=self.pdt.get_qstring_mr_camp(),
            column_context_mr=self.dt.get_col_context_mr_camp()
        ) | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain"
        )

        last_ref = (datetime.strptime(str(max(self.pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        dates = self.pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools([DateToolDesc], tool_choice='DateToolDesc')
        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: self.pdt.get_refs()) | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(
            table_desc=self.pdt.get_qstring_mr_camp(),
            metadados=self.dt.get_col_context_mr_camp()
        )
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.resposta_prompt = self.resposta_prompt.partial(
            table_desc=self.pdt.get_qstring_mr_camp(),
            metadados=self.dt.get_col_context_mr_camp()
        )
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            [ask_more_info], parallel_tool_calls=False
        )

        self.model_query_generator = self.query_generation_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # FLAG DE MUDANÇA: Modelo para geração de gráficos via LLM
        self.model_graph_generator = self.graph_generation_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.build_workflow()

    def run(self, context, verbose: bool = True):
        print("Streamlit session state:")
        print(context)
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
                print(output)
                for key, value in output.items():
                    if key.endswith("agent") and verbose:
                        print(f"Agent {key} working...")
                    elif key.endswith("_action") and verbose:
                        if value["messages"][0].name == "view_pandas_dataframes":
                            print("Current action: viewing dataframes")
                        else:
                            if "actions" in value:
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
                        print(f"Final output: {value['inter']}")
                        print(f"Final action chain: {' -> '.join(value['actions'])} -> <END>")

                    if "actions" in value:
                        current_action.append(" -> ".join(value["actions"][-1]).replace("<BEGIN> -> ", ""))
                    messages = value.get('messages', None)
                    if 'inter' in value and value.get('inter') is not None:
                        inter_list.append(value['inter'])
                    print('---')

                final_action = current_action[-1] if current_action else ""
                agent_response = messages[-1].content
                final_table = inter_list[-1] if inter_list else []
                final_message = agent_response.replace('<END>', '').replace('<BEGIN>', "")

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

    # Método de geração da query e obtenção do dataframe via Athena
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
        print("Query Gerada para Homologação:")
        print(generated_query)
        df = self.run_query(generated_query)
        if 'safra' not in df.columns and 'year' in df.columns and 'month' in df.columns:
            df['safra'] = df['year'].astype(str) + df['month'].astype(str).str.zfill(2)
            print("Coluna 'safra' criada a partir de 'year' e 'month'.")
        self.pdt.df = df
        return {"generated_query": generated_query, "df": df}

    # FLAG DE MUDANÇA: Novo método para geração de gráficos via LLM
    def call_graph_generation(self, state):
        question = state["question"]
        # Prepara o prompt para gerar o código do gráfico com base na switchkey (plot_library)
        prompt_filled = self.graph_generation_prompt.partial(
            library=self.plot_library,
            question=question
        )
        response = self.model_graph_generator.invoke({
            "library": self.plot_library,
            "question": question
        })
        graph_code = response.content.strip()
        print("Código do gráfico gerado para Homologação:")
        print(graph_code)
        # Aqui, você pode executar o código gerado de forma segura. Exemplo:
        # exec(graph_code, globals(), locals())
        # Para segurança, é recomendável avaliar o código antes de executar.
        # Neste exemplo, apenas retornamos o código gerado.
        return {"graph_code": graph_code}

    def build_workflow(self):
        workflow = StateGraph(dict)
        # Adiciona o nó de geração de query para obter os dados
        workflow.add_node("query_generation", self.call_query_generator)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("add_count", self.add_count)
        workflow.add_node("resposta", self.call_resposta)
        # FLAG DE MUDANÇA: Novo nó de geração de gráficos adicionado
        workflow.add_node("graph_generation", self.call_graph_generation)
        workflow.set_entry_point("query_generation")
        workflow.add_edge("query_generation", "date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "mr_camp_agent")
        workflow.add_edge("mr_camp_agent", "add_count")
        workflow.add_edge("add_count", "mr_camp_action")
        # Adiciona uma condicional: se a pergunta envolver "gráfico" ou similar, desvia para o nó de graph_generation
        # Caso contrário, segue para "resposta"
        def route_based_on_question(state):
            q = state["question"].lower()
            if "grafico" in q or "chart" in q or "plot" in q:
                return "graph_generation"
            else:
                return "resposta"
        workflow.add_conditional_edges("mr_camp_action", route_based_on_question, {"graph_generation": "graph_generation", "resposta": "resposta"})
        workflow.add_conditional_edges(
            "resposta",
            self.need_info,
            {"more_info": "mr_camp_enrich_agent", "ok": "END"}
        )
        workflow.add_edge("sugest_pergunta", "END")
        self.app = workflow.compile()

    def call_tool(self, state):
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
