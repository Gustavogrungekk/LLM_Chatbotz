# Basics
import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence, List
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import awswrangler as wr  # FLAG: Import necessário para execução de queries no Athena

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
        # FLAG: Inicialização da variável de metadados a partir do YAML
        # Suponha que os metadados já foram carregados e atribuídos a self.metadata
        # Exemplo:
        # with open("metadata-v2.yaml", "r") as f:
        #     self.metadata = yaml.safe_load(f)
        # Para este exemplo, vamos assumir que self.metadata já está disponível.
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
                        "sql": "SELECT produto, AVG(potencial) as media_potencial FROM gold WHERE year = 2024 AND month = 5 AND canal = 'VAI' GROUP BY produto LIMIT 100"
                    },
                    # ... outros exemplos
                ]
            },
            "columns": [
                {"name": "cnpj", "type": "bigint", "description": "Campo unico para identificar o cliente que recebeu alguma camapnha", "ignore_values": []},
                {"name": "produto", "type": "string", "description": "Tipo de produto financeiro", "ignore_values": []},
                {"name": "abastecido", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente foi abastecido! Exemplo WHERE abastecido=1, COUNT(DISTINCT WHEN abastecido=1)"},
                {"name": "potencial", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente foi potencial! Exemplo WHERE potencial=1, COUNT(DISTINCT WHEN potencial=1)"},
                {"name": "visto", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente visualizou a campanha! Exemplo WHERE visto=1, COUNT(DISTINCT WHEN visto=1)"},
                {"name": "clique", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente clicou em uma campanha! Exemplo WHERE clique=1, COUNT(DISTINCT WHEN clique=1)"},
                {"name": "atuado", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente clicou em uma campanha! Exemplo WHERE atuado=1, COUNT(DISTINCT WHEN atuado=1)"},
                {"name": "disponivel", "type": "int", "description": "Campo numerico podendo ser 0 ou 1 para identificar se um cliente clicou em uma campanha! Exemplo WHERE disponivel=1, COUNT(DISTINCT WHEN disponivel=1)"},
                {"name": "canal", "type": "string", "description": "Campo para identificar em qual canal foi disparado a campanha"},
                {"name": "metrica_engajamento", "type": "double", "description": "Índice de engajamento do cliente"}
            ],
            "query_guidelines": [
                "Sempre usar partições year/month/canal como strings",
                "Usar COUNT(DISTINCT CASE WHEN) para métricas binárias",
                "Comparar conversões entre estágios do funil",
                "Validar formato das datas (YYYY para year, MM para month)"
            ]
        }
        # Continuação da inicialização
        pass

    # ############## PROMPT EXTRAÇÃO DE DATAS ##############
    self.date_prompt = ChatPromptTemplate.from_messages(
        ("system", """
        Como analista de dados brasileiro especialista em python, sua função é extrair as informações relativas a data.
        
        Você está na data: {last_ref}
        
        Sempre forneça a data como um código pd.date_range() com o argumento freq 'ME' e no formato 'YYYY-mm-dd'.
        Caso exista apenas a informação mês, retorne start-'0000-mm-01' e end-'0000-mm-dd'.
        Caso exista apenas a informação ano, retorne todo o intervalo do ano.
        Caso não exista informação de data, retorne pd.date_range(start='0000-00-00', end='0000-00-00', freq='ME').
        Caso a pergunta contenha a expressão "mês a mês" ou "referência", retorne pd.date_range(start='3333-00-00', end='3333-00-00', freq='ME').
        Caso a pergunta contenha "último(s) mês(es)", retorne os últimos meses de acordo com a pergunta.
        
        Nunca forneça intervalos de data maiores que fevereiro de 2025.
        """),
        (MessagesPlaceholder(variable_name="memory"), "user", "(question}")
    )

    # ############## PROMPT ENRIQUECIMENTO PERGUNTA MÁQUINA DE RESULTADOS CAMPANHA ##############
    self.enrich_mr_camp_str = """
    Como engenheiro de prompt, sua função é reescrever e detalhar a pergunta de forma que um modelo de LLM consiga responder.
    
    Considere que você tem acesso a seguinte tabela para enriquecer a resposta:
    {table_description_mr}
    {column_context_mr}
    
    Pergunta do usuário:
    {question}
    
    Reescreva de forma sucinta a pergunta indicando quais filtros são necessários realizar para respondê-la.
    Atente-se à pergunta! Não infira nada além do que está nela.
    
    Caso a pergunta contenha algum conceito que não está nos metadados, redija a pergunta de forma a dizer que não consegue responder.
    
    Considere que a pergunta possui o seguinte filtro na coluna 'safra': {date_filter}
    """

    self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", self.enrich_mr_camp_str),
            MessagePlaceholder(variable_name="memory"),
            MessagePlaceholder(variable_name="messages")
        ]
    )

    # ############## PROMPT MÁQUINA DE RESULTADOS CAMPANHA ##############
    self.mr_camp_prompt_str = """
    Como engenheiro de dados brasileiro, especializado em análise de dados bancários de engajamento e CRM (Customer Relationship Management) usando a linguagem de programação Python, seu papel é responder exclusivamente a perguntas sobre a Máquina de Resultados, um conjunto de dados utilizado para acompanhar o desempenho de campanhas e ações de CRM.
    
    Você tem acesso ao dataframe 'df' com informações sobre:
    {table_description_mr}
    
    Baseando-se nas descrições das colunas disponíveis no CSV a seguir:
    {column_context_mr}
    
    Identifique quais colunas estão diretamente relacionadas com a pergunta feita no chat. Após essa identificação, desenvolva e execute uma sequência de comandos utilizando a ferramenta 'evaluate_pandas_chain', estruturando-os da seguinte maneira:
    
    <BEGIN> -> action1 -> action2 -> action3 -> <END>.
    
    Atente-se às seguintes observações:
    
    - Sempre use str.contains() para procurar os valores nas colunas do tipo string.
    - Todos os valores das colunas do tipo string estão em CAPSLOCK.
    - Caso a pergunta contenha algum conceito que não está nos metadados, não infira uma resposta.
    - Retorne uma tabela em markdown sempre que for pedido.
    """
    self.mr_camp_prmopt = ChatPromptTemplate.from_messages(
        [
            ("system", self.mr_camp_prompt_str),
            MessagePlaceholder(variable_name="messages", n_message=1)  # n_messages=1 para pegar apenas a última mensagem
        ]
    )

    # Agente para verificação de pergunta
    self.suges_pergunta_prompt_desc = """
    Você é um assistente de IA especializado em melhorar a clareza e a completude das perguntas dos usuários, especialmente após falhas no processo de análise.
    
    Sua tarefa é analisar a pergunta original do usuário para identificar se há informações faltantes ou ambiguas. Seu retorno deve ser focado em questionar o usuário para obter os detalhes necessários que permitirão uma análise mais precisa e uma resposta completa.
    
    Lembre-se de que, embora você não seja capaz de executar códigos, há um outro agente neste sistema que tem acesso ao dataframe 'df' com informações sobre: {table_desc} e é capaz de executar códigos em Python para análise de dados.
    
    Esse dataframe inclui as seguintes colunas: {metadados}.
    
    Se a pergunta estiver clara, confirme o entendimento da pergunta com o usuário. Depois, verifique se falta alguma informação, ou se é preciso especificar mais algum contexto da pergunta para ser possível respondê-la. Se identificar que faltam informações, pergunte ao usuário os detalhes necessários para esclarecer a pergunta, pedindo para ele verificar se o entendimento está correto e responder com 'sim' ou 'não'.
    
    Primeiro, analise a pergunta do usuário para identificar quais colunas estão diretamente relacionadas com a pergunta feita.
    
    Se não souber o que responder, peça para o usuário confirmar seu entendimento da pergunta dele.
    
    Você deve evitar mencionar falhas técnicas ou limitações; concentre-se em melhorar a comunicação com o usuário.
    """
    
    self.suges_pergunta_prompt = ChatPromptTemplate.from_messages(
        ("system", self.suges_pergunta_prompt_desc),
        (MessagesPlaceholder(variable_name="memory"), "(question)")
    )

    # Agente para análise de respostas
    self.resposta_prompt_desc = """
    Você é um analista de dados brasileiro especializado em dados bancários e engajamento do cliente.
    
    Sua função é verificar a qualidade e a completude das respostas técnicas fornecidas pelos assistentes para garantir que todas as informações necessárias estejam presentes para responder corretamente à pergunta do usuário.
    
    Os assistentes têm acesso ao dataframe 'df' com informações sobre:
    
    [table_desc]. As descrições das colunas disponíveis estão no seguinte contexto: {metadados}.
    
    A requisição do usuário é a seguinte: {question}
    
    Primeiro, analise minuciosamente a pergunta do usuário e, em seguida, faça o mesmo com **todas** as informações dadas pelos assistentes. Verifique se as respostas dos assistentes contêm **todas** as informações necessárias para responder à pergunta do usuário.
    
    Se as informações forem suficientes: **Valide**, formate e organize a resposta para que seja clara e compreensível, respondendo **exclusivamente** à pergunta do usuário feita no chat, utilizando sempre todas as informações do assistente. Inclua o período de datas referentes aos valores apresentados.
    
    Se a pergunta não solicitar nenhum dado, apenas responda à requisição de maneira formal e amigável.
    
    Se as informações forem insuficientes: Identifique as lacunas e sempre utilize **exclusivamente** a ferramenta "ask_more_info". Especifique o motivo pelo qual faltam informações e quais são esses dados faltantes. Se as informações não fizerem sentido: utilize **exclusivamente** a ferramenta "ask_more_info".
    
    Mantenha um tom profissional e assertivo. Seja claro ao identificar erros ou lacunas, mas também colaborativo, sugerindo próximos passos de forma construtiva.
    """
    
    self.resposta_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", self.resposta_prompt_desc),
            (MessagesPlaceholder(variable_name="memory"),),
            (MessagesPlaceholder(variable_name="messages"),)
        ]
    )

    def init_model(self):
        # Inicializa o modelo e ferramentas

        # Inicializa a ferramenta Pandas
        pdt = PandasTool()
        self.pdt = pdt
        tool_evaluate_pandas_chain = pdt.evaluate_pandas_chain

        # Inicializa a ferramenta Document
        dt = DocumentTool()
        self.dt = dt

        # Configura as ferramentas que serão usadas
        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)

        # Converte as ferramentas para OpenAI
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Agente enriquecedor da Máquina de Resultados Campanha
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(column_context_mr=dt.get_col_context_mr_camp())

        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # Agente Máquina de Resultados Campanha
        self.mr_camp_prompt = self.mr_camp_prompt.partial(table_description_mr=pdt.get_qstring_mr_camp())
        self.mr_camp_prompt = self.mr_camp_prompt.partial(column_context_mr=dt.get_col_context_mr_camp())

        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain")

        # Define date Tool
        last_ref = (datetime.strptime(str(max(pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        dates = pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref)
        self.date_prompt = self.date_prompt.partial(datas_disponiveis=dates)

        date_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools([DateToolDesc], tool_choice='DateToolDesc')
        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)
        
        # Inclusão do modelo para verificação da pergunta
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # Inclusão do verificador de resposta
        self.resposta_prompt = self.resposta_prompt.partial(table_desc=pdt.get_qstring_mr_camp())
        self.resposta_prompt = self.resposta_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools([ask_more_info], parallel_tool_calls=False)

        # FLAG DE MUDANÇA: Novo prompt para geração de query
        self.query_generation_prompt_str = """
        Você é um engenheiro de dados especializado em gerar queries SQL para o Amazon Athena.
        Baseando-se na pergunta do usuário e nos metadados da tabela, gere uma query SQL que atenda aos seguintes requisitos:
        - Utilize as partições year, month e canal como strings.
        - Não permita operações: {forbidden_operations}.
        - Limite o retorno a no máximo {maximum_rows} linhas.
        - Utilize os metadados abaixo para orientar a consulta:
        {metadata}
        Diretrizes adicionais:
        {query_guidelines}
        Pergunta do Usuário:
        {question}

        Apenas retorne a query SQL final, formatada para ser executada no Athena.
        """
        self.query_generation_prompt = ChatPromptTemplate.from_messages(
            [("system", self.query_generation_prompt_str)]
        )
        self.model_query_generator = self.query_generation_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        # Construção do workflow
        self.build_workflow()

    def run(self, context, verbose: bool = True):
        print("Streamlit session state:")
        print(context)
        print(type(context))

        query = context['messages'][-1]["content"]
        memory = context['messages'][:-1]

        inputs = {"messages": [HumanMessage(content=query)], "actions": ["<BEGIN>"], "question": query, "memory": memory, "attempts_count": 0}

        try:
            current_action = []
            inter_list = []

            for output in self.app.stream(inputs, {"recursion_limit": 100}, stream_mode='updates'):
                print(output)
                for idx, (key, value) in enumerate(output.items()):
                    if key.endswith("agent") and verbose:
                        print(f"Agent {key} working...")
                    elif key.endswith("_action") and verbose:
                        if value["messages"][0].name == "view_pandas_dataframes":
                            print("Current action:")
                            print("`viewing dataframes`")
                        else:
                            if "actions" in value.keys():
                                print(f"Current action:")
                                print(f"`{value['actions']}`")
                                print(f"Current output:")
                                print(value["inter"])
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
                        print(f"Final output:")
                        print(value["inter"])
                        print(f"Final action chain:")
                        print(" -> ".join(value["actions"]) + " -> <END>")

                    if "actions" in value.keys():
                        current_action.append("->".join(value["actions"][-1]).replace("<BEGIN> -> ", "").replace("import pandas as pd;", ""))
                    messages = value.get('messages', None)
                    if 'inter' in value:
                        inter = value.get('inter', None)
                        if inter is not None:
                            inter_list.append(inter)
                    print("INTER LIST NO FOR: \n", inter_list)
                    print('---')

                if current_action:
                    final_action = current_action[-1]
                else:
                    final_action = ""

                agent_response = messages[-1].content
                if inter_list:
                    final_table = inter_list[-1]
                else:
                    final_table = []
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
            database='',
            workgroup='',
            ctas_approach=False
        )
        if not hasattr(self, 'athenas_time'):
            self.athenas_time = []
        self.athenas_time.append(datetime.now() - inicio)
        print(f"TEMPO EXEC ATHENA: {datetime.now() - inicio}")
        return df
