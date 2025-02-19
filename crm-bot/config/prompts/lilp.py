# Basics
import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence, List
import operator
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableParallel
from langchain.agents import Tool

# Langgraph
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Rag tools
from rag_tools.pandas_tools import PandasTool
from rag_tools.documents_tools import DocumentTool
from rag_tools.date_tool import date_tool, DateToolDesc
from rag_tools.more_info_tool import ask_more_info

# Self-defined functions
from util_functions import get_last_chains

# Define o tipo do estado
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[List], operator.add]
    inter: pd.DataFrame
    memory: str
    date_filter: str
    attempts_count: int
    agent: str
    metadados: str
    table_desc: str
    additional_filters: str
    sql_query: str
    df: pd.DataFrame

class MrAgent():
    def __init__(self):
        # Inicializações dos prompts (todos definidos dentro do __init__ para uso do self)
        self.date_prompt = ChatPromptTemplate.from_messages([
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
            MessagesPlaceholder(variable_name="memory"),
            ("user", "(question)")
        ])

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
        self.enrich_mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.enrich_mr_camp_str),
            MessagesPlaceholder(variable_name="memory"),
            MessagesPlaceholder(variable_name="messages")
        ])

        self.sql_gen_prompt_str = """
        Como especialista em AWS Athena e análise de dados bancários, sua função é criar queries SQL eficientes seguindo estas regras:
        **Tabela e Partições:**
        - Nome da tabela: `tbl_coeres_painel_anomes_v1_gold`
        - Partições obrigatórias (sempre filtrar como strings):
          - `year` (formato 'YYYY', ex: '2025')
          - `month` (formato 'MM', ex: '01')
          - `canal` (valores como 'VAI', 'APP', etc.)

        **Colunas Disponíveis:**
        1. `cnpj` (bigint): Identificador único do cliente
        2. `produto` (string): Tipo de produto financeiro (ex: cartão, empréstimo)
        3. `abastecido` (int): 0 ou 1 - cliente recebeu campanha
        4. `potencial` (int): 0 ou 1 - cliente é potencial
        5. `visto` (int): 0 ou 1 - cliente visualizou a campanha
        6. `clique` (int): 0 ou 1 - cliente clicou na campanha
        7. `atuado` (int): 0 ou 1 - cliente foi atuado
        8. `disponivel` (int): 0 ou 1 - campanha disponível
        9. `metrica_engajamento` (double): Índice de engajamento

        **Regras Obrigatórias:**
        1. SEMPRE inclua filtros para `year`, `month` e `canal` como strings (ex: `year = '2025'`).
        2. Use `COUNT(DISTINCT CASE WHEN coluna = 1 THEN cnpj END)` para métricas binárias.
        3. Selecione explicitamente as colunas necessárias - NUNCA use `SELECT *`.
        4. Valide datas: `year` com 4 dígitos, `month` com 2 dígitos.
        5. Priorize performance com `LIMIT` e filtros adequados.

        **Exemplo de Query Válida:**
        ```sql
        SELECT 
          canal,
          COUNT(DISTINCT CASE WHEN potencial = 1 THEN cnpj END) AS clientes_potenciais,
          COUNT(DISTINCT CASE WHEN visto = 1 THEN cnpj END) AS visualizacoes
        FROM tbl_coeres_painel_anomes_v1_gold
        WHERE year = '2025' 
          AND month = '01'
          AND canal = 'VAI'
        GROUP BY canal
        LIMIT 1000
        ```
        """
        self.sql_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", self.sql_gen_prompt_str),
            ("user", "Pergunta: {question}\nMetadados: {metadados}")
        ])

        self.mr_camp_prompt_str = """
        Como engenheiro de dados brasileiro, especializado em análise de dados bancários e engajamento do cliente, seu papel é responder exclusivamente a perguntas sobre a Máquina de Resultados, um conjunto de dados utilizado para acompanhar o desempenho de campanhas e ações de CRM.
        
        Você tem acesso ao dataframe 'df' obtido através da execução da query SQL no Athena. Esta query foi gerada dinamicamente com base nos metadados e na pergunta do usuário.
        
        Baseando-se nas descrições das colunas disponíveis:
        {column_context_mr}
        
        Identifique quais colunas estão diretamente relacionadas com a pergunta feita no chat. Após essa identificação, desenvolva e execute uma sequência de comandos utilizando a ferramenta 'evaluate_pandas_chain', estruturando-os da seguinte maneira:
        
        <BEGIN> -> action1 -> action2 -> action3 -> <END>.
        
        Atente-se:
        - Use str.contains() para buscar valores em colunas de string.
        - Os valores em colunas string estão em CAPSLOCK.
        - Se a pergunta contiver conceitos não presentes nos metadados, não infira uma resposta.
        - Retorne uma tabela em markdown quando solicitado.
        """
        self.mr_camp_prompt = ChatPromptTemplate.from_messages([
            ("system", self.mr_camp_prompt_str),
            MessagesPlaceholder(variable_name="messages", n_message=1)
        ])

        self.suges_pergunta_prompt_desc = """
        Você é um assistente de IA especializado em melhorar a clareza das perguntas dos usuários.
        
        Analise a pergunta original para identificar se há informações faltantes ou ambiguidades. Caso necessário, questione o usuário para obter os detalhes que permitam uma resposta completa.
        
        Considere que há um dataframe com as seguintes colunas: {metadados} e descrição: {table_desc}.
        
        Se a pergunta estiver clara, confirme o entendimento; se não, peça esclarecimentos.
        """
        self.suges_pergunta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.suges_pergunta_prompt_desc),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "(question)")
        ])

        self.resposta_prompt_desc = """
        Você é um analista de dados especializado em dados bancários e engajamento.
        
        Verifique se as respostas técnicas fornecidas contêm todas as informações necessárias para responder à pergunta do usuário.
        
        Utilize as informações do dataframe (descrição: {table_desc} e metadados: {metadados}) para validar a resposta.
        
        Se as informações estiverem completas, formate a resposta; se não, indique que faltam dados e solicite mais informações.
        """
        self.resposta_prompt = ChatPromptTemplate.from_messages([
            ("system", self.resposta_prompt_desc),
            MessagesPlaceholder(variable_name="memory"),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Inicializa os modelos e ferramentas
        self.init_model()

    # --- Nós adaptados para retornar PromptValues ---
    def call_date_extractor(self, state):
        date_list = self.date_extractor.invoke(state)
        # Se date_list for uma lista de mensagens, converte para string; caso contrário, retorna como string
        if isinstance(date_list, list):
            return " ".join([msg.content for msg in date_list])
        return str(date_list)

    def call_model_mr_camp_enrich(self, state):
        response = self.model_enrich_mr_camp.invoke(state)
        # Retorna uma lista de mensagens
        return [response]

    def call_sql_generator(self, state):
        response = self.sql_gen_model.invoke(state)
        sql_query = response.content.strip()
        # Retorna a query como string
        return sql_query

    def call_run_athena_query(self, state):
        # Obtém a query da etapa anterior (que deve ser uma string)
        query = state.get("sql_query", "")
        if query:
            df = self.run_query(query)
            # Armazena o dataframe no estado para uso posterior (side effect)
            state["df"] = df
        # Retorna uma string vazia para não afetar o fluxo de prompt
        return ""

    def call_model_mr_camp(self, state):
        response = self.model_mr_camp.invoke(state)
        return [response]

    def call_sugest_pergunta(self, state):
        sugestao = self.sugest_model.invoke(state)
        return [sugestao]

    def call_resposta(self, state):
        resposta = self.resposta_model.invoke(state)
        if not resposta.tool_calls:
            return [resposta]
        else:
            return [AIMessage(content="Mais informações:")]

    def end_node(self, state):
        return [AIMessage(content="Fim do workflow.")]

    # Para simplificar, o nó que executa ferramentas agora retorna um resumo em string.
    def call_tool(self, state):
        # Se houver chamadas de ferramenta no último message, executa e retorna um resumo.
        last_message = state["messages"][-1]
        if "tool_calls" in last_message.additional_kwargs:
            for tc in last_message.additional_kwargs["tool_calls"]:
                tool_input = tc["function"]["arguments"]
                tool_input_dict = json.loads(tool_input)
                # Se necessário, modifique o dicionário com dados do estado
                action = ToolInvocation(
                    tool=tc["function"]["name"],
                    tool_input=tool_input_dict
                )
                response, attempted_action, inter = self.tool_executor.invoke(action)
                summary = f"Tool '{action.tool}' executada com ação: {attempted_action}. Resultado: {response}"
                return summary
        return "Nenhuma ferramenta executada."

    # --- Configuração dos modelos e workflow ---
    def init_model(self):
        pdt = PandasTool()
        self.pdt = pdt
        tool_evaluate_pandas_chain = pdt.evaluate_pandas_chain

        dt = DocumentTool()
        self.dt = dt

        tools = [tool_evaluate_pandas_chain]
        self.tool_executor = ToolExecutor(tools)
        self.tools = [convert_to_openai_tool(t) for t in tools]

        # Configuração dos prompts e modelos
        self.enrich_mr_camp_prompt = self.enrich_mr_camp_prompt.partial(
            table_description_mr=pdt.get_qstring_mr_camp()
        ).partial(
            column_context_mr=dt.get_col_context_mr_camp()
        )
        self.model_enrich_mr_camp = self.enrich_mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.sql_gen_prompt = self.sql_gen_prompt.partial(metadados=dt.get_col_context_mr_camp())
        self.sql_gen_model = self.sql_gen_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.mr_camp_prompt = self.mr_camp_prompt.partial(column_context_mr=dt.get_col_context_mr_camp())
        self.model_mr_camp = self.mr_camp_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            self.tools, parallel_tool_calls=False, tool_choice="evaluate_pandas_chain"
        )

        last_ref = (datetime.strptime(str(max(pdt.get_refs())), "%Y%m") + relativedelta(months=1)).strftime("%Y/%m/%d")
        dates = pdt.get_refs()
        self.date_prompt = self.date_prompt.partial(last_ref=last_ref).partial(datas_disponiveis=dates)
        date_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            [DateToolDesc], tool_choice='DateToolDesc'
        )
        partial_model = self.date_prompt | date_llm | JsonOutputKeyToolsParser(key_name='DateToolDesc') | (lambda x: x[0]["pandas_str"])
        self.date_extractor = RunnableParallel(pandas_str=partial_model, refs_list=lambda x: pdt.get_refs()) | ChatOpenAI(
            model="gpt-4-0125-preview", temperature=0, seed=1
        )

        self.suges_pergunta_prompt = self.suges_pergunta_prompt.partial(
            table_desc=pdt.get_qstring_mr_camp()
        ).partial(
            metadados=dt.get_col_context_mr_camp()
        )
        self.sugest_model = self.suges_pergunta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1)

        self.resposta_prompt = self.resposta_prompt.partial(
            table_desc=pdt.get_qstring_mr_camp()
        ).partial(
            metadados=dt.get_col_context_mr_camp()
        )
        self.resposta_model = self.resposta_prompt | ChatOpenAI(model="gpt-4-0125-preview", temperature=0, seed=1).bind_tools(
            [ask_more_info], parallel_tool_calls=False
        )

        self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("date_extraction", self.call_date_extractor)
        workflow.add_node("mr_camp_enrich_agent", self.call_model_mr_camp_enrich)
        workflow.add_node("sql_generator", self.call_sql_generator)
        workflow.add_node("run_athena_query", self.call_run_athena_query)
        workflow.add_node("mr_camp_agent", self.call_model_mr_camp)
        workflow.add_node("add_count", self.add_count)  # Função condicional, não é chamada diretamente para prompt
        workflow.add_node("mr_camp_action", self.call_tool)
        workflow.add_node("sugest_pergunta", self.call_sugest_pergunta)
        workflow.add_node("resposta", self.call_resposta)
        workflow.add_node("END", self.end_node)

        workflow.set_entry_point("date_extraction")
        workflow.add_edge("date_extraction", "mr_camp_enrich_agent")
        workflow.add_edge("mr_camp_enrich_agent", "sql_generator")
        workflow.add_edge("sql_generator", "run_athena_query")
        workflow.add_edge("run_athena_query", "mr_camp_agent")
        workflow.add_edge("mr_camp_agent", "add_count")
        workflow.add_edge("add_count", "mr_camp_action")
        workflow.add_conditional_edges("mr_camp_action", self.should_ask, {"ask": "sugest_pergunta", "not_ask": "resposta"})
        workflow.add_conditional_edges("resposta", self.need_info, {"more_info": "mr_camp_enrich_agent", "ok": "END"})
        workflow.add_edge("sugest_pergunta", "END")
        self.app = workflow.compile()

    def run_query(self, query: str):
        inicio = datetime.now()
        import awswrangler as wr
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

    # As funções de condição (não precisam retornar PromptValues)
    def should_ask(self, state):
        last_message = state['messages'][-1]
        if (("An exception occured" in last_message['content']) and (state['attempts_count'] >= 2)) or (state['attempts_count'] >= 4):
            return "ask"
        return "not_ask"

    def add_count(self, state):
        # Apenas atualiza a contagem; não é usado como prompt
        return {"attempts_count": state['attempts_count'] + 1}

    def need_info(self, state):
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.content.startswith("Mais informações:"):
            return "more_info"
        return "ok"
