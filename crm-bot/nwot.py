O erro `cannot import name 'AgentState' from 'langgraph.graph'` indica que a classe `AgentState` não está disponível no módulo `langgraph.graph`. Isso pode acontecer devido a uma incompatibilidade de versão ou porque a classe foi movida/renomeada em versões mais recentes da biblioteca.

Para resolver esse problema, você pode fazer o seguinte:

### 1. **Verifique a Documentação do `langgraph`**
   - Consulte a documentação oficial da biblioteca `langgraph` para confirmar se a classe `AgentState` ainda existe e como ela deve ser usada.
   - Se a classe foi renomeada ou movida, você precisará ajustar o código de acordo.

### 2. **Substitua `AgentState` por um Dicionário**
   Se você não encontrar a classe `AgentState` na documentação, uma solução simples é substituí-la por um dicionário Python comum. O `AgentState` provavelmente era usado para armazenar o estado do agente, e um dicionário pode cumprir o mesmo papel.

   Substitua:
   ```python
   from langchain.agents import StateGraph, AgentState
   ```

   Por:
   ```python
   from langchain.agents import StateGraph
   ```

   E, no método `build_workflow`, substitua:
   ```python
   workflow = StateGraph(AgentState)
   ```

   Por:
   ```python
   workflow = StateGraph(dict)  # Usa um dicionário para armazenar o estado
   ```

   Isso deve resolver o problema, pois o `StateGraph` pode usar um dicionário para armazenar o estado do agente.

### 3. **Atualize o Código Completo**
   Aqui está o código revisado com a substituição de `AgentState` por um dicionário:

   ```python
   import yaml
   import pandas as pd
   import awswrangler as wr
   from datetime import datetime
   from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
   from langchain.schema import HumanMessage, AIMessage
   from langchain.agents import StateGraph

   # Load table metadata
   with open('chatbot/app/config/tables/table_metadata.yaml', 'r') as f:
       TABLE_METADATA = yaml.safe_load(f)

   # Load table persona
   with open('chatbot/app/config/prompts/personas.yaml', 'r') as f:
       TABLE_PERSONA = yaml.safe_load(f)

   class MrAgent:

       def __init__(self):
           self.athena_tool = AthenaQueryTool()  # Instância do AthenaQueryTool
           self.init_prompts()
           self.init_models()
           self.build_workflow()

       def init_prompts(self):
           # Inicializando os prompts
           self.date_prompt = ChatPromptTemplate.from_messages([
               ("system", """Como analista de dados brasileiro especialista em AWS Athena, extraia informações de data para partições. Sempre retorne datas no formato 'YYYY-MM-DD'. Use filtros de partição year/month/ e canal se necessário."""),
               MessagesPlaceholder(variable_name="memory"),
               ("user", '{question}')
           ])
           
           self.mr_camp_prompt_str = f"""
           Como engenheiro de dados especializado em AWS Athena, gere queries SQL seguindo estas regras:
           
           {self.athena_tool.get_query_guidelines()}
           
           Colunas disponíveis:
           {self.athena_tool.get_column_context()}
           
           Diretrizes:
           - Use sempre filtros de partição year/month e canal se especificado pelo usuário
           - Formate valores de data como strings
           - Use COUNT (DISTINCT CASE WHEN) para métricas binárias
           - Limite resultados a {self.athena_tool.metadata['table_config']['security']['maximum_rows']} linhas
           
           Exemplos válidos:
           {chr(10).join((ex['sql'] for ex in self.athena_tool.metadata['table_config']['query_examples']))}
           """
           
           self.mr_camp_output = ChatPromptTemplate.from_messages([
               ("system", self.mr_camp_prompt_str),
               MessagesPlaceholder(variable_name="messages", n_messages=-1)
           ])

       def init_models(self):
           # Inicialização de modelos (caso necessário)
           pass

       def build_workflow(self):
           # Criação do fluxo de trabalho
           workflow = StateGraph(dict)  # Usa um dicionário para armazenar o estado
           
           # Adicionando as funções ao fluxo de trabalho
           workflow.add_node("generate_query", self.generate_query)
           workflow.add_node("execute_query", self.execute_query)
           
           # Definindo o ponto de entrada
           workflow.set_entry_point("generate_query")
           
           # Adicionando transições de estado
           workflow.add_edge("generate_query", "execute_query")
           workflow.add_edge("execute_query", "END")
           
           # Compilando o fluxo de trabalho
           self.app = workflow
           self.app.compile()

       def generate_query(self, state: dict) -> dict:
           # Função para gerar a consulta SQL com base no estado atual
           response = self.model_mr_camp.invoke(state)
           return {
               "messages": [response],
               "query": response.tool_calls[0]['args']['date_filter']
           }

       def execute_query(self, state: dict) -> dict:
           # Função para executar a consulta gerada
           try:
               query = state['query']  # Recupera a query gerada
               print(f"Query gerada: {query}")  # Exibe a query gerada para homologação
               
               # Executando a query usando Athena (ajuste conforme necessário)
               df = wr.athena.read_sql_query(
                   sql_query=query,
                   database=self.athena_tool.metadata['table_config']['database'],
                   workgroup=self.athena_tool.metadata['table_config']['workgroup'],
                   ctas_approach=True
               )
               
               # Exibe o resultado da query
               result_message = f"Resultado da query: \n{df.head().to_markdown()}"
               print(result_message)
               
               return {
                   "messages": [AIMessage(content=result_message)],
                   "inter": df  # Retorna o DataFrame como 'inter'
               }
           except Exception as e:
               # Em caso de erro, exibe a mensagem de erro
               error_msg = f"Erro na query: {str(e)}"
               return {
                   "messages": [AIMessage(content=error_msg)]
               }

       def run(self, context: dict) -> dict:
           # Função para rodar o agente com o contexto fornecido
           inputs = {
               "messages": [HumanMessage(content=context['messages'][-1]["content"])],
               "question": context['messages'][-1]["content"],
               "memory": context['messages'][:-1],  # Atualizado para pegar toda a memória anterior
               "attempts_count": 0
           }
           
           # Inicia o fluxo de trabalho
           result = self.app.invoke(inputs)
           
           # Retorna o resultado das mensagens e o DataFrame (se houver)
           return result['messages'][-1].content, result.get('inter', pd.DataFrame())

       def run_query(self, query: str):
           # Função para executar diretamente uma consulta Athena
           return wr.athena.read_sql_query(
               sql_query=query,
               database=self.athena_tool.metadata['table_config']['database'],
               workgroup=self.athena_tool.metadata['table_config']['workgroup'],
               ctas_approach=False  # Definido como False aqui
           )

   class AthenaQueryTool:

       def __init__(self):
           # Supondo que 'TABLE_METADATA' é carregado de algum lugar
           self.metadata = TABLE_METADATA  # Isso é apenas um exemplo, altere conforme necessário
           self._last_ref = None

       def get_query_guidelines(self):
           return "\n".join(self.metadata['query_guidelines'])

       def get_column_context(self):
           return "\n".join([f"{col['name']} ({col['type']}): {col['description']}" for col in self.metadata['columns']])

       def get_partition_filters(self, date_filter):
           if not date_filter or date_filter[0] == '0000-00-00':
               return ""

           years_months = set()
           for date_str in date_filter:
               if pd.isnull(pd.to_datetime(date_str, errors='coerce')):
                   continue
               dt = datetime.strptime(date_str, "%Y-%m-%d")
               years_months.add((str(dt.year), f"{dt.month:02d}"))

           partition_filters = []
           for year, month in years_months:
               partition_filters.append(f"(year = '{year}' AND month = '{month}')")

           return " AND ".join(partition_filters) if partition_filters else ""

       def generate_sql_query(self, date_filter: list, additional_filters: str = "") -> str:
           base_query = f"SELECT * FROM {self.metadata['table_config']['name']}"
           
           # Adiciona WHERE com filtros de partição
           where_clauses = []
           partition_filter = self.get_partition_filters(date_filter)
           if partition_filter:
               where_clauses.append(partition_filter)

           # Adiciona filtros adicionais
           if additional_filters:
               where_clauses.append(additional_filters)

           # Adiciona filtros de segurança
           where_clauses.append(" AND ".join([
               f"{col['name']} NOT IN ({','.join(map(repr, col['ignore_values']))})"
               for col in self.metadata['columns'] if col['ignore_values']
           ]))

           if where_clauses:
               base_query += " WHERE " + " AND ".join(where_clauses)

           # Adiciona o limite de segurança
           base_query += f"\nLIMIT {self.metadata['table_config']['security']['maximum_rows']}"

           return base_query




AXASASASASAS
python
Copy
self.mr_camp_output = ChatPromptTemplate.from_messages([
    ("system", self.mr_camp_prompt_str),
    MessagesPlaceholder(variable_name="messages", n_messages=-1)  # Remova o n_messages
])
Por:

python
Copy
self.mr_camp_output = ChatPromptTemplate.from_messages([
    ("system", self.mr_camp_prompt_str),
    MessagesPlaceholder(variable_name="messages")  # Sem n_messages
