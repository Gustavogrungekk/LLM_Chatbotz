""""
Possuo este codigo do qual com base em uma pergunta do usuario nosso agente reposnde de forma dinamica se refrenciando no AWS ATHENA
Porem antes de disparar qualquer query para o Athena gostaria de poder visualziar a query que foi gerada para meios de homologacao!
Assim eu teria certeza do que esta sendo gerado esta dentro do esperado! poderia ate mesmo ser um print na tela!
Mas a ideia e acompanhar o passo a passo do agente.
Pontos 

1 - O agente deve conseguir interpretar dinamicamente o input da pergunta do usuario 
por exemplo qual foi o potencial do canal VAI em janeiro de 2025. E extrair estes campos de datas para conseguir filtrar na tabela!
Caso nenhum periodo seja especificado pegar o ultimo mes.

2 - Oferecer respostas intuitivas com os resultados obtidos ou ate mesmo insights para o usuario finaL!

3 - Forneca-me o codigo completo no termino atualizado ja com todas as modificacoes nada de modicicacoes quebradas por partes!

4 - PONTO CRITICO DO PROCESSO! NAO ALTERAR O COMPORTAMENTO DO AGENTE SOMENTE O QUE LHE FOI PEDIDO. FORNCECER O CODIGO REVISADO E REVISADO POR COMPLETO 
PARA GARANTIR DE QUE ELE IRA FUNCIONAR

arquivo personas.yaml:
date_extractor:
  description: "Especialista em extração de intervalos temporais, conversão para filtros AWS Athena, e interpretação de períodos relativos."
  template: |
    Converta todas as referências temporais em filtros para AWS Athena de acordo com as seguintes regras:
    - **Formato de data:** DATE 'YYYY-MM-DD' (exemplo: '2025-01-15')
    - **Intervalos de data:** Use o operador **BETWEEN** para expressar intervalos, com o formato: `BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`
    - **Data Atual:** Utilize a variável **{current_date}** para fazer referências dinâmicas à data de hoje.
    - **Último Mês Disponível:** Utilize a variável **{last_month}** para se referir ao mês anterior (exemplo: '2025-01-01').
    - **Caso a pergunta **não mencione datas explicitamente**, use a seguinte condição para o último mês disponível:
      `WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}')`
      
    Exemplos de conversão:
    - Para "janeiro de 2025", use:
      `WHERE year = '2025' AND month = '01'`
    - Para um intervalo de janeiro a março de 2025, use:
      `WHERE date_column BETWEEN '2025-01-01' AND '2025-03-31'`
    - Caso não haja menção de datas, sempre considere o último mês disponível:
      `WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}')`
    
    Importante: Considere sempre a coerência no uso de datas e intervalos em AWS Athena.


query_generator:
  description: "Gere consultas SQL otimizadas para o AWS Athena com base na questão apresentada, levando em consideração a estrutura da tabela e os filtros exigidos."
  template: |
    Gere a consulta SQL para a pergunta: {question}
    
    Estrutura da Tabela:
    - **Nome da Tabela:** {table_name}
    - **Partições:** {partitions} (Detalhe as partições da tabela para otimização)
    
    Exemplos de consultas:
    {query_examples}
    
    Regras:
    1. **Filtros obrigatórios:** Use sempre os filtros exigidos por {required_filters}, se não fornecidos, adicione as condições mais relevantes com base na tabela.
    2. **Texto em maiúsculas:** Todos os valores textuais devem ser escritos em maiúsculas (exemplo: "CATEGORY = 'FOOD'").
    3. **Limite de linhas:** Utilize a cláusula `LIMIT {max_rows}` para limitar o número de resultados a {max_rows}.
    4. **Otimização de consulta:** Se a tabela tiver partições, sempre prefira utilizar as partições de forma eficiente para reduzir o custo da consulta.
    
    Exemplos de consultas:
    - Se a pergunta for sobre o total de vendas no último mês, com uma tabela que tenha partições mensais:
      `SELECT SUM(sales) FROM {table_name} WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}') LIMIT {max_rows}`


response_formatter:
  description: "Formate os resultados de consultas SQL, apresentando-os de forma clara e destacando as métricas relevantes para análise."
  template: |
    Formate os resultados de acordo com os dados e filtros fornecidos:
    
    **Dados obtidos:**
    {inter}
    
    **Pergunta original:**
    {question}
    
    **Filtros aplicados:**
    {filters}
    
    **Estrutura:**
    - Use uma tabela **Markdown** para exibir os resultados de forma organizada.
    - Destaque as **métricas** de interesse, como totais, médias ou contagens.
    - Utilize **negrito** para destacar as métricas mais relevantes.
    - Quando necessário, forneça **gráficos ou tabelas adicionais** para melhor visualização (exemplo: somatório de valores por categoria).
    
    Exemplo de saída:
    | Métrica       | Valor   |
    |---------------|---------|
    | Total Vendas  | **$5000** |
    | Quantidade    | **200**   |
    
    **Observações:** Caso haja valores nulos ou ausentes, sempre mencione isso explicitamente no resultado formatado.


data_viz_expert:
  description: "Especialista em visualização de dados, com foco nas bibliotecas Plotly, Seaborn e Matplotlib. Este especialista cria visualizações baseadas nos resultados das queries SQL executadas no AWS Athena. Ele converte dados obtidos de consultas em gráficos impactantes, interativos e claros, com o objetivo de facilitar a análise e tomada de decisões."
  template: |
    **Objetivo:** Criar uma visualização a partir do resultado da consulta SQL gerada pelo especialista em query (AWS Athena).

    **Resultados da Query SQL:**
    {query_results}

    **Escolha da Ferramenta:**
    - **Plotly:** Para gráficos interativos, como gráficos de dispersão, de linhas, de barras ou dashboards.
    - **Seaborn:** Para gráficos estatísticos ou categóricos, como boxplots, heatmaps e pairplots.
    - **Matplotlib:** Para gráficos mais personalizados ou estáticos, quando houver necessidade de maior controle visual.

    **Regras para Visualização:**
    1. **Clareza e Legibilidade:** A visualização deve ser clara, com eixos bem definidos, título, legendas e anotações explicativas, quando necessário.
    2. **Escolha do Tipo de Gráfico:** Baseado nos dados, escolha o gráfico adequado:
       - **Gráfico de Dispersão (Scatter):** Quando se tratar de variáveis numéricas contínuas.
       - **Gráfico de Linhas:** Para representar tendências ao longo do tempo.
       - **Gráfico de Barras:** Quando houver comparações entre categorias.
       - **Boxplot:** Para representar distribuições e outliers.
       - **Heatmap:** Quando for necessário representar a correlação entre variáveis.
    3. **Interatividade (Plotly):** Quando o gráfico envolver grandes volumes de dados ou quando a interação for importante (ex: zoom, hover).
    4. **Estilo e Apresentação:** Use cores, tamanhos e formatos adequados para representar os dados, garantindo uma boa apresentação visual.
    5. **Exportação:** Forneça uma opção para exportar o gráfico gerado para formatos como PNG, JPG ou HTML, dependendo da ferramenta utilizada.

    **Exemplos de Visualizações a partir de Resultados de Query:**
    
    - **Gráfico de Dispersão com Plotly (A partir de resultados de query):**
      ```python
      import plotly.express as px

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      fig = px.scatter(df, x="column_x", y="column_y", color="category_column", title="Gráfico de Dispersão")
      fig.show()  # Exibe o gráfico interativo
      ```

    - **Gráfico de Linha com Matplotlib (A partir de resultados de query):**
      ```python
      import matplotlib.pyplot as plt

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      plt.plot(df['date_column'], df['value_column'], label="Valor")
      plt.title("Variação de Valores ao Longo do Tempo")
      plt.xlabel("Data")
      plt.ylabel("Valor")
      plt.legend()
      plt.show()  # Exibe o gráfico
      ```

    - **Gráfico de Boxplot com Seaborn (A partir de resultados de query):**
      ```python
      import seaborn as sns
      import matplotlib.pyplot as plt

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      sns.boxplot(x="category_column", y="value_column", data=df)
      plt.title("Distribuição de Valores por Categoria")
      plt.show()  # Exibe o gráfico
      ```

    **Considerações Finais:**
    - **Tendências e Correlações:** Se os dados indicarem uma tendência ao longo do tempo, considere adicionar linhas de tendência ou gráficos de dispersão.
    - **Agrupamento por Categorias:** Se os dados contiverem várias categorias (ex: regiões, produtos), utilize gráficos de barras ou boxplots para facilitar comparações.
    - **Interatividade em Dashboards:** Se os resultados forem apresentados em um dashboard, use **Plotly** para gráficos interativos, permitindo que o usuário filtre ou explore diferentes aspectos dos dados.

    **Exemplo de Fluxo:**
    1. O especialista em query gera uma consulta SQL no AWS Athena.
    2. O resultado da consulta é passado para o especialista em visualização.
    3. O especialista cria gráficos com base no resultado da query, aplicando as regras de visualização adequadas.
    4. A visualização é apresentada para análise e tomada de decisão.

    
arquivo table_metadata.yaml:
Possuo este codigo do qual com base em uma pergunta do usuario nosso agente reposnde de forma dinamica se refrenciando no AWS ATHENA
Porem antes de disparar qualquer query para o Athena gostaria de poder visualziar a query que foi gerada para meios de homologacao!
Assim eu teria certeza do que esta sendo gerado esta dentro do esperado! poderia ate mesmo ser um print na tela!
Mas a ideia e acompanhar o passo a passo do agente.
Pontos 

1 - O agente deve conseguir interpretar dinamicamente o input da pergunta do usuario 
por exemplo qual foi o potencial do canal VAI em janeiro de 2025. E extrair estes campos de datas para conseguir filtrar na tabela!
Caso nenhum periodo seja especificado pegar o ultimo disponivel na base do athena! 

2 - Oferecer respostas intuitivas com os resultados obtidos ou ate mesmo insights para o usuario finaL!

3 - Forneca-me o codigo completo no termino atualizado ja com todas as modificacoes nada de modicicacoes quebradas por parntes!

arquivo personas.yaml:
date_extractor:
  description: "Especialista em extração de intervalos temporais, conversão para filtros AWS Athena, e interpretação de períodos relativos."
  template: |
    Converta todas as referências temporais em filtros para AWS Athena de acordo com as seguintes regras:
    - **Formato de data:** DATE 'YYYY-MM-DD' (exemplo: '2025-01-15')
    - **Intervalos de data:** Use o operador **BETWEEN** para expressar intervalos, com o formato: `BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`
    - **Data Atual:** Utilize a variável **{current_date}** para fazer referências dinâmicas à data de hoje.
    - **Último Mês Disponível:** Utilize a variável **{last_month}** para se referir ao mês anterior (exemplo: '2025-01-01').
    - **Caso a pergunta **não mencione datas explicitamente**, use a seguinte condição para o último mês disponível:
      `WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}')`
      
    Exemplos de conversão:
    - Para "janeiro de 2025", use:
      `WHERE year = '2025' AND month = '01'`
    - Para um intervalo de janeiro a março de 2025, use:
      `WHERE date_column BETWEEN '2025-01-01' AND '2025-03-31'`
    - Caso não haja menção de datas, sempre considere o último mês disponível:
      `WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}')`
    
    Importante: Considere sempre a coerência no uso de datas e intervalos em AWS Athena.


query_generator:
  description: "Gere consultas SQL otimizadas para o AWS Athena com base na questão apresentada, levando em consideração a estrutura da tabela e os filtros exigidos."
  template: |
    Gere a consulta SQL para a pergunta: {question}
    
    Estrutura da Tabela:
    - **Nome da Tabela:** {table_name}
    - **Partições:** {partitions} (Detalhe as partições da tabela para otimização)
    
    Exemplos de consultas:
    {query_examples}
    
    Regras:
    1. **Filtros obrigatórios:** Use sempre os filtros exigidos por {required_filters}, se não fornecidos, adicione as condições mais relevantes com base na tabela.
    2. **Texto em maiúsculas:** Todos os valores textuais devem ser escritos em maiúsculas (exemplo: "CATEGORY = 'FOOD'").
    3. **Limite de linhas:** Utilize a cláusula `LIMIT {max_rows}` para limitar o número de resultados a {max_rows}.
    4. **Otimização de consulta:** Se a tabela tiver partições, sempre prefira utilizar as partições de forma eficiente para reduzir o custo da consulta.
    
    Exemplos de consultas:
    - Se a pergunta for sobre o total de vendas no último mês, com uma tabela que tenha partições mensais:
      `SELECT SUM(sales) FROM {table_name} WHERE year = EXTRACT(YEAR FROM DATE '{last_month}') AND month = EXTRACT(MONTH FROM DATE '{last_month}') LIMIT {max_rows}`


response_formatter:
  description: "Formate os resultados de consultas SQL, apresentando-os de forma clara e destacando as métricas relevantes para análise."
  template: |
    Formate os resultados de acordo com os dados e filtros fornecidos:
    
    **Dados obtidos:**
    {inter}
    
    **Pergunta original:**
    {question}
    
    **Filtros aplicados:**
    {filters}
    
    **Estrutura:**
    - Use uma tabela **Markdown** para exibir os resultados de forma organizada.
    - Destaque as **métricas** de interesse, como totais, médias ou contagens.
    - Utilize **negrito** para destacar as métricas mais relevantes.
    - Quando necessário, forneça **gráficos ou tabelas adicionais** para melhor visualização (exemplo: somatório de valores por categoria).
    
    Exemplo de saída:
    | Métrica       | Valor   |
    |---------------|---------|
    | Total Vendas  | **$5000** |
    | Quantidade    | **200**   |
    
    **Observações:** Caso haja valores nulos ou ausentes, sempre mencione isso explicitamente no resultado formatado.


data_viz_expert:
  description: "Especialista em visualização de dados, com foco nas bibliotecas Plotly, Seaborn e Matplotlib. Este especialista cria visualizações baseadas nos resultados das queries SQL executadas no AWS Athena. Ele converte dados obtidos de consultas em gráficos impactantes, interativos e claros, com o objetivo de facilitar a análise e tomada de decisões."
  template: |
    **Objetivo:** Criar uma visualização a partir do resultado da consulta SQL gerada pelo especialista em query (AWS Athena).

    **Resultados da Query SQL:**
    {query_results}

    **Escolha da Ferramenta:**
    - **Plotly:** Para gráficos interativos, como gráficos de dispersão, de linhas, de barras ou dashboards.
    - **Seaborn:** Para gráficos estatísticos ou categóricos, como boxplots, heatmaps e pairplots.
    - **Matplotlib:** Para gráficos mais personalizados ou estáticos, quando houver necessidade de maior controle visual.

    **Regras para Visualização:**
    1. **Clareza e Legibilidade:** A visualização deve ser clara, com eixos bem definidos, título, legendas e anotações explicativas, quando necessário.
    2. **Escolha do Tipo de Gráfico:** Baseado nos dados, escolha o gráfico adequado:
       - **Gráfico de Dispersão (Scatter):** Quando se tratar de variáveis numéricas contínuas.
       - **Gráfico de Linhas:** Para representar tendências ao longo do tempo.
       - **Gráfico de Barras:** Quando houver comparações entre categorias.
       - **Boxplot:** Para representar distribuições e outliers.
       - **Heatmap:** Quando for necessário representar a correlação entre variáveis.
    3. **Interatividade (Plotly):** Quando o gráfico envolver grandes volumes de dados ou quando a interação for importante (ex: zoom, hover).
    4. **Estilo e Apresentação:** Use cores, tamanhos e formatos adequados para representar os dados, garantindo uma boa apresentação visual.
    5. **Exportação:** Forneça uma opção para exportar o gráfico gerado para formatos como PNG, JPG ou HTML, dependendo da ferramenta utilizada.

    **Exemplos de Visualizações a partir de Resultados de Query:**
    
    - **Gráfico de Dispersão com Plotly (A partir de resultados de query):**
      ```python
      import plotly.express as px

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      fig = px.scatter(df, x="column_x", y="column_y", color="category_column", title="Gráfico de Dispersão")
      fig.show()  # Exibe o gráfico interativo
      ```

    - **Gráfico de Linha com Matplotlib (A partir de resultados de query):**
      ```python
      import matplotlib.pyplot as plt

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      plt.plot(df['date_column'], df['value_column'], label="Valor")
      plt.title("Variação de Valores ao Longo do Tempo")
      plt.xlabel("Data")
      plt.ylabel("Valor")
      plt.legend()
      plt.show()  # Exibe o gráfico
      ```

    - **Gráfico de Boxplot com Seaborn (A partir de resultados de query):**
      ```python
      import seaborn as sns
      import matplotlib.pyplot as plt

      # Dados da consulta SQL
      df = {query_results}  # DataFrame com os resultados da query

      sns.boxplot(x="category_column", y="value_column", data=df)
      plt.title("Distribuição de Valores por Categoria")
      plt.show()  # Exibe o gráfico
      ```

    **Considerações Finais:**
    - **Tendências e Correlações:** Se os dados indicarem uma tendência ao longo do tempo, considere adicionar linhas de tendência ou gráficos de dispersão.
    - **Agrupamento por Categorias:** Se os dados contiverem várias categorias (ex: regiões, produtos), utilize gráficos de barras ou boxplots para facilitar comparações.
    - **Interatividade em Dashboards:** Se os resultados forem apresentados em um dashboard, use **Plotly** para gráficos interativos, permitindo que o usuário filtre ou explore diferentes aspectos dos dados.

    **Exemplo de Fluxo:**
    1. O especialista em query gera uma consulta SQL no AWS Athena.
    2. O resultado da consulta é passado para o especialista em visualização.
    3. O especialista cria gráficos com base no resultado da query, aplicando as regras de visualização adequadas.
    4. A visualização é apresentada para análise e tomada de decisão.

    
arquivo table_metadata.yaml:
""""


# Load table metadata
with open('chatbot/app/config/tables/table_metadata.yaml', 'r') as f:
    TABLE_METADATA = yaml.safe_load(f)

# Load table persona
with open('chatbot/app/config/prompts/personas.yaml', 'r') as f:
    TABLE_PERSONA = yaml.safe_load(f)

# Ja tenho todos os import no meu ambiente
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
        workflow = StateGraph(AgentState)  # Define o fluxo de estados
        
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
            print(query)
            
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

# Abaixo estão as implementações adicionais das funções citadas no código original

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

"""


# Load table metadata
with open('chatbot/app/config/tables/table_metadata.yaml', 'r') as f:
    TABLE_METADATA = yaml.safe_load(f)

# Load table persona
with open('chatbot/app/config/prompts/personas.yaml', 'r') as f:
    TABLE_PERSONA = yaml.safe_load(f)

# Ja tenho todos os import no meu ambiente
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
        workflow = StateGraph(AgentState)  # Define o fluxo de estados
        
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
            print(query)
            
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

# Abaixo estão as implementações adicionais das funções citadas no código original

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
