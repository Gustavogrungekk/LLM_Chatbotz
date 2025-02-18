import yaml
import pandas as pd
import awswrangler as wr
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.graphs import StateGraph
from langchain.states import AgentState

# Load table metadata
with open('chatbot/app/config/tables/table_metadata.yaml', 'r') as f:
    TABLE_METADATA = yaml.safe_load(f)

# Load table persona
with open('chatbot/app/config/prompts/personas.yaml', 'r') as f:
    TABLE_PERSONA = yaml.safe_load(f)

class AthenaQueryTool:
    def __init__(self):
        self.metadata = TABLE_METADATA
        self._last_ref = None

    def get_query_guidelines(self):
        return "\n".join(self.metadata['query_guidelines'])

    def get_column_context(self):
        return "\n".join([f"{col['name']} ({col['type']}): {col['description']}" 
                        for col in self.metadata['columns']])

    def get_partition_filters(self, date_filter):
        if not date_filter or date_filter[0] == '0000-00-00':
            return ""

        years_months = set()
        for date_str in date_filter:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                years_months.add((str(dt.year), f"{dt.month:02d}"))
            except:
                continue

        partition_filters = []
        for year, month in years_months:
            partition_filters.append(f"(year = '{year}' AND month = '{month}')")

        return " AND ".join(partition_filters) if partition_filters else ""

    def get_latest_partition(self):
        query = f"""
        SELECT MAX(year) as max_year, MAX(month) as max_month 
        FROM {self.metadata['table_config']['name']}
        """
        try:
            df = wr.athena.read_sql_query(
                sql_query=query,
                database=self.metadata['table_config']['database'],
                workgroup=self.metadata['table_config']['workgroup'],
                ctas_approach=False
            )
            return str(df['max_year'].iloc[0]), f"{df['max_month'].iloc[0]:02d}"
        except:
            return datetime.now().strftime("%Y"), datetime.now().strftime("%m")

class MrAgent:
    def __init__(self):
        self.athena_tool = AthenaQueryTool()
        self.init_prompts()
        self.init_models()
        self.build_workflow()

    def init_prompts(self):
        self.date_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extraia datas de consultas em português brasileiro. Formato final: 'YYYY-MM-DD'.
            Exemplos válidos:
            - "janeiro 2023" → ['2023-01-01']
            - "entre março e maio 2024" → ['2024-03-01', '2024-05-01']
            - "último trimestre" → [<datas calculadas>]"""),
            MessagesPlaceholder(variable_name="memory"),
            ("user", '{question}')
        ])

        self.mr_camp_prompt_str = f"""
        Gere queries SQL para AWS Athena seguindo:
        
        Diretrizes:
        {self.athena_tool.get_query_guidelines()}
        
        Colunas disponíveis:
        {self.athena_tool.get_column_context()}
        
        Regras:
        - Use sempre filtros de partição: {{partition_filters}}
        - Formate datas como strings
        - Máximo de {self.athena_tool.metadata['table_config']['security']['maximum_rows']} linhas
        """

    def init_models(self):
        # Inicialização fictícia de modelos (substituir por implementação real)
        self.model_date_extractor = lambda x: AIMessage(content="2024-01-01")
        self.model_mr_camp = lambda x: AIMessage(content="SELECT * FROM table", 
                                               tool_calls=[{'args': {'query': 'SELECT * FROM table'}}])

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("extract_dates", self.extract_dates)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        
        workflow.set_entry_point("extract_dates")
        workflow.add_edge("extract_dates", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", END)
        
        self.app = workflow.compile()

    def extract_dates(self, state: dict) -> dict:
        chain = self.date_prompt | self.model_date_extractor
        response = chain.invoke({
            "question": state["question"],
            "memory": state.get("memory", [])
        })
        
        # Extração de datas fictícia (implementar parser real)
        extracted_dates = [response.content] if response.content else []
        
        if not extracted_dates:
            year, month = self.athena_tool.get_latest_partition()
            extracted_dates = [f"{year}-{month}-01"]
        
        state["partition_filters"] = self.athena_tool.get_partition_filters(extracted_dates)
        return state

    def generate_query(self, state: dict) -> dict:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.mr_camp_prompt_str.format(
                partition_filters=state["partition_filters"] or "último período disponível")),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        response = self.model_mr_camp(prompt_template.format(messages=state["messages"]))
        generated_query = response.tool_calls[0]['args']['query']
        
        print("\n" + "="*40)
        print("QUERY GERADA PARA HOMOLOGAÇÃO:")
        print(generated_query)
        print("="*40 + "\n")
        
        return {"query": generated_query, "messages": [response]}

    def execute_query(self, state: dict) -> dict:
        try:
            df = wr.athena.read_sql_query(
                sql_query=state["query"],
                database=self.athena_tool.metadata['table_config']['database'],
                workgroup=self.athena_tool.metadata['table_config']['workgroup'],
                ctas_approach=True
            )
            
            insights = []
            if not df.empty:
                insights.append(f"📊 Total de registros: {len(df)}")
                if 'potencial' in df.columns:
                    insights.append(f"⚡ Potencial total: {df['potencial'].sum():,.2f}")
                if 'canal' in df.columns:
                    insights.append(f"📶 Canais presentes: {', '.join(df['canal'].unique())}")
                
                result_msg = f"✅ Resultados:\n{df.head().to_markdown()}\n\n💡 Insights:\n" + "\n".join(insights)
            else:
                result_msg = "ℹ️ Nenhum resultado encontrado com os filtros atuais"
            
            return {"messages": [AIMessage(content=result_msg)], "inter": df}
        
        except Exception as e:
            error_msg = f"❌ Erro na execução da query: {str(e)}"
            return {"messages": [AIMessage(content=error_msg)]}

    def run(self, context: dict) -> dict:
        inputs = {
            "question": context['messages'][-1]["content"],
            "memory": context['messages'][:-1],
            "messages": [HumanMessage(content=context['messages'][-1]["content"])]
        }
        
        result = self.app.invoke(inputs)
        return result['messages'][-1].content, result.get('inter', None)

# Exemplo de uso
if __name__ == "__main__":
    agent = MrAgent()
    test_context = {
        "messages": [{"content": "Qual o potencial do canal VAI em janeiro de 2024?"}]
    }
    response, data = agent.run(test_context)
    print(response)
