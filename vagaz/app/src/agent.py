# app/src/agente.py

import yaml
import time
import awswrangler as wr
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def get_llm():
    with open('config/llm_config.yaml', 'r', encoding='utf-8') as llm_file:
        llm_config = yaml.safe_load(llm_file)
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4-0125-preview"),
        temperature=llm_config.get("temperature", 0),
        seed=llm_config.get("seed", 1)
    )

# Classe para gerar curiosidades de forma dinâmica via LLM
class CuriosityAgent:
    def __init__(self, topic="Banco Itaú", prompt_template=None):
        self.topic = topic
        self.llm = get_llm()
        if prompt_template is None:
            self.prompt_template = (
                "Você é um especialista em curiosidades sobre {topic}. "
                "Forneça uma curiosidade interessante, exclusiva e concisa sobre {topic}."
            )
        else:
            self.prompt_template = prompt_template

    def get_curiosity(self):
        template = PromptTemplate(template=self.prompt_template, input_variables=["topic"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(topic=self.topic)

# Sub-agente para enriquecer o contexto
class ContextEnricher:
    def __init__(self, prompt):
        self.prompt = prompt
        self.llm = get_llm()
    
    def enrich(self, context):
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context)

# Sub-agente para extração de datas com suporte para data fixa (enabler)
class DateExtractor:
    def __init__(self, prompt, fixed_date_enabled=False, fixed_date_value=None):
        self.prompt = prompt
        self.llm = get_llm()
        self.fixed_date_enabled = fixed_date_enabled
        self.fixed_date_value = fixed_date_value
    
    def extract(self, context):
        if self.fixed_date_enabled and self.fixed_date_value:
            return self.fixed_date_value
        template = PromptTemplate(template=self.prompt, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context)

# Sub-agente para construir a query SQL
class QueryBuilder:
    def __init__(self, prompt, metadata):
        self.prompt = prompt
        self.metadata = metadata
        self.llm = get_llm()
    
    def build(self, context, date_info):
        template = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "date_info", "metadata"]
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(context=context, date_info=date_info, metadata=self.metadata)

# Sub-agente para gerar insights a partir dos dados
class InsightsAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.llm = get_llm()
    
    def generate(self, df):
        data_str = df.to_csv(index=False)
        template = PromptTemplate(template=self.prompt, input_variables=["data"])
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(data=data_str)

# Sub-agente para gerar visualizações dos dados
class DataVizAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.library = "plotly"  # Pode alternar para "matplotlib"
    
    def plot(self, df):
        if self.library == "plotly":
            import plotly.express as px
            if df.empty:
                return "Nenhum dado para visualizar."
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualização do Potencial")
            return fig.to_html()
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.bar(df[df.columns[0]], df[df.columns[1]])
            ax.set_title("Visualização do Potencial")
            return "Visualização com matplotlib gerada."

# Agente principal que orquestra os sub-agentes
class Agent:
    def __init__(self):
        with open('data/metadata/metadata.yaml', 'r', encoding='utf-8') as meta_file:
            self.metadata = yaml.safe_load(meta_file)
        with open('data/prompts/prompts.yaml', 'r', encoding='utf-8') as prompts_file:
            self.prompts = yaml.safe_load(prompts_file)
        
        self.llm = get_llm()
        
        # Configuração para ambiente de testes (dev)
        test_mode = self.prompts.get('test_mode', False)
        fixed_date_enabled = self.prompts.get('fixed_date_enabled', False) if test_mode else False
        fixed_date_value = self.prompts.get('fixed_date_value', None) if fixed_date_enabled else None
        
        self.context_enricher = ContextEnricher(
            self.prompts.get('context_enrichment', "Enriqueça o contexto: {context}")
        )
        self.date_extractor = DateExtractor(
            self.prompts.get('date_extraction', "Extraia as datas do seguinte contexto: {context}"),
            fixed_date_enabled=fixed_date_enabled,
            fixed_date_value=fixed_date_value
        )
        self.query_builder = QueryBuilder(
            self.prompts.get('query_builder', 
                "Especialista em AWS Athena que irá gerar nossas queries aplicando as boas práticas.\n\n"
                "Metadados:\n{metadata}\n\n"
                "Consulta: {context}\n\n"
                "Monte a query SQL correspondente."
            ),
            self.metadata
        )
        self.insights_agent = InsightsAgent(
            self.prompts.get('insights', 
                "Analise os seguintes dados (em CSV) e gere insights de negócio relevantes: {data}"
            )
        )
        self.dataviz_agent = DataVizAgent(
            self.prompts.get('dataviz', 
                "Especialista em visualização de dados! Crie uma visualização interativa para os dados fornecidos."
            )
        )
        self.curiosity_agent = CuriosityAgent(topic="Banco Itaú")
        
        # Lista de tópicos proibidos para manter o agente no escopo
        self.forbidden_topics = self.prompts.get('forbidden_topics', ['entertainment', 'politics', 'war'])
    
    def init(self):
        print("Agente inicializado com sucesso.")
    
    def run(self, input_data):
        context = input_data.get("context")
        
        # 1. Enriquecer o contexto
        enriched_context = self.context_enricher.enrich(context)
        
        # 1.1 Verificar tópicos proibidos após o enriquecimento
        for topic in self.forbidden_topics:
            if topic.lower() in enriched_context.lower():
                return {"error": f"A consulta contém o tópico '{topic}', que está fora do escopo permitido. Por favor, reformule a pergunta."}
        
        # 2. Extrair as informações de datas/períodos
        date_info = self.date_extractor.extract(enriched_context)
        
        # 3. Construir a query SQL
        query = self.query_builder.build(enriched_context, date_info)
        
        # 4. Executar a query com controle de erros e tentativas
        max_attempts = 3
        attempt = 0
        timeout_minutes = 2
        start_time = time.time()
        df = None

        while attempt < max_attempts:
            try:
                df = wr.athena.read_sql_query(
                    sql=query,
                    database=self.metadata['table_config']['database'],
                    workgroup=self.metadata['table_config']['workgroup']
                )
                break  # Query executada com sucesso
            except Exception as e:
                attempt += 1
                elapsed = time.time() - start_time
                if elapsed > timeout_minutes * 60:
                    break
                curiosity = self.curiosity_agent.get_curiosity()
                print(f"Enquanto consultamos os dados, aqui vai uma curiosidade: {curiosity} | Tentativa: {attempt}")
                time.sleep(10)
        
        if df is None:
            return {"error": "Desculpe, a consulta demorou demais ou encontrou um erro. Tente novamente mais tarde."}
        
        # 5. Gerar insights
        insights = self.insights_agent.generate(df)
        
        # 6. Gerar visualização
        viz_html = self.dataviz_agent.plot(df)
        
        # 7. Montar a resposta final
        resposta = (
            "**Resposta do Agente:**\n\n"
            "📊 **Resultados da Análise**\n"
            f"{insights}\n\n"
            "📈 **Visualização:**\n"
            f"{viz_html}\n\n"
            "🔧 **Query Executada:**\n"
            "```sql\n" + query + "\n```"
        )
        return resposta
