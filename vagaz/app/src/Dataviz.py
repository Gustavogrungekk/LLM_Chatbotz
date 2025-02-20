import yaml
import plotly.express as px
import matplotlib.pyplot as plt

class DataVizAgent:
    def __init__(self, prompt, config_path="config/dataviz_config.yaml"):
        self.prompt = prompt  # Prompt para auxiliar na escolha do gráfico, se necessário
        # Carrega a configuração de visualizações do YAML
        with open(config_path, "r", encoding="utf-8") as f:
            self.viz_config = yaml.safe_load(f)
        # Seleção inicial de biblioteca (podemos manter Plotly como padrão)
        self.library = "plotly"
        self.llm = get_llm()  # Reutilizamos a função get_llm definida no agente principal

    def choose_plot_type(self, metadata, df):
        """
        Utiliza um LLM para decidir o melhor tipo de gráfico com base nos metadados e colunas do DataFrame.
        """
        available = list(self.viz_config.get("available_plots", {}).keys())
        prompt_template = (
            "Você é um especialista em visualização de dados. Com base nos metadados abaixo:\n\n"
            "{metadata}\n\n"
            "E considerando as colunas disponíveis no DataFrame: {columns}, "
            "escolha o tipo de gráfico mais adequado dentre as seguintes opções: {options}. "
            "Retorne apenas o nome do gráfico (por exemplo, 'bar', 'line', 'scatter', ou 'pie')."
        )
        # Prepara as variáveis
        columns = ", ".join(df.columns)
        options = ", ".join(available)
        formatted_prompt = prompt_template.format(metadata=metadata, columns=columns, options=options)
        template = PromptTemplate(template=formatted_prompt, input_variables=[])
        chain = LLMChain(llm=self.llm, prompt=template)
        chosen_plot = chain.run()
        # Caso o LLM não retorne um dos tipos esperados, usa o padrão
        if chosen_plot.strip().lower() not in available:
            return self.viz_config.get("default", "bar")
        return chosen_plot.strip().lower()

    def plot(self, df, metadata=""):
        # Decide dinamicamente qual gráfico usar
        plot_type = self.choose_plot_type(metadata, df)
        
        if self.library == "plotly":
            if plot_type == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualização do Potencial")
            elif plot_type == "line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Tendência ao longo do tempo")
            elif plot_type == "scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Correlação entre variáveis")
            elif plot_type == "pie":
                # Para gráfico de pizza, assume que df possui uma coluna categórica e uma numérica
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Proporção por Categoria")
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Visualização do Potencial")
            return fig.to_html()
        else:
            # Exemplo simplificado com matplotlib
            if plot_type == "bar":
                fig, ax = plt.subplots()
                ax.bar(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Visualização do Potencial")
            elif plot_type == "line":
                fig, ax = plt.subplots()
                ax.plot(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Tendência ao longo do tempo")
            elif plot_type == "scatter":
                fig, ax = plt.subplots()
                ax.scatter(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Correlação entre variáveis")
            elif plot_type == "pie":
                fig, ax = plt.subplots()
                ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
                ax.set_title("Proporção por Categoria")
            else:
                fig, ax = plt.subplots()
                ax.bar(df[df.columns[0]], df[df.columns[1]])
                ax.set_title("Visualização do Potencial")
            # Aqui seria necessário converter a figura em HTML ou imagem conforme a aplicação
            return "Visualização gerada com matplotlib."