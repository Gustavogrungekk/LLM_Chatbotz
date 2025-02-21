import streamlit as st
import toml
import datetime
import os
from agent import AdvancedAgent  # Certifique-se de que o AdvancedAgent esteja disponível
from streamlit_autorefresh import st_autorefresh  # Instale via: pip install streamlit-autorefresh

# Caminho do arquivo users.toml (na raiz)
USERS_FILE_PATH = "users.toml"
LOGS_FOLDER = "logs/"

# ----- Funções de gerenciamento de usuários localmente -----
def load_users():
    if os.path.exists(USERS_FILE_PATH):
        with open(USERS_FILE_PATH, "r", encoding="utf-8") as f:
            return toml.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE_PATH, "w", encoding="utf-8") as f:
        toml.dump(users, f)

def register_user(reg_info):
    users = load_users()
    racf = reg_info["racf"]
    # Define o status do usuário como pending para aprovação e senha vazia
    users[racf] = {
        "funcional": reg_info["funcional"],
        "departamento": reg_info["departamento"],
        "motivo": reg_info["motivo"],
        "status": "pending",
        "password": ""
    }
    save_users(users)

def save_log(username, log_data):
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
    # Formata o timestamp sem ":" para evitar erro no Windows
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{LOGS_FOLDER}{username}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(log_data)

def load_last_log(username):
    # Para simplificar, não implementamos a listagem e recuperação do último log.
    return None

def download_metadata():
    try:
        with open('data/metadata/metadata.yaml', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "Metadados não disponíveis."

# Função auxiliar para obter uma curiosidade (usando RandomCuriosityAgent do agent.py)
def get_curiosity():
    from agent import RandomCuriosityAgent
    agent = RandomCuriosityAgent()
    return agent.get_curiosity()

# ----- Função Principal do Painel -----
def main():
    st.markdown("<h1 style='text-align: center;'>Painel do Agente de Negócios</h1>", unsafe_allow_html=True)
    
    # Centraliza a logo usando st.image dentro de um container HTML
    logo_path = "templates/logos/logo.png"
    if os.path.exists(logo_path):
        st.markdown(
            f"<div style='display: flex; justify-content: center;'><img src='{logo_path}' width='300'></div>",
            unsafe_allow_html=True
        )
    else:
        st.error("Logo não encontrada no caminho: " + logo_path)
    
    # Se o usuário não estiver autenticado, exibe a área de Login/Cadastro com switch
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        auth_mode = st.radio("Selecione a opção", ("Login", "Cadastro"), index=0)
        with st.container():
            if auth_mode == "Login":
                st.subheader("Login")
                racf = st.text_input("RACF (Username)")
                password = st.text_input("Senha", type="password")
                if st.button("Entrar"):
                    users = load_users()
                    if racf in users:
                        user_info = users[racf]
                        if user_info.get("password") == password and user_info.get("status") == "active":
                            st.session_state.logged_in = True
                            st.session_state.username = racf
                            st.success("Login efetuado com sucesso!")
                        elif user_info.get("status") != "active":
                            st.error("Seu cadastro ainda está pendente aprovação ou foi desativado.")
                        else:
                            st.error("Senha incorreta.")
                    else:
                        st.error("Usuário não encontrado. Caso seja novo, utilize a opção Cadastro.")
            else:
                st.subheader("Cadastro")
                reg_racf = st.text_input("Novo RACF (Username)", key="reg_racf")
                reg_funcional = st.text_input("Funcional", key="reg_funcional")
                reg_departamento = st.text_input("Departamento", key="reg_departamento")
                reg_motivo = st.text_input("Motivo do acesso", key="reg_motivo")
                if st.button("Cadastrar"):
                    if reg_racf and reg_funcional and reg_departamento and reg_motivo:
                        reg_info = {
                            "racf": reg_racf,
                            "funcional": reg_funcional,
                            "departamento": reg_departamento,
                            "motivo": reg_motivo
                        }
                        register_user(reg_info)
                        st.info("Já notificamos o nosso time, em breve seu usuário será criado.")
                    else:
                        st.error("Preencha todos os campos obrigatórios.")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 0.9em;'>"
            "Disclaimer: As respostas fornecidas pelo Agente Biônico são geradas automaticamente e podem conter erros. "
            "Utilize essas informações apenas como apoio, e sempre verifique com um especialista antes de tomar decisões críticas."
            "</p>", unsafe_allow_html=True
        )
        return

    # Conteúdo do painel para usuário autenticado
    st.markdown(f"<h3 style='text-align: center;'>Bem-vindo, {st.session_state.username}</h3>", unsafe_allow_html=True)
    
    # Área de curiosidades (atualizada a cada 12 segundos)
    st_autorefresh(interval=12000, limit=None, key="curiosity_refresh")
    curiosity_text = get_curiosity()
    st.markdown(f"<p style='text-align: center; font-style: italic; color: gray;'>Curiosidade: {curiosity_text}</p>", unsafe_allow_html=True)
    
    # Sidebar com instruções e metadados
    st.sidebar.subheader("Instruções de Uso")
    st.sidebar.write("Utilize o painel para interagir com o agente de negócios. Consulte os metadados para entender os dados disponíveis.")
    st.sidebar.write("Dados disponíveis desde 2020 até 2023")
    if st.sidebar.button("Baixar Metadados"):
        metadata = download_metadata()
        st.sidebar.download_button("Download Metadata", data=metadata, file_name="metadata.yaml", mime="text/yaml")
    
    # Recupera ou inicializa o histórico de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Container para exibição do chat
    st.markdown("<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if "user" in msg:
            st.markdown(f"<p style='text-align: right; background-color: #dcf8c6; padding: 5px; border-radius: 5px;'>Você: {msg['user']}</p>", unsafe_allow_html=True)
        elif "agent" in msg:
            st.markdown(f"<p style='text-align: left; background-color: #f1f0f0; padding: 5px; border-radius: 5px;'>Agente: {msg['agent']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Área para digitar nova mensagem (chat input)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Digite sua mensagem:")
        submit_button = st.form_submit_button(label="Enviar")
    
    if submit_button and user_input:
        st.session_state.chat_history.append({"user": user_input})
        if "agent" not in st.session_state:
            st.session_state.agent = AdvancedAgent()
        result = st.session_state.agent.run({"context": user_input})
        agent_response = result.get("response", "Nenhuma resposta gerada.")
        st.session_state.chat_history.append({"agent": agent_response})
        # Exibe também a query gerada para análise
        st.write("Query gerada:", result.get("query", ""))
        log_data = "\n".join(
            [f"Usuário: {msg.get('user', '')}\nAgente: {msg.get('agent', '')}"
             for msg in st.session_state.chat_history]
        )
        save_log(st.session_state.username, log_data)
    
    # Botão para baixar o histórico da conversa
    if st.button("Baixar Histórico"):
        history_str = "\n".join(
            [f"Você: {msg.get('user','')}\nAgente: {msg.get('agent','')}"
             for msg in st.session_state.chat_history]
        )
        st.download_button("Download Histórico", data=history_str, file_name="historico.txt", mime="text/plain")
    
    # Seção de Feedback
    st.markdown("<h4>Feedback</h4>", unsafe_allow_html=True)
    feedback = st.text_area("Deixe um comentário (opcional)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍"):
            st.success("Obrigado pelo feedback!")
            feedback_log = f"{st.session_state.username} deu feedback positivo: {feedback}"
            save_log(st.session_state.username, feedback_log)
    with col2:
        if st.button("👎"):
            st.info("Obrigado, vamos melhorar!")
            feedback_log = f"{st.session_state.username} deu feedback negativo: {feedback}"
            save_log(st.session_state.username, feedback_log)
    
    # Disclaimer final
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 0.9em;'>"
        "Disclaimer: As respostas fornecidas pelo Agente Biônico são geradas automaticamente e podem conter erros. "
        "Utilize essas informações apenas como apoio, e sempre verifique com um especialista antes de tomar decisões críticas."
        "</p>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
