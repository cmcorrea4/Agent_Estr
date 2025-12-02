import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
import os
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de gestión energética--ESTRA",
    page_icon="🏭",
    layout="wide"
)

# Título principal
st.title("🏭 Diagnóstico de gestión energética--ESTRA")
st.markdown("**Obtén datos del sistema energético y analízalos con IA avanzada**")

# Función para consultar el endpoint de energía
@st.cache_data(ttl=300)
def consultar_endpoint_energia(username, password):
    """Consulta el endpoint de energía y retorna los datos en formato JSON"""
    try:
        url = "https://energy-api-628964750053.us-east1.run.app/test-summary"
        
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'User-Agent': 'StreamlitApp/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                return data, None
            except json.JSONDecodeError as e:
                return None, f"Error parseando JSON: {str(e)}"
        else:
            return None, f"Error HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return None, "Timeout: El servidor tardó demasiado en responder"
    except requests.exceptions.ConnectionError:
        return None, "Error de conexión: No se pudo conectar al servidor"
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"

# Función para convertir JSON a DataFrame
def json_to_dataframe(json_data):
    """Convierte los datos JSON del endpoint a un DataFrame de pandas"""
    try:
        if isinstance(json_data, dict):
            df = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        else:
            df = pd.DataFrame({'data': [json_data]})
        
        return df, None
    except Exception as e:
        return None, f"Error convirtiendo JSON a DataFrame: {str(e)}"

# Función para mostrar información del DataFrame
def mostrar_info_dataframe(df):
    """Muestra información básica del DataFrame"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Filas", df.shape[0])
    with col2:
        st.metric("📊 Columnas", df.shape[1])
    with col3:
        st.metric("💾 Tamaño (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
    with col4:
        st.metric("🔢 Valores No Nulos", df.count().sum())

# Inicializar session state
if 'chat_history_energia' not in st.session_state:
    st.session_state.chat_history_energia = []

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    st.subheader("🔌 Configuración del Endpoint")
    
    if "df_energia" not in st.session_state:
        api_username = st.text_input(
            "👤 Usuario del Endpoint:",
            placeholder="Ingresa tu usuario",
            help="Usuario para autenticación del endpoint de energía"
        )
        
        api_password = st.text_input(
            "🔒 Contraseña del Endpoint:",
            type="password",
            placeholder="Ingresa tu contraseña",
            help="Contraseña para autenticación del endpoint"
        )
        
        endpoint_configured = bool(api_username and api_password)
        
        if endpoint_configured:
            st.success("✅ Credenciales configuradas")
        else:
            st.warning("⚠️ Ingresa usuario y contraseña")
    else:
        api_username = st.session_state.get('api_username', '')
        api_password = st.session_state.get('api_password', '')
        endpoint_configured = True
        st.success("✅ Sesión activa")
    
    st.markdown("---")
    
    st.subheader("🤖 Configuración de OpenAI")
    
    openai_api_key = st.text_input(
        "🔑 API Key de OpenAI:",
        type="password",
        placeholder="sk-...",
        value=st.session_state.get('openai_api_key', ''),
        help="Ingresa tu API Key de OpenAI"
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ Ingresa tu API Key")
    
    model_name = "gpt-4"
    temperature = 0.1
    
    st.markdown("---")
    
    # Botón obtener datos
    if st.button("🔌 Obtener Datos del Sistema", use_container_width=True, disabled=not endpoint_configured):
        with st.spinner("Consultando endpoint..."):
            datos_json, error = consultar_endpoint_energia(api_username, api_password)
            
            if datos_json is not None:
                df_energia_temp, error_df = json_to_dataframe(datos_json)
                
                if df_energia_temp is not None:
                    st.session_state.df_energia = df_energia_temp
                    st.session_state.datos_json = datos_json
                    st.session_state.api_username = api_username
                    st.session_state.api_password = api_password
                    st.success("✅ Datos cargados")
                else:
                    st.error(f"❌ {error_df}")
            else:
                st.error(f"❌ {error}")
    
    # Estado
    if "df_energia" in st.session_state:
        st.success("🟢 Datos listos")
        st.info(f"📊 {st.session_state.df_energia.shape[0]} filas, {st.session_state.df_energia.shape[1]} columnas")
    else:
        st.warning("🔴 Sin datos")

# Contenido principal
if "df_energia" not in st.session_state:
    st.info("👆 Configura las credenciales y haz clic en 'Obtener Datos del Sistema' para comenzar")
    
    st.markdown("---")
    st.subheader("ℹ️ Sobre esta aplicación")
    st.markdown("""
    Esta aplicación integra dos funcionalidades principales:
    
    1. **🔌 Obtención de datos**: Consulta el endpoint de energía de ESTRA
    2. **🤖 Análisis con IA**: Procesa los datos usando un agente.
    
    **Funcionalidades:**
    - Conexión automática al sistema de energía ESTRA
    - Conversión de JSON a DataFrame de pandas
    - Análisis inteligente con preguntas en lenguaje natural
    - Visualizaciones automáticas
    - Estadísticas descriptivas
    """)

else:
    df_energia = st.session_state.df_energia
    datos_json = st.session_state.datos_json
    
    st.success("✅ Datos del sistema energético cargados")
    
    st.header("📊 Información del Dataset")
    mostrar_info_dataframe(df_energia)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas", "🗂️ JSON Original"])
    
    with tab1:
        st.subheader("Vista de los Datos")
        st.dataframe(df_energia, use_container_width=True)
    
    with tab2:
        st.subheader("Información del Dataset")
        if not df_energia.empty:
            info_df = pd.DataFrame({
                'Columna': df_energia.columns,
                'Tipo': df_energia.dtypes.astype(str),
                'No Nulos': df_energia.count(),
                'Nulos': df_energia.isnull().sum(),
                '% Nulos': (df_energia.isnull().sum() / len(df_energia) * 100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)
        else:
            st.warning("DataFrame vacío")
    
    with tab3:
        st.subheader("Estadísticas Descriptivas")
        numeric_df = df_energia.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No hay columnas numéricas.")
            text_df = df_energia.select_dtypes(include=['object'])
            if not text_df.empty:
                for col in text_df.columns:
                    st.write(f"• **{col}**: {df_energia[col].nunique()} valores únicos")
    
    with tab4:
        st.subheader("Datos JSON Originales")
        st.json(datos_json)
    
    # Agente de Análisis IA
    st.header("🤖 Agente de Análisis IA")
    
    if not st.session_state.get('openai_api_key'):
        st.warning("⚠️ Configura tu API Key de OpenAI en la barra lateral.")
    else:
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=st.session_state.openai_api_key
            )
            
            agent = create_pandas_dataframe_agent(
                llm,
                df_energia,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            
            st.success("🎯 Agente IA inicializado")
            
            st.subheader("💡 Ejemplos de preguntas:")
            examples = [
                "¿Qué información contiene el dataset?",
                "¿Cuáles son las columnas disponibles?",
                "¿Qué Moldes tienen la mayor productividad efectiva?",
                "¿En qué fechas se trabajó el model 15252?",
                "¿Cuál Molde tiene mayor SECn?",
            ]
            
            for i, example in enumerate(examples, 1):
                st.write(f"{i}. {example}")
            
            st.subheader("❓ Consulta los datos con IA")
            
            # Usar form para evitar reruns
            with st.form(key="chat_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Escribe tu pregunta:",
                    placeholder="Ej: ¿Cuál es el tiempo de producción total vs efectivo?"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    submit_button = st.form_submit_button("🚀 Analizar", type="primary")
                with col2:
                    pass
            
            # Botón limpiar fuera del form
            if st.button("🗑️ Limpiar historial"):
                st.session_state.chat_history_energia = []
            
            # Procesar pregunta
            if submit_button and user_question:
                with st.spinner("🔄 El agente está analizando..."):
                    try:
                        response = agent.invoke({"input": user_question})
                        
                        st.session_state.chat_history_energia.append({
                            "question": user_question,
                            "answer": response["output"]
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Intenta reformular tu pregunta.")
            
            # Mostrar historial
            if st.session_state.chat_history_energia:
                st.subheader("💬 Análisis Realizados")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history_energia)):
                    with st.expander(
                        f"❓ {chat['question'][:50]}..." if len(chat['question']) > 50 
                        else f"❓ {chat['question']}", 
                        expanded=(i==0)
                    ):
                        st.write("**Pregunta:**")
                        st.write(chat['question'])
                        st.write("**Análisis del Agente IA:**")
                        st.write(chat['answer'])
                        st.divider()
            
        except Exception as e:
            st.error(f"❌ Error al inicializar el agente: {str(e)}")
            st.info("Verifica que tu API key sea válida.")
    
    st.markdown("---")
    if st.button("🔄 Actualizar Datos del Sistema", use_container_width=True):
        consultar_endpoint_energia.clear()
        for key in ['df_energia', 'datos_json', 'api_username', 'api_password']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.chat_history_energia = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    🏭 ESTRA - Sistema Integrado de Análisis Energético con IA | Powered by SUME & SOSPOL
    </div>
    """, 
    unsafe_allow_html=True
)
