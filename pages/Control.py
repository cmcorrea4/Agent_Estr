import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="ESTRA CUSUM - Análisis con IA",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 ESTRA CUSUM - Análisis con IA")
st.markdown("**Obtén datos CUSUM y analízalos con IA avanzada**")

# Función para consultar el endpoint CUSUM
@st.cache_data(ttl=300)  # Cache por 5 minutos
def consultar_endpoint_cusum():
    """Consulta el endpoint CUSUM y retorna los datos en formato JSON"""
    try:
        url = "https://energy-api-628964750053.us-east1.run.app/test-cusum"
        
        # Usar credenciales desde secrets
        username = st.secrets["settings"]["API_USERNAME"]   
        password = st.secrets["settings"]["API_PASSWORD"]   
        
        # Crear credenciales de autenticación
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        # Configurar headers
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'User-Agent': 'StreamlitApp/1.0',
            'Accept': 'application/json'
        }
        
        # Realizar la petición
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                return data, None  # datos, error
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
            # Si es un diccionario, convertir a DataFrame con una fila
            df = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            # Si es una lista, convertir directamente
            df = pd.DataFrame(json_data)
        else:
            # Si es otro tipo, intentar convertir
            df = pd.DataFrame({'data': [json_data]})
        
        return df, None
    except Exception as e:
        return None, f"Error convirtiendo JSON a DataFrame: {str(e)}"

# Función para mostrar información básica del DataFrame
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

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # Configuración de OpenAI
    st.subheader("🤖 Configuración OpenAI")
    openai_api_key = st.secrets["settings"]["OPENAI_API_KEY"] 
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ API Key requerida para el agente IA")
    
    # Configuración del modelo
    model_name = "gpt-4"
    temperature = 0.1
    st.markdown("---")
    
    # Botón para obtener datos del endpoint
    if st.button("📊 Obtener Datos CUSUM", use_container_width=True):
        with st.spinner("Consultando endpoint CUSUM..."):
            datos_json, error = consultar_endpoint_cusum()
            
            if datos_json is not None:
                st.success("✅ Datos CUSUM obtenidos")
                
                # Convertir JSON a DataFrame
                df_cusum, error_df = json_to_dataframe(datos_json)
                
                if df_cusum is not None:
                    st.success("✅ DataFrame creado exitosamente")
                    # Guardar en session state
                    st.session_state.df_cusum = df_cusum
                    st.session_state.datos_json_cusum = datos_json
                    st.rerun()
                else:
                    st.error(f"❌ Error creando DataFrame: {error_df}")
            else:
                st.error(f"❌ Error obteniendo datos: {error}")
    
    # Estado de la conexión
    if "df_cusum" in st.session_state:
        st.success("🟢 Datos cargados y listos")
        st.info(f"📊 DataFrame: {st.session_state.df_cusum.shape[0]} filas, {st.session_state.df_cusum.shape[1]} columnas")
    else:
        st.warning("🔴 Sin datos CUSUM")

# Contenido principal
if "df_cusum" not in st.session_state:
    st.info("👆 Haz clic en 'Obtener Datos CUSUM' en la barra lateral para comenzar")
    
    # Información sobre la aplicación
    st.markdown("---")
    st.subheader("ℹ️ Sobre esta aplicación")
    st.markdown("""
    Esta aplicación integra dos funcionalidades principales:
    
    1. **📊 Obtención de datos CUSUM**: Consulta el endpoint de control de calidad
    2. **🤖 Análisis con IA**: Procesa los datos usando un agente inteligente de pandas
    
    **Funcionalidades:**
    - Conexión automática al endpoint CUSUM
    - Conversión de JSON a DataFrame de pandas
    - Análisis inteligente con preguntas en lenguaje natural
    - Estadísticas descriptivas
    """)

else:
    # Mostrar los datos obtenidos
    df_cusum = st.session_state.df_cusum
    datos_json_cusum = st.session_state.datos_json_cusum
    
    st.success("✅ Datos CUSUM cargados exitosamente")
    
    # Mostrar información básica del DataFrame
    st.header("📊 Información del Dataset CUSUM")
    mostrar_info_dataframe(df_cusum)
    
    # Tabs para diferentes vistas de los datos
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas", "🗂️ JSON Original"])
    
    with tab1:
        st.subheader("Vista de los Datos")
        st.dataframe(df_cusum, use_container_width=True)
    
    with tab2:
        st.subheader("Información del Dataset")
        if not df_cusum.empty:
            info_df = pd.DataFrame({
                'Columna': df_cusum.columns,
                'Tipo': df_cusum.dtypes.astype(str),
                'No Nulos': df_cusum.count(),
                'Nulos': df_cusum.isnull().sum(),
                '% Nulos': (df_cusum.isnull().sum() / len(df_cusum) * 100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)
        else:
            st.warning("DataFrame vacío")
    
    with tab3:
        st.subheader("Estadísticas Descriptivas")
        numeric_df = df_cusum.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No hay columnas numéricas para estadísticas descriptivas.")
            st.write("**Resumen de columnas de texto:**")
            text_df = df_cusum.select_dtypes(include=['object'])
            if not text_df.empty:
                for col in text_df.columns:
                    unique_vals = df_cusum[col].nunique()
                    st.write(f"• **{col}**: {unique_vals} valores únicos")
    
    with tab4:
        st.subheader("Datos JSON Originales")
        st.json(datos_json_cusum)
    
    # Agente de Análisis IA
    st.header("🤖 Agente de Análisis IA")
    
    if not openai_api_key:
        st.warning("⚠️ Configura tu API Key de OpenAI en la barra lateral para usar el agente inteligente.")
    else:
        try:
            # Inicializar el modelo de OpenAI
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
            
            # Crear el agente de pandas
            agent = create_pandas_dataframe_agent(
                llm,
                df_cusum,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            
            st.success("🎯 Agente IA inicializado correctamente")
            
            # Ejemplos específicos para datos CUSUM
            st.subheader("💡 Ejemplos de preguntas sobre datos CUSUM:")
            examples = [
                "¿Qué información contiene el dataset?",
                "¿Cuáles son las columnas disponibles?",
                "¿Cuáles son las estadísticas principales de los datos?",
                "¿Hay valores atípicos en los datos?",
                "¿Qué periodo cubren los datos (fechas)?",
                "¿Cuáles son los valores máximos y mínimos?",
                "¿Existe algún patrón en los datos?",
                "¿Hay datos faltantes?"
            ]
            
            for i, example in enumerate(examples, 1):
                st.write(f"{i}. {example}")
            
            # Interface para hacer preguntas
            st.subheader("❓ Consulta los datos con IA")
            
            # Historial de conversación
            if 'chat_history_cusum' not in st.session_state:
                st.session_state.chat_history_cusum = []
            
            # Campo de entrada para la pregunta
            user_question = st.text_input(
                "Escribe tu pregunta sobre los datos CUSUM:",
                placeholder="Ej: ¿Cuáles son las principales características de estos datos?",
                key="user_input_cusum"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🚀 Analizar", type="primary")
            with col2:
                clear_button = st.button("🗑️ Limpiar historial")
            
            if clear_button:
                st.session_state.chat_history_cusum = []
                st.rerun()
            
            if ask_button and user_question:
                with st.spinner("🔄 El agente está analizando los datos CUSUM..."):
                    try:
                        # Ejecutar la pregunta con el agente
                        response = agent.invoke({"input": user_question})
                        
                        # Agregar al historial
                        st.session_state.chat_history_cusum.append({
                            "question": user_question,
                            "answer": response["output"]
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                        st.info("💡 Intenta reformular tu pregunta o verifica la sintaxis.")
            
            # Mostrar historial de conversación
            if st.session_state.chat_history_cusum:
                st.subheader("💬 Análisis Realizados")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history_cusum)):
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
            st.info("Verifica que tu API key de OpenAI sea válida y tenga créditos disponibles.")
    
    # Botón para actualizar datos
    st.markdown("---")
    if st.button("🔄 Actualizar Datos CUSUM", use_container_width=True):
        # Limpiar cache y session state
        consultar_endpoint_cusum.clear()
        if "df_cusum" in st.session_state:
            del st.session_state.df_cusum
        if "datos_json_cusum" in st.session_state:
            del st.session_state.datos_json_cusum
        if "chat_history_cusum" in st.session_state:
            st.session_state.chat_history_cusum = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    📊 ESTRA CUSUM - Sistema de Análisis de Control de Calidad con IA | Powered by SUME & SOSPOL
    </div>
    """, 
    unsafe_allow_html=True
)
