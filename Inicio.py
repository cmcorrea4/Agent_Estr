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
    page_title="ESTRA - Análisis Energético con IA",
    page_icon="🏭",
    layout="wide"
)

# Título principal
st.title("🏭 ESTRA - Análisis Inteligente de Datos Energéticos")
st.markdown("**Obtén datos del sistema energético y analízalos con IA avanzada**")

# Función para consultar el endpoint de energía
@st.cache_data(ttl=300)  # Cache por 5 minutos
def consultar_endpoint_energia(url, username, password):
    """Consulta el endpoint de energía y retorna los datos en formato JSON"""
    try:
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

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # Configuración del Endpoint API
    st.subheader("🔌 Configuración del Endpoint")
    
    # URL fija del endpoint
    api_url = "https://energy-api-628964750053.us-east1.run.app/test-summary"
    
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
    
    # Validar que todos los campos estén completos
    endpoint_configured = bool(api_username and api_password)
    
    if endpoint_configured:
        st.success("✅ Credenciales del endpoint configuradas")
    else:
        st.warning("⚠️ Ingresa usuario y contraseña del endpoint")
    
    st.markdown("---")
    
    # Configuración de OpenAI
    st.subheader("🤖 Configuración OpenAI")
    
    openai_api_key = st.text_input(
        "🔑 API Key de OpenAI:",
        type="password",
        placeholder="sk-...",
        help="Ingresa tu API key de OpenAI para usar el agente inteligente"
    )
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ API Key requerida para el agente IA")
    
    # Configuración del modelo
    model_name = st.selectbox(
        "🤖 Modelo OpenAI:",
        ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
        index=0
    )
    
    temperature = st.slider(
        "🌡️ Temperatura:",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Controla la creatividad (0 = preciso, 1 = creativo)"
    )
    
    st.markdown("---")
    
    # Botón para obtener datos del endpoint
    if st.button("🔌 Obtener Datos del Sistema", use_container_width=True, disabled=not endpoint_configured):
        with st.spinner("Consultando endpoint de energía..."):
            datos_json, error = consultar_endpoint_energia(api_url, api_username, api_password)
            
            if datos_json is not None:
                st.success("✅ Datos obtenidos del sistema")
                
                # Convertir JSON a DataFrame
                df_energia, error_df = json_to_dataframe(datos_json)
                
                if df_energia is not None:
                    st.success("✅ DataFrame creado exitosamente")
                    # Guardar en session state
                    st.session_state.df_energia = df_energia
                    st.session_state.datos_json = datos_json
                    st.rerun()
                else:
                    st.error(f"❌ Error creando DataFrame: {error_df}")
            else:
                st.error(f"❌ Error obteniendo datos: {error}")
    
    # Estado de la conexión
    if "df_energia" in st.session_state:
        st.success("🟢 Datos cargados y listos")
        st.info(f"📊 DataFrame: {st.session_state.df_energia.shape[0]} filas, {st.session_state.df_energia.shape[1]} columnas")
    else:
        st.warning("🔴 Sin datos del sistema")

# Contenido principal
if "df_energia" not in st.session_state:
    st.info("👆 Configura el endpoint y
