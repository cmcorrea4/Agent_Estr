import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
import os
from datetime import datetime, timedelta
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

# Función para consultar el endpoint de energía con filtros de fecha
@st.cache_data(ttl=300)  # Cache por 5 minutos
def consultar_endpoint_energia(username, password, date_start=None, date_end=None):
    """Consulta el endpoint de energía y retorna los datos en formato JSON"""
    try:
        # URL base
        url = "https://energy-api-628964750053.us-east1.run.app/test-summary"
        
        # Agregar parámetros de fecha si están disponibles
        params = {}
        if date_start:
            params['dateStart'] = date_start
        if date_end:
            params['dateEnd'] = date_end
        
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
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
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
    """Convierte los datos JSON del endpoint a un DataFrame de pandas aplanado"""
    try:
        # 1. Caso ideal: El JSON tiene una llave 'data' que contiene la lista de registros
        if isinstance(json_data, dict) and 'data' in json_data and isinstance(json_data['data'], list):
            # json_normalize aplana automáticamente diccionarios anidados
            df = pd.json_normalize(json_data['data'])
            
        # 2. Caso donde el JSON es directamente una lista de diccionarios
        elif isinstance(json_data, list):
            df = pd.json_normalize(json_data)
            
        # 3. Caso donde es un solo diccionario sin la llave 'data'
        elif isinstance(json_data, dict):
            df = pd.json_normalize([json_data])
            
        # 4. Otros formatos
        else:
            df = pd.DataFrame({'datos': [json_data]})
        
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

# Función para obtener el inicio de la semana (lunes)
def get_week_start(date):
    """Retorna el inicio de la semana (lunes) para una fecha dada"""
    return date - timedelta(days=date.weekday())

# Función para obtener el fin de la semana (domingo)
def get_week_end(date):
    """Retorna el fin de la semana (domingo) para una fecha dada"""
    return date + timedelta(days=6 - date.weekday())

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # Configuración del Endpoint API
    st.subheader("🔌 Configuración del Endpoint")
    
    # Solo mostrar campos si no hay datos cargados
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
        
        # Validar que todos los campos estén completos
        endpoint_configured = bool(api_username and api_password)
        
        if endpoint_configured:
            st.success("✅ Credenciales del endpoint configuradas")
        else:
            st.warning("⚠️ Ingresa usuario y contraseña del endpoint")
    else:
        # Si ya hay datos cargados, usar las credenciales guardadas
        api_username = st.session_state.get('api_username', '')
        api_password = st.session_state.get('api_password', '')
        endpoint_configured = True
        st.success("✅ Sesión activa")
    
    st.markdown("---")
    
    # Filtros de fecha
    st.subheader("📅 Filtro de Fechas")
    
    # Selector de tipo de filtro
    filter_type = st.radio(
        "Tipo de filtro:",
        ["Por semana", "Por rango de fechas"],
        help="Selecciona cómo deseas filtrar los datos"
    )
    
    if filter_type == "Por semana":
        # Obtener la semana actual por defecto
        today = datetime.now().date()
        default_week_start = get_week_start(today)
        
        # Selector de semana
        selected_week = st.date_input(
            "Selecciona una fecha (se usará su semana completa):",
            value=st.session_state.get('selected_week', today),
            help="Selecciona cualquier día de la semana que deseas consultar"
        )
        
        # Calcular inicio y fin de la semana
        date_start = get_week_start(selected_week)
        date_end = get_week_end(selected_week)
        
        # Mostrar información de la semana
        st.info(f"📅 Semana del **{date_start.strftime('%d/%m/%Y')}** al **{date_end.strftime('%d/%m/%Y')}**")
        st.write(f"🗓️ {7} días")
        
        dates_valid = True
        
    else:  # Por rango de fechas
        # Obtener fechas guardadas o usar valores por defecto
        default_start = st.session_state.get('date_start', datetime(2024, 1, 1).date())
        default_end = st.session_state.get('date_end', datetime.now().date())
        
        date_start = st.date_input(
            "Fecha de inicio:",
            value=default_start,
            help="Selecciona la fecha inicial del rango"
        )
        
        date_end = st.date_input(
            "Fecha de fin:",
            value=default_end,
            help="Selecciona la fecha final del rango"
        )
        
        # Validar que la fecha de inicio sea menor que la de fin
        if date_start > date_end:
            st.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin")
            dates_valid = False
        else:
            dates_valid = True
            dias = (date_end - date_start).days + 1
            st.info(f"📊 Rango: {dias} días")
    
    st.markdown("---")
    
    # Configuración de OpenAI API Key
    st.subheader("🤖 Configuración de OpenAI")
    
    if "openai_api_key" not in st.session_state:
        openai_api_key = st.text_input(
            "🔑 API Key de OpenAI:",
            type="password",
            placeholder="sk-...",
            help="Ingresa tu API Key de OpenAI para usar el agente inteligente"
        )
        
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API Key configurada")
        else:
            st.warning("⚠️ Ingresa tu API Key de OpenAI")
    else:
        openai_api_key = st.session_state.openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("✅ API Key configurada")
        
        # Botón para cambiar API Key
        if st.button("🔄 Cambiar API Key"):
            del st.session_state.openai_api_key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            st.rerun()
    
    # Configuración del modelo (fija, sin mostrar)
    model_name = "gpt-4o-mini"
    temperature = 0.1
    
    st.markdown("---")
    
    # Botón para obtener datos del endpoint
    if st.button("🔌 Obtener Datos del Sistema", width="stretch", 
                 disabled=not (endpoint_configured and dates_valid)):
        with st.spinner("Consultando endpoint de energía..."):
            # Convertir fechas a formato string YYYY-MM-DD
            date_start_str = date_start.strftime('%Y-%m-%d')
            date_end_str = date_end.strftime('%Y-%m-%d')
            
            datos_json, error = consultar_endpoint_energia(
                api_username, 
                api_password, 
                date_start_str, 
                date_end_str
            )
            
            if datos_json is not None:
                st.success("✅ Datos obtenidos del sistema")
                
                # Convertir JSON a DataFrame
                df_energia, error_df = json_to_dataframe(datos_json)
                
                if df_energia is not None:
                    st.success("✅ DataFrame creado exitosamente")
                    # Guardar en session state
                    st.session_state.df_energia = df_energia
                    st.session_state.datos_json = datos_json
                    # Guardar credenciales y fechas para no pedirlas de nuevo
                    st.session_state.api_username = api_username
                    st.session_state.api_password = api_password
                    st.session_state.date_start = date_start
                    st.session_state.date_end = date_end
                    st.session_state.filter_type = filter_type
                    if filter_type == "Por semana":
                        st.session_state.selected_week = selected_week
                    st.rerun()
                else:
                    st.error(f"❌ Error creando DataFrame: {error_df}")
            else:
                st.error(f"❌ Error obteniendo datos: {error}")
    
    # Estado de la conexión
    if "df_energia" in st.session_state:
        st.success("🟢 Datos cargados y listos")
        st.info(f"📊 DataFrame: {st.session_state.df_energia.shape[0]} filas, {st.session_state.df_energia.shape[1]} columnas")
        
        # Mostrar rango de fechas actual
        if 'date_start' in st.session_state and 'date_end' in st.session_state:
            if st.session_state.get('filter_type') == "Por semana":
                st.info(f"📅 Semana: {st.session_state.date_start.strftime('%d/%m/%Y')} - {st.session_state.date_end.strftime('%d/%m/%Y')}")
            else:
                st.info(f"📅 Período: {st.session_state.date_start.strftime('%d/%m/%Y')} - {st.session_state.date_end.strftime('%d/%m/%Y')}")
    else:
        st.warning("🔴 Sin datos del sistema")

# Contenido principal
if "df_energia" not in st.session_state:
    st.info("👆 Configura las credenciales, selecciona el filtro de fechas y haz clic en 'Obtener Datos del Sistema' en la barra lateral para comenzar")
    
    # Información sobre la aplicación
    st.markdown("---")
    st.subheader("ℹ️ Sobre esta aplicación")
    st.markdown("""
    Esta aplicación integra dos funcionalidades principales:
    
    1. **🔌 Obtención de datos**: Consulta el endpoint de energía de ESTRA con filtros de fecha
    2. **🤖 Análisis con IA**: Procesa los datos usando un agente.
    
    **Funcionalidades:**
    - Conexión automática al sistema de energía ESTRA
    - Filtrado por semana completa (lunes a domingo)
    - Filtrado por rango de fechas personalizado
    - Conversión de JSON a DataFrame de pandas
    - Análisis inteligente con preguntas en lenguaje natural
    - Visualizaciones automáticas
    - Estadísticas descriptivas
    """)

else:
    # Mostrar los datos obtenidos
    df_energia = st.session_state.df_energia
    datos_json = st.session_state.datos_json
    
    st.success("✅ Datos del sistema energético cargados exitosamente")
    
    # Mostrar el resumen que viene de la API si existe
    if isinstance(datos_json, dict) and 'chatbotSummary' in datos_json:
        st.info(datos_json['chatbotSummary'])
    
    # Mostrar rango de fechas de los datos cargados
    if 'date_start' in st.session_state and 'date_end' in st.session_state:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"📅 Desde: **{st.session_state.date_start.strftime('%d/%m/%Y')}**")
        with col2:
            st.info(f"📅 Hasta: **{st.session_state.date_end.strftime('%d/%m/%Y')}**")
        with col3:
            dias = (st.session_state.date_end - st.session_state.date_start).days + 1
            st.metric("📊 Días", dias)
    
    # Mostrar información básica del DataFrame
    st.header("📊 Información del Dataset")
    mostrar_info_dataframe(df_energia)
    
    # Tabs para diferentes vistas de los datos
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas", "🗂️ JSON Original"])
    
    with tab1:
        st.subheader("Vista de los Datos")
        st.dataframe(df_energia, width="stretch")
    
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
            st.dataframe(info_df, width="stretch")
        else:
            st.warning("DataFrame vacío")
    
    with tab3:
        st.subheader("Estadísticas Descriptivas")
        numeric_df = df_energia.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), width="stretch")
        else:
            st.info("No hay columnas numéricas para estadísticas descriptivas.")
        
        st.write("**Resumen de columnas de texto:**")
        text_df = df_energia.select_dtypes(include=['object'])
        if not text_df.empty:
            for col in text_df.columns:
                try:
                    unique_vals = df_energia[col].nunique()
                    st.write(f"• **{col}**: {unique_vals} valores únicos")
                except TypeError:
                    unique_vals = df_energia[col].astype(str).nunique()
                    st.write(f"• **{col}**: {unique_vals} valores únicos (datos anidados)")
    
    with tab4:
        st.subheader("Datos JSON Originales")
        st.json(datos_json)
    
    # Agente de Análisis IA
    st.header("🤖 Agente de Análisis IA")

    system_prompt = """
    Eres un analista experto en datos energéticos industriales.
    Responde SIEMPRE en español.
    Usa lenguaje técnico claro y adecuado para ingenieros.
    Nunca respondas en inglés, responde simepe con datos .
    """
    
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.warning("⚠️ Configura tu API Key de OpenAI en la barra lateral para usar el agente inteligente.")
    else:
        try:
            # Inicializar el modelo de OpenAI
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=st.session_state.openai_api_key
            )
            
            # Crear el agente de pandas
            agent = create_pandas_dataframe_agent(
                llm,
                df_energia,
                verbose=True,
                allow_dangerous_code=True,
                prefix=system_prompt
            )
            
            st.success("🎯 Agente IA inicializado correctamente")
            
            # Ejemplos específicos para datos energéticos
            st.subheader("💡 Ejemplos de preguntas sobre datos energéticos:")
            examples = [
                "¿Qué información contiene el dataset?",
                "¿Cuáles son las columnas disponibles?",
                "¿Que Moldes tienen la mayor productividad efectiva?",
                "¿En que fechas se trabajó el model 15252?",
                "¿Cuál Molde tiene mayor SECn?",
                "¿Qué periodo cubren los datos (fechas)?",
                "Cuáles son los mayores porcentaje de tiempo de paro y a que referencias corresponden?"
            ]
            
            for i, example in enumerate(examples, 1):
                st.write(f"{i}. {example}")
            
            # Interface para hacer preguntas
            st.subheader("❓ Consulta los datos con IA")
            
            # Historial de conversación
            if 'chat_history_energia' not in st.session_state:
                st.session_state.chat_history_energia = []
            
            # Campo de entrada para la pregunta
            user_question = st.text_input(
                "Escribe tu pregunta sobre los datos energéticos:",
                placeholder="Ej: ¿Cuál es el tiempo de producción total vs efectivo?",
                key="user_input_energia"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🚀 Analizar", type="primary")
            with col2:
                clear_button = st.button("🗑️ Limpiar historial")
            
            if clear_button:
                st.session_state.chat_history_energia = []
                st.rerun()
            
            if ask_button and user_question:
                with st.spinner("🔄 El agente está analizando los datos energéticos..."):
                    try:
                        # Ejecutar la pregunta con el agente
                        response = agent.invoke({"input": user_question})
                        
                        # Agregar al historial
                        st.session_state.chat_history_energia.append({
                            "question": user_question,
                            "answer": response["output"]
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                        st.info("💡 Intenta reformular tu pregunta o verifica la sintaxis.")
            
            # Mostrar historial de conversación
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
            st.info("Verifica que tu API key de OpenAI sea válida y tenga créditos disponibles.")
    
    # Botón para actualizar datos
    st.markdown("---")
    if st.button("🔄 Actualizar Datos del Sistema", width="stretch"):
        # Limpiar cache y session state
        consultar_endpoint_energia.clear()
        if "df_energia" in st.session_state:
            del st.session_state.df_energia
        if "datos_json" in st.session_state:
            del st.session_state.datos_json
        if "chat_history_energia" in st.session_state:
            st.session_state.chat_history_energia = []
        if "api_username" in st.session_state:
            del st.session_state.api_username
        if "api_password" in st.session_state:
            del st.session_state.api_password
        # No borrar las fechas para mantenerlas como referencia
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
