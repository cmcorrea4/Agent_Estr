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
from langchain_core.messages import HumanMessage, SystemMessage
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de gestión energética--ESTRA",
    page_icon="🏭",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE ENDPOINTS
# ─────────────────────────────────────────────
ENDPOINTS = {
    "summary":   "https://energy-api-628964750053.us-east1.run.app/test-summary",
    "moldes":    "https://energy-api-628964750053.us-east1.run.app/test-mold",
    "referencias": "https://energy-api-628964750053.us-east1.run.app/test-reference",
    "linea_base": "https://energy-api-628964750053.us-east1.run.app/test-baseline",
}

ENDPOINT_LABELS = {
    "summary":    "📊 Resumen General",
    "moldes":     "🔩 Moldes",
    "referencias": "🏷️ Referencias",
    "linea_base": "📐 Línea Base",
}

# ─────────────────────────────────────────────
# FUNCIONES DE UTILIDAD
# ─────────────────────────────────────────────

def get_week_start(date):
    return date - timedelta(days=date.weekday())

def get_week_end(date):
    return date + timedelta(days=6 - date.weekday())

def get_auth_header(username, password):
    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {encoded}',
        'User-Agent': 'StreamlitApp/1.0',
        'Accept': 'application/json'
    }

@st.cache_data(ttl=300)
def consultar_endpoint(endpoint_key, username, password, date_start=None, date_end=None):
    """Consulta cualquier endpoint de la API energética."""
    try:
        url = ENDPOINTS[endpoint_key]
        params = {}
        if date_start:
            params['dateStart'] = date_start
        if date_end:
            params['dateEnd'] = date_end

        headers = get_auth_header(username, password)
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            try:
                return response.json(), None
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


def json_to_dataframe(json_data):
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


def mostrar_info_dataframe(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Filas", df.shape[0])
    with col2:
        st.metric("📊 Columnas", df.shape[1])
    with col3:
        st.metric("💾 Tamaño (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
    with col4:
        st.metric("🔢 Valores No Nulos", df.count().sum())


# ─────────────────────────────────────────────
# ROUTER INTELIGENTE CON LLM
# ─────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """
Eres un clasificador de intención para una aplicación de análisis energético industrial.
Tu única tarea es leer la pregunta del usuario y responder EXCLUSIVAMENTE con una de estas cuatro palabras:

- summary      → preguntas generales sobre el dataset, consumo global, producción total, resumen, eficiencia general, periodos de tiempo
- moldes       → preguntas sobre moldes, mold, SECn por molde, productividad de moldes, tiempos de paro de moldes
- referencias  → preguntas sobre referencias, productos, referencia específica, SKU, códigos de producto
- linea_base   → preguntas sobre línea base, baseline, consumo de referencia, metas energéticas, benchmarks

Responde SOLO con la palabra clave, sin explicación, sin puntos, sin mayúsculas.
"""

def clasificar_intencion(pregunta: str, llm: ChatOpenAI) -> str:
    """Usa el LLM para clasificar la intención y determinar qué endpoint usar."""
    try:
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=pregunta)
        ]
        response = llm.invoke(messages)
        intencion = response.content.strip().lower()

        # Validar que la respuesta sea válida
        if intencion not in ENDPOINTS:
            # Fallback a summary si la clasificación falla
            return "summary"
        return intencion
    except Exception:
        return "summary"


def cargar_dataframe_por_intencion(intencion: str) -> tuple[pd.DataFrame | None, str | None]:
    """Carga el DataFrame del endpoint correspondiente a la intención (bajo demanda)."""
    cache_key = f"df_{intencion}"

    # Si ya está en session_state, no volver a consultar
    if cache_key in st.session_state:
        return st.session_state[cache_key], None

    username = st.session_state.get('api_username', '')
    password = st.session_state.get('api_password', '')
    date_start = st.session_state.get('date_start')
    date_end = st.session_state.get('date_end')

    date_start_str = date_start.strftime('%Y-%m-%d') if date_start else None
    date_end_str = date_end.strftime('%Y-%m-%d') if date_end else None

    datos_json, error = consultar_endpoint(intencion, username, password, date_start_str, date_end_str)

    if error:
        return None, error

    df, error_df = json_to_dataframe(datos_json)
    if error_df:
        return None, error_df

    # Guardar en session_state para reutilizar
    st.session_state[cache_key] = df
    st.session_state[f"json_{intencion}"] = datos_json
    return df, None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.title("🏭 Diagnóstico de gestión energética--ESTRA")
st.markdown("**Obtén datos del sistema energético y analízalos con IA avanzada**")

with st.sidebar:
    st.header("⚙️ Panel de Control")

    # ── Credenciales del endpoint ──
    st.subheader("🔌 Configuración del Endpoint")

    if "df_energia" not in st.session_state:
        api_username = st.text_input("👤 Usuario del Endpoint:", placeholder="Ingresa tu usuario")
        api_password = st.text_input("🔒 Contraseña del Endpoint:", type="password", placeholder="Ingresa tu contraseña")
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

    # ── Filtros de fecha ──
    st.subheader("📅 Filtro de Fechas")
    filter_type = st.radio("Tipo de filtro:", ["Por semana", "Por rango de fechas"])

    if filter_type == "Por semana":
        today = datetime.now().date()
        selected_week = st.date_input(
            "Selecciona una fecha (se usará su semana completa):",
            value=st.session_state.get('selected_week', today)
        )
        date_start = get_week_start(selected_week)
        date_end = get_week_end(selected_week)
        st.info(f"📅 Semana del **{date_start.strftime('%d/%m/%Y')}** al **{date_end.strftime('%d/%m/%Y')}**")
        dates_valid = True
    else:
        default_start = st.session_state.get('date_start', datetime(2024, 1, 1).date())
        default_end = st.session_state.get('date_end', datetime.now().date())
        date_start = st.date_input("Fecha de inicio:", value=default_start)
        date_end = st.date_input("Fecha de fin:", value=default_end)
        if date_start > date_end:
            st.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin")
            dates_valid = False
        else:
            dates_valid = True
            dias = (date_end - date_start).days + 1
            st.info(f"📊 Rango: {dias} días")

    st.markdown("---")

    # ── OpenAI API Key ──
    st.subheader("🤖 Configuración de OpenAI")

    if "openai_api_key" not in st.session_state:
        openai_api_key = st.text_input("🔑 API Key de OpenAI:", type="password", placeholder="sk-...")
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
        if st.button("🔄 Cambiar API Key"):
            del st.session_state.openai_api_key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            st.rerun()

    model_name = "gpt-4"
    temperature = 0.1

    st.markdown("---")

    # ── Botón obtener datos (summary) ──
    if st.button("🔌 Obtener Datos del Sistema", use_container_width=True,
                 disabled=not (endpoint_configured and dates_valid)):
        with st.spinner("Consultando endpoint de energía..."):
            date_start_str = date_start.strftime('%Y-%m-%d')
            date_end_str = date_end.strftime('%Y-%m-%d')

            datos_json, error = consultar_endpoint("summary", api_username, api_password, date_start_str, date_end_str)

            if datos_json is not None:
                df_energia, error_df = json_to_dataframe(datos_json)
                if df_energia is not None:
                    st.session_state.df_energia = df_energia
                    st.session_state.datos_json = datos_json
                    st.session_state.api_username = api_username
                    st.session_state.api_password = api_password
                    st.session_state.date_start = date_start
                    st.session_state.date_end = date_end
                    st.session_state.filter_type = filter_type
                    if filter_type == "Por semana":
                        st.session_state.selected_week = selected_week
                    # Limpiar DataFrames cargados previamente para forzar recarga con nuevas fechas
                    for key in ["df_moldes", "df_referencias", "df_linea_base",
                                "json_moldes", "json_referencias", "json_linea_base"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    consultar_endpoint.clear()
                    st.success("✅ Datos cargados correctamente")
                    st.rerun()
                else:
                    st.error(f"❌ Error creando DataFrame: {error_df}")
            else:
                st.error(f"❌ Error obteniendo datos: {error}")

    # Estado de la conexión
    if "df_energia" in st.session_state:
        st.success("🟢 Datos cargados y listos")
        st.info(f"📊 {st.session_state.df_energia.shape[0]} filas × {st.session_state.df_energia.shape[1]} columnas")
        if 'date_start' in st.session_state:
            st.info(f"📅 {st.session_state.date_start.strftime('%d/%m/%Y')} → {st.session_state.date_end.strftime('%d/%m/%Y')}")

        # Mostrar qué endpoints ya han sido cargados bajo demanda
        st.markdown("**Endpoints cargados:**")
        for key in ["moldes", "referencias", "linea_base"]:
            if f"df_{key}" in st.session_state:
                st.success(f"  {ENDPOINT_LABELS[key]} ✅")
    else:
        st.warning("🔴 Sin datos del sistema")


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────

if "df_energia" not in st.session_state:
    st.info("👆 Configura las credenciales, selecciona el filtro de fechas y haz clic en 'Obtener Datos del Sistema'")
    st.markdown("---")
    st.subheader("ℹ️ Sobre esta aplicación")
    st.markdown("""
    Esta aplicación integra cuatro endpoints del sistema ESTRA:

    | Endpoint | Cuándo se usa |
    |---|---|
    | 📊 Resumen General (`/test-summary`) | Carga inicial y preguntas generales |
    | 🔩 Moldes (`/test-mold`) | Preguntas sobre moldes y SECn por molde |
    | 🏷️ Referencias (`/test-reference`) | Preguntas sobre referencias o productos |
    | 📐 Línea Base (`/test-baseline`) | Preguntas sobre metas o benchmarks energéticos |

    El **router inteligente** usa GPT para detectar la intención de cada pregunta y cargar automáticamente el endpoint correcto, solo cuando se necesita.
    """)

else:
    df_energia = st.session_state.df_energia
    datos_json = st.session_state.datos_json

    st.success("✅ Datos del sistema energético cargados exitosamente")

    if 'date_start' in st.session_state:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"📅 Desde: **{st.session_state.date_start.strftime('%d/%m/%Y')}**")
        with col2:
            st.info(f"📅 Hasta: **{st.session_state.date_end.strftime('%d/%m/%Y')}**")
        with col3:
            dias = (st.session_state.date_end - st.session_state.date_start).days + 1
            st.metric("📊 Días", dias)

    # ── Información del Dataset (summary) ──
    st.header("📊 Resumen General del Dataset")
    mostrar_info_dataframe(df_energia)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas", "🗂️ JSON Original"])

    with tab1:
        st.dataframe(df_energia, use_container_width=True)
    with tab2:
        info_df = pd.DataFrame({
            'Columna': df_energia.columns,
            'Tipo': df_energia.dtypes.astype(str),
            'No Nulos': df_energia.count(),
            'Nulos': df_energia.isnull().sum(),
            '% Nulos': (df_energia.isnull().sum() / len(df_energia) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
    with tab3:
        numeric_df = df_energia.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No hay columnas numéricas para estadísticas descriptivas.")
    with tab4:
        st.json(datos_json)

    # ─────────────────────────────────────────────
    # AGENTE IA CON ROUTER INTELIGENTE
    # ─────────────────────────────────────────────
    st.header("🤖 Agente de Análisis IA con Router Inteligente")

    st.markdown("""
    El agente detecta automáticamente el tipo de pregunta y consulta el endpoint correspondiente:
    
    | Tu pregunta menciona... | Endpoint que se usa |
    |---|---|
    | moldes, SECn por molde, productividad de molde | 🔩 `/test-mold` |
    | referencia, producto, SKU, código | 🏷️ `/test-reference` |
    | línea base, baseline, meta, benchmark | 📐 `/test-baseline` |
    | general, resumen, producción total, periodo | 📊 `/test-summary` |
    """)

    AGENT_SYSTEM_PROMPT = """
    Eres un analista experto en datos energéticos industriales para la empresa ESTRA.
    Responde SIEMPRE en español.
    Usa lenguaje técnico claro y adecuado para ingenieros.
    Cuando presentes números, incluye unidades cuando sea posible.
    Sé conciso pero completo en tus respuestas.
    """

    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.warning("⚠️ Configura tu API Key de OpenAI en la barra lateral para usar el agente inteligente.")
    else:
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=st.session_state.openai_api_key
            )

            # ── Ejemplos de preguntas ──
            st.subheader("💡 Ejemplos de preguntas:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🔩 Moldes:**
                - ¿Qué moldes tienen la mayor productividad efectiva?
                - ¿Cuál molde tiene mayor SECn?
                - ¿En qué fechas se trabajó el molde 15252?

                **🏷️ Referencias:**
                - ¿Qué referencias tienen mayor consumo energético?
                - ¿Cuáles son los productos con mayor tiempo de paro?
                """)
            with col2:
                st.markdown("""
                **📐 Línea Base:**
                - ¿Cuál es la línea base de consumo energético?
                - ¿Qué referencias están por encima del benchmark?

                **📊 General:**
                - ¿Qué información contiene el dataset?
                - ¿Cuál es el consumo total del periodo?
                """)

            # ── Historial de conversación ──
            if 'chat_history_energia' not in st.session_state:
                st.session_state.chat_history_energia = []

            # ── Input del usuario ──
            st.subheader("❓ Consulta los datos con IA")
            user_question = st.text_input(
                "Escribe tu pregunta:",
                placeholder="Ej: ¿Qué moldes tienen mayor SECn?",
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
                with st.spinner("🔍 Clasificando la pregunta con el router inteligente..."):
                    intencion = clasificar_intencion(user_question, llm)

                endpoint_label = ENDPOINT_LABELS.get(intencion, "📊 Resumen General")
                st.info(f"🎯 Router detectó: **{endpoint_label}** → consultando `{ENDPOINTS[intencion]}`")

                # Cargar el DataFrame bajo demanda
                if intencion == "summary":
                    df_para_agente = df_energia
                    carga_error = None
                else:
                    with st.spinner(f"📡 Cargando datos de {endpoint_label}..."):
                        df_para_agente, carga_error = cargar_dataframe_por_intencion(intencion)

                if carga_error:
                    st.error(f"❌ Error cargando {endpoint_label}: {carga_error}")
                elif df_para_agente is not None:
                    st.success(f"✅ Datos de {endpoint_label} listos: {df_para_agente.shape[0]} filas × {df_para_agente.shape[1]} columnas")

                    with st.spinner("🤖 El agente está analizando los datos..."):
                        try:
                            agent = create_pandas_dataframe_agent(
                                llm,
                                df_para_agente,
                                verbose=False,
                                allow_dangerous_code=True,
                                prefix=AGENT_SYSTEM_PROMPT
                            )

                            response = agent.invoke({"input": user_question})

                            st.session_state.chat_history_energia.append({
                                "question": user_question,
                                "answer": response["output"],
                                "endpoint": endpoint_label,
                                "df_shape": f"{df_para_agente.shape[0]} filas × {df_para_agente.shape[1]} columnas"
                            })
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                            st.info("💡 Intenta reformular tu pregunta.")

            # ── Historial de conversación ──
            if st.session_state.chat_history_energia:
                st.subheader("💬 Análisis Realizados")

                for i, chat in enumerate(reversed(st.session_state.chat_history_energia)):
                    label = f"❓ {chat['question'][:50]}..." if len(chat['question']) > 50 else f"❓ {chat['question']}"
                    with st.expander(label, expanded=(i == 0)):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write("**Pregunta:**")
                            st.write(chat['question'])
                        with col2:
                            st.caption(f"Endpoint: {chat.get('endpoint', 'N/A')}")
                            st.caption(f"Datos: {chat.get('df_shape', 'N/A')}")
                        st.write("**Análisis del Agente IA:**")
                        st.write(chat['answer'])
                        st.divider()

        except Exception as e:
            st.error(f"❌ Error al inicializar el agente: {str(e)}")
            st.info("Verifica que tu API key de OpenAI sea válida y tenga créditos disponibles.")

    # ── Botón para actualizar datos ──
    st.markdown("---")
    if st.button("🔄 Actualizar Todos los Datos", use_container_width=True):
        consultar_endpoint.clear()
        keys_to_delete = [
            "df_energia", "datos_json", "chat_history_energia",
            "api_username", "api_password",
            "df_moldes", "df_referencias", "df_linea_base",
            "json_moldes", "json_referencias", "json_linea_base"
        ]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
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
