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
    page_title="Control - Análisis con IA",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Control - Análisis con IA")
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
    """Convierte los datos JSON del endpoint a un DataFrame de pandas con parsing especializado para CUSUM"""
    
    def parse_cusum_column(df):
        """Parsea específicamente la columna CUSUM que contiene listas de diccionarios"""
        cusum_rows = []
        base_columns = [col for col in df.columns if col != 'cusum']
        
        for idx, row in df.iterrows():
            base_data = {col: row[col] for col in base_columns}
            
            if 'cusum' in row and row['cusum'] is not None:
                try:
                    # Si cusum es string, intentar parsearlo como JSON
                    if isinstance(row['cusum'], str):
                        import ast
                        cusum_data = ast.literal_eval(row['cusum'])
                    else:
                        cusum_data = row['cusum']
                    
                    # Si cusum_data es una lista de diccionarios
                    if isinstance(cusum_data, list) and len(cusum_data) > 0:
                        for cusum_item in cusum_data:
                            if isinstance(cusum_item, dict):
                                # Combinar datos base con datos de cusum
                                combined_row = base_data.copy()
                                combined_row.update(cusum_item)
                                cusum_rows.append(combined_row)
                    else:
                        # Si no es lista, mantener el registro original
                        base_data['cusum_raw'] = str(cusum_data)
                        cusum_rows.append(base_data)
                        
                except Exception as e:
                    # En caso de error, mantener datos originales
                    base_data['cusum_raw'] = str(row['cusum'])
                    base_data['cusum_parse_error'] = str(e)
                    cusum_rows.append(base_data)
            else:
                cusum_rows.append(base_data)
        
        return pd.DataFrame(cusum_rows) if cusum_rows else df
    
    def normalize_nested_data(data):
        """Normaliza datos anidados recursivamente"""
        if isinstance(data, dict):
            flattened = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)) and len(str(value)) > 500:
                    # Si es muy largo, convertir a string
                    flattened[key] = str(value)
                elif isinstance(value, dict):
                    # Aplanar diccionarios anidados
                    for nested_key, nested_value in value.items():
                        flattened[f"{key}_{nested_key}"] = nested_value
                elif isinstance(value, list) and len(value) > 0:
                    # Manejar listas especiales
                    if isinstance(value[0], dict):
                        # Lista de diccionarios - será procesada por parse_cusum_column
                        flattened[key] = value
                    else:
                        # Lista simple
                        flattened[key] = str(value)
                else:
                    flattened[key] = value
            return flattened
        return data
    
    def clean_dataframe(df):
        """Limpia el DataFrame de tipos problemáticos"""
        cleaned_df = df.copy()
        
        # Convertir columnas de fecha si existen
        date_columns = ['date', 'timestamp', 'created_at', 'updated_at']
        for col in cleaned_df.columns:
            if any(date_col in col.lower() for date_col in date_columns):
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                except:
                    pass
        
        # Limpiar otros tipos problemáticos
        for col in cleaned_df.columns:
            try:
                if cleaned_df[col].dtype == 'object':
                    # Verificar si contiene listas, dicts, etc.
                    sample_values = cleaned_df[col].dropna().head(3)
                    if not sample_values.empty:
                        sample_value = sample_values.iloc[0]
                        if isinstance(sample_value, (list, dict, tuple)) and col != 'cusum':
                            cleaned_df[col] = cleaned_df[col].astype(str)
                        else:
                            # Intentar conversión a numérico si es posible
                            numeric_series = pd.to_numeric(cleaned_df[col], errors='ignore')
                            if not numeric_series.equals(cleaned_df[col]):
                                cleaned_df[col] = numeric_series
            except:
                # Si hay cualquier error, convertir a string
                if col != 'cusum':  # No convertir cusum todavía
                    cleaned_df[col] = cleaned_df[col].astype(str)
        
        return cleaned_df
    
    try:
        df = None
        parsing_method = ""
        
        # Estrategia 1: Lista de diccionarios (más común)
        if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
            try:
                # Normalizar datos anidados
                normalized_data = [normalize_nested_data(item) for item in json_data]
                df = pd.DataFrame(normalized_data)
                parsing_method = "Lista de diccionarios normalizados"
            except:
                df = pd.DataFrame(json_data)
                parsing_method = "Lista de diccionarios directa"
        
        # Estrategia 2: Diccionario único
        elif isinstance(json_data, dict):
            # Verificar si el diccionario tiene una clave que contenga los datos principales
            data_keys = ['data', 'results', 'items', 'records', 'rows']
            main_data = None
            
            for key in data_keys:
                if key in json_data and isinstance(json_data[key], list):
                    main_data = json_data[key]
                    parsing_method = f"Diccionario con clave '{key}'"
                    break
            
            if main_data:
                # Usar los datos de la clave principal
                if len(main_data) > 0 and isinstance(main_data[0], dict):
                    normalized_data = [normalize_nested_data(item) for item in main_data]
                    df = pd.DataFrame(normalized_data)
                else:
                    df = pd.DataFrame(main_data)
            else:
                # Normalizar el diccionario completo
                normalized_dict = normalize_nested_data(json_data)
                df = pd.DataFrame([normalized_dict])
                parsing_method = "Diccionario único normalizado"
        
        # Estrategia 3: Lista simple
        elif isinstance(json_data, list):
            df = pd.DataFrame({'values': json_data})
            parsing_method = "Lista simple"
        
        # Estrategia 4: Otros tipos
        else:
            df = pd.DataFrame({'data': [json_data]})
            parsing_method = "Datos como valor único"
        
        # Procesamiento especializado para datos CUSUM
        if df is not None and 'cusum' in df.columns:
            try:
                df_expanded = parse_cusum_column(df)
                if len(df_expanded) > len(df):
                    df = df_expanded
                    parsing_method += " + Expansión CUSUM"
            except Exception as e:
                # Si falla la expansión, mantener DataFrame original
                parsing_method += f" (Error expansión CUSUM: {str(e)[:50]})"
        
        # Limpiar el DataFrame resultante
        if df is not None:
            df = clean_dataframe(df)
            
            # Verificar si el DataFrame está vacío
            if df.empty:
                df = pd.DataFrame({'raw_data': [str(json_data)]})
                parsing_method = "Fallback a datos raw"
            
            # Información adicional sobre el parsing
            parsing_info = {
                'method': parsing_method,
                'original_type': type(json_data).__name__,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'has_cusum_data': 'cusum' in str(json_data).lower(),
                'cusum_expanded': 'Expansión CUSUM' in parsing_method,
                'data_sample': str(json_data)[:300] + "..." if len(str(json_data)) > 300 else str(json_data)
            }
            
            return df, None, parsing_info
        else:
            return None, "No se pudo parsear el JSON con ninguna estrategia", None
            
    except Exception as e:
        # Último recurso: crear DataFrame con datos raw
        try:
            fallback_df = pd.DataFrame({
                'raw_json': [str(json_data)],
                'json_type': [type(json_data).__name__],
                'json_length': [len(str(json_data))]
            })
            parsing_info = {
                'method': 'Fallback completo',
                'original_type': type(json_data).__name__,
                'rows': 1,
                'columns': 3,
                'error': str(e)
            }
            return fallback_df, None, parsing_info
        except:
            return None, f"Error crítico convirtiendo JSON: {str(e)}", None

# Función para mostrar métricas CUSUM principales
def mostrar_metricas_cusum(df):
    """Muestra métricas específicas de CUSUM"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'abt_slope' in df.columns:
                abt_slope = df['abt_slope'].iloc[0] if not df['abt_slope'].empty else "N/A"
                st.metric("📈 ABT Slope", f"{abt_slope}")
            else:
                st.metric("📈 ABT Slope", "N/A")
        
        with col2:
            if 'abt_intercept' in df.columns:
                abt_intercept = df['abt_intercept'].iloc[0] if not df['abt_intercept'].empty else "N/A"
                st.metric("📊 ABT Intercept", f"{abt_intercept}")
            else:
                st.metric("📊 ABT Intercept", "N/A")
        
        with col3:
            # Buscar columna de fecha
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                try:
                    # Convertir a datetime si no lo está
                    if df[date_col].dtype == 'object':
                        dates = pd.to_datetime(df[date_col], errors='coerce')
                    else:
                        dates = df[date_col]
                    
                    ultima_fecha = dates.max()
                    if pd.notna(ultima_fecha):
                        st.metric("📅 Última Fecha", ultima_fecha.strftime("%Y-%m-%d"))
                    else:
                        st.metric("📅 Última Fecha", "N/A")
                except:
                    st.metric("📅 Última Fecha", "Error")
            else:
                st.metric("📅 Última Fecha", "No disponible")
        
        with col4:
            if 'cusumkWh' in df.columns:
                try:
                    cusum_values = pd.to_numeric(df['cusumkWh'], errors='coerce').dropna()
                    if not cusum_values.empty:
                        ultimo_cusum = cusum_values.iloc[-1]
                        st.metric("⚡ Último CUSUM kWh", f"{ultimo_cusum:.2f}")
                    else:
                        st.metric("⚡ Último CUSUM kWh", "N/A")
                except:
                    st.metric("⚡ Último CUSUM kWh", "Error")
            else:
                st.metric("⚡ Último CUSUM kWh", "No disponible")
                
    except Exception as e:
        st.error(f"Error mostrando métricas CUSUM: {str(e)}")

# Función para crear gráfico CUSUM
def crear_grafico_cusum(df):
    """Crea un gráfico de la columna cusumkWh usando herramientas estándar de Streamlit"""
    st.header("📈 Gráfico CUSUM kWh")
    
    # Verificar si existe la columna cusumkWh
    if 'cusumkWh' not in df.columns:
        st.warning("⚠️ No se encontró la columna 'cusumkWh' en los datos")
        return
    
    try:
        # Preparar los datos
        cusum_data = df.copy()
        
        # Convertir cusumkWh a numérico
        cusum_data['cusumkWh'] = pd.to_numeric(cusum_data['cusumkWh'], errors='coerce')
        
        # Buscar columna de fecha para el eje X
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        if date_columns:
            date_col = date_columns[0]
            # Convertir fechas
            cusum_data[date_col] = pd.to_datetime(cusum_data[date_col], errors='coerce')
            
            # Filtrar datos válidos
            valid_data = cusum_data.dropna(subset=[date_col, 'cusumkWh'])
            
            if valid_data.empty:
                st.warning("⚠️ No hay datos válidos para graficar")
                return
            
            # Ordenar por fecha
            valid_data = valid_data.sort_values(date_col)
            
            # Crear el gráfico con Streamlit
            st.subheader("Control Chart CUSUM - Consumo de Energía")
            
            # Preparar datos para el gráfico
            chart_data = valid_data.set_index(date_col)[['cusumkWh']].copy()
            
            # Usar st.line_chart para el gráfico principal
            st.line_chart(chart_data, height=400)
            
            # Mostrar información adicional
            st.info(f"📊 **Información del gráfico**: {len(valid_data)} puntos de datos desde {valid_data[date_col].min().strftime('%Y-%m-%d')} hasta {valid_data[date_col].max().strftime('%Y-%m-%d')}")
            
        else:
            # Si no hay columna de fecha, usar índice
            valid_data = cusum_data.dropna(subset=['cusumkWh'])
            
            if valid_data.empty:
                st.warning("⚠️ No hay datos válidos para graficar")
                return
            
            st.subheader("Control Chart CUSUM - Consumo de Energía")
            
            # Crear DataFrame con índice para el gráfico
            chart_data = pd.DataFrame({
                'CUSUM kWh': valid_data['cusumkWh'].values
            })
            
            # Usar st.line_chart
            st.line_chart(chart_data, height=400)
        
        # Mostrar estadísticas del gráfico
        st.subheader("📊 Estadísticas del CUSUM")
        col1, col2, col3, col4 = st.columns(4)
        
        cusum_values = valid_data['cusumkWh']
        
        with col1:
            st.metric("📊 Promedio", f"{cusum_values.mean():.2f}")
        with col2:
            st.metric("📏 Desv. Estándar", f"{cusum_values.std():.2f}")
        with col3:
            st.metric("⬆️ Máximo", f"{cusum_values.max():.2f}")
        with col4:
            st.metric("⬇️ Mínimo", f"{cusum_values.min():.2f}")
        
        # Análisis de tendencia
        st.subheader("🔍 Análisis de Tendencia")
        
        # Calcular tendencia simple
        if len(cusum_values) > 1:
            # Calcular diferencias para ver la tendencia
            diferencias = cusum_values.diff().dropna()
            tendencia_promedio = diferencias.mean()
            
            # Calcular correlación con el tiempo (aproximada)
            x_numeric = np.arange(len(cusum_values))
            correlation = np.corrcoef(x_numeric, cusum_values)[0, 1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Tendencia Promedio", f"{tendencia_promedio:.4f}")
            with col2:
                st.metric("📊 Correlación Temporal", f"{correlation:.4f}")
            with col3:
                if tendencia_promedio > 0.1:
                    tendencia_text = "↗️ Creciente"
                    color = "normal"
                elif tendencia_promedio < -0.1:
                    tendencia_text = "↘️ Decreciente"
                    color = "normal"
                else:
                    tendencia_text = "➡️ Estable"
                    color = "normal"
                st.metric("🎯 Dirección", tendencia_text)
            
            # Interpretación
            if tendencia_promedio > 0.1:
                st.warning("⚠️ **Interpretación**: Tendencia creciente significativa. El proceso puede estar fuera de control.")
            elif tendencia_promedio < -0.1:
                st.warning("⚠️ **Interpretación**: Tendencia decreciente significativa. El proceso puede estar mejorando o fuera de control.")
            else:
                st.success("✅ **Interpretación**: Tendencia estable. El proceso parece estar bajo control.")
        
        # Mostrar tabla de valores fuera de control si existen
        if len(cusum_values) > 1:
            # Calcular límites para detectar valores fuera de control
            limite_superior = cusum_values.mean() + 2 * cusum_values.std()
            limite_inferior = cusum_values.mean() - 2 * cusum_values.std()
            puntos_fuera_control = ((cusum_values > limite_superior) | (cusum_values < limite_inferior)).sum()
            
            if puntos_fuera_control > 0:
                st.subheader("🚨 Valores Fuera de Control")
                fuera_control = valid_data[(cusum_values > limite_superior) | (cusum_values < limite_inferior)]
                
                if date_columns:
                    st.dataframe(fuera_control[[date_col, 'cusumkWh']].round(2), use_container_width=True)
                else:
                    st.dataframe(fuera_control[['cusumkWh']].round(2), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error creando gráfico CUSUM: {str(e)}")
        st.info("💡 Verifica que los datos de cusumkWh sean numéricos válidos.")
def mostrar_info_dataframe(df):
    """Muestra información básica del DataFrame"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Filas", df.shape[0])
    with col2:
        st.metric("📊 Columnas", df.shape[1])
    with col3:
        try:
            memory_usage = df.memory_usage(deep=True).sum() / 1024
            st.metric("💾 Tamaño (KB)", f"{memory_usage:.1f}")
        except:
            st.metric("💾 Tamaño (KB)", "N/A")
    with col4:
        try:
            non_null_count = df.count().sum()
            st.metric("🔢 Valores No Nulos", non_null_count)
        except:
            st.metric("🔢 Valores No Nulos", "N/A")

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
                
                # Convertir JSON a DataFrame con la nueva función
                result = json_to_dataframe(datos_json)
                
                if len(result) == 3:  # Nueva función con parsing_info
                    df_cusum, error_df, parsing_info = result
                else:  # Función antigua sin parsing_info
                    df_cusum, error_df = result
                    parsing_info = None
                
                if df_cusum is not None:
                    st.success("✅ DataFrame creado exitosamente")
                    
                    # Mostrar información del parsing si está disponible
                    if parsing_info:
                        with st.expander("🔍 Información del Parsing"):
                            st.json(parsing_info)
                    
                    # Guardar en session state
                    st.session_state.df_cusum = df_cusum
                    st.session_state.datos_json_cusum = datos_json
                    st.session_state.parsing_info = parsing_info
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
    
    # Mostrar métricas específicas CUSUM
    st.header("⚡ Métricas CUSUM Principales")
    mostrar_metricas_cusum(df_cusum)
    
    # Crear gráfico CUSUM
    crear_grafico_cusum(df_cusum)
    
    # Tabs para diferentes vistas de los datos
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas", "🗂️ JSON Original"])
    
    with tab1:
        st.subheader("Vista de los Datos")
        st.dataframe(df_cusum, use_container_width=True)
    
    with tab2:
        st.subheader("Información del Dataset")
        if not df_cusum.empty:
            try:
                info_df = pd.DataFrame({
                    'Columna': df_cusum.columns,
                    'Tipo': df_cusum.dtypes.astype(str),
                    'No Nulos': df_cusum.count(),
                    'Nulos': df_cusum.isnull().sum(),
                    '% Nulos': (df_cusum.isnull().sum() / len(df_cusum) * 100).round(2)
                })
                st.dataframe(info_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error mostrando información del dataset: {str(e)}")
                # Mostrar información básica como alternativa
                st.write(f"**Columnas:** {list(df_cusum.columns)}")
                st.write(f"**Tipos de datos:** {dict(df_cusum.dtypes.astype(str))}")
        else:
            st.warning("DataFrame vacío")
    
    with tab3:
        st.subheader("Estadísticas Descriptivas")
        try:
            numeric_df = df_cusum.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No hay columnas numéricas para estadísticas descriptivas.")
                
            st.write("**Resumen de columnas:**")
            text_df = df_cusum.select_dtypes(include=['object', 'string'])
            if not text_df.empty:
                for col in text_df.columns:
                    try:
                        unique_vals = df_cusum[col].nunique()
                        st.write(f"• **{col}**: {unique_vals} valores únicos")
                    except Exception as e:
                        st.write(f"• **{col}**: Error calculando valores únicos - {str(e)[:50]}")
        except Exception as e:
            st.error(f"Error en estadísticas: {str(e)}")
            st.write("**Información básica disponible:**")
            st.write(f"- Forma del DataFrame: {df_cusum.shape}")
            st.write(f"- Columnas: {list(df_cusum.columns)}")
    
    with tab4:
        st.subheader("Datos JSON Originales")
        try:
            st.json(datos_json_cusum)
        except Exception as e:
            st.error(f"Error mostrando JSON: {str(e)}")
            st.text("Contenido del JSON (como texto):")
            st.text(str(datos_json_cusum))
    
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
                "¿Cómo evoluciona el CUSUM a lo largo del tiempo?",
                "¿Hay tendencias significativas en el consumo de energía?",
                "¿Cuándo el proceso estuvo fuera de control?",
                "¿Qué periodo cubren los datos (fechas)?",
                "¿Cuáles son los valores máximos y mínimos de CUSUM?",
                "¿Existe algún patrón en los datos CUSUM?",
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
    📊 Control - Sistema de Análisis de Control de Calidad con IA | Powered by SUME & SOSPOL
    </div>
    """, 
    unsafe_allow_html=True
)
