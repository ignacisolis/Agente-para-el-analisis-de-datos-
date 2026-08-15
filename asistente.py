import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

st.set_page_config(page_title="Agente Data Analyst (Gratis con Groq)", layout="wide")
st.title("📊 Agente Data Analyst con IA (Groq - Llama 3)")
st.write("Sube tu CSV y realiza consultas de datos o gráficos sin costo de API.")

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")

uploaded_file = st.file_uploader("Carga tu dataset en formato CSV", type=["csv"])


def load_csv_robust(uploaded_file):
    """Lee el CSV probando distintas codificaciones hasta encontrar una que funcione."""
    raw = uploaded_file.read()
    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=None,             # Detecta automáticamente el separador (comas, punto y coma, tabulación)
                engine="python",      # Motor más flexible para leer archivos imperfectos
                encoding=enc,
                on_bad_lines="skip"   # Omite filas con formatos dañados sin romper el programa
            )
            # Normaliza nombres de columnas: quita espacios sobrantes al inicio/final
            df.columns = [str(c).strip() for c in df.columns]
            return df, enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


# Lista blanca de módulos que el código generado puede importar
ALLOWED_MODULES = {
    "pandas", "matplotlib", "matplotlib.pyplot", "numpy", "seaborn",
    "scipy", "scipy.stats",
    "sklearn", "sklearn.linear_model", "sklearn.cluster",
    "sklearn.preprocessing", "sklearn.model_selection", "sklearn.metrics",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Reemplazo restringido de __import__: solo permite módulos de la lista blanca."""
    if name not in ALLOWED_MODULES:
        raise ImportError(f"Import de '{name}' no permitido en el sandbox.")
    return __import__(name, globals, locals, fromlist, level)


# Builtins seguros que el código generado puede usar (excluye eval, exec, open, etc.)
SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "len": len,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "type": type,
    "__import__": safe_import,
}


def execute_generated_code(code_str: str, df: pd.DataFrame):
    local_vars = {"df": df, "plt": plt, "pd": pd, "np": np, "sns": sns}
    global_vars = {"__builtins__": SAFE_BUILTINS}

    plt.clf()

    try:
        exec(code_str, global_vars, local_vars)
        result = local_vars.get("result", None)

        fig = plt.gcf()
        has_plot = len(fig.axes) > 0

        return result, fig if has_plot else None, None
    except KeyError as e:
        columnas_disponibles = ", ".join(df.columns)
        mensaje = f"No se encontró la columna {e}. Columnas disponibles: {columnas_disponibles}"
        return None, None, mensaje
    except Exception as e:
        return None, None, str(e)


if uploaded_file and api_key:
    # Cliente oficial de Groq
    client = Groq(api_key=api_key)

    try:
        df, used_encoding = load_csv_robust(uploaded_file)
        if used_encoding != "utf-8":
            st.caption(f"⚠️ Archivo leído con codificación `{used_encoding}` (no era UTF-8).")
    except Exception as e:
        st.error(f"No se pudo leer el archivo CSV: {e}")
        st.stop()

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    query = st.text_input(
        "¿Qué quieres saber o graficar de tus datos?",
        placeholder="Ej: ¿Cuál es el promedio de ventas por categoría? o Haz un gráfico del top 5 productos"
    )

    if st.button("Consultar al Agente"):
        if not query:
            st.warning("Por favor, ingresa una consulta.")
        else:
            columns_info = f"Columnas y tipos:\n{df.dtypes.to_string()}\n\nMuestra de datos:\n{df.head(2).to_string()}"

            prompt = f"""
            Eres un Agente Analista de Datos experto en Python.
            Tu objetivo es generar código Python válido para responder a la siguiente pregunta sobre un DataFrame llamado `df`.

            Contexto del DataFrame:
            {columns_info}

            Pregunta del usuario: "{query}"

            REGLAS ESTRICTAS DE RESPUESTA:
            1. Devuelve ÚNICAMENTE código en Python ejecutable. No incluyas nada de texto explicativo antes o después.
            2. NO uses bloques de formato markdown como ```python o ```. Devuelve texto plano de código únicamente.
            3. Si la consulta requiere un cálculo o respuesta escrita, asigna el resultado a la variable `result`.
            4. Si la consulta pide un gráfico, usa `plt` (Matplotlib) o `sns` (Seaborn) para crearlo. No uses `plt.show()`.
            5. Solo puedes usar `df`, `pd`, `plt`, `np` y `sns`. También puedes importar `scipy` o submódulos de `sklearn` (como `sklearn.linear_model`, `sklearn.cluster`, `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.metrics`) si la consulta requiere estadística avanzada o modelos predictivos. No importes otras librerías como `os` o `sys`.
            6. Usa EXACTAMENTE los nombres de columnas listados arriba (respetando mayúsculas, minúsculas y tildes). No inventes ni traduzcas nombres de columnas.
            """

            with st.spinner("El Agente (Llama 3) está analizando los datos..."):
                try:
                    # Modelo gratuito Llama 3.3 de 70B parámetros en Groq
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )

                    generated_code = response.choices[0].message.content.strip()
                    generated_code = generated_code.replace("```python", "").replace("```", "").strip()

                    with st.expander("Ver código ejecutado por el agente"):
                        st.code(generated_code, language="python")

                    result, fig, error = execute_generated_code(generated_code, df)

                    if error:
                        st.error(f"Error al ejecutar el análisis: {error}")
                    else:
                        if result is not None:
                            st.subheader("Resultado:")
                            if isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.success(f"**Respuesta:** {result}")

                        if fig:
                            st.subheader("Gráfico generado:")
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error en la llamada a Groq: {e}")

elif not api_key:
    st.info("Ingresa tu Groq API Key gratuita en la barra lateral para comenzar.")