import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Importamos la función principal de tu script backend
from Inversion_experimental_app import generate_perovskite_samples

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Perovskite Synthesis Optimizer",
    page_icon="☀️",
    layout="wide"
)

# --- CARGA DE MODELOS EN CACHÉ ---
@st.cache_resource
def load_all_models():
    regressor = joblib.load('final_xgboost_model.joblib')
    generator = joblib.load('gmr_model_5_clusters.joblib')
    lle_model = joblib.load('lle_cos_model.pkl')
    return regressor, generator, lle_model

try:
    regressor, generator, lle_model = load_all_models()
    modelos_cargados = True
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    modelos_cargados = False

# --- INTERFAZ DE USUARIO ---
st.title("☀️ Herramienta de Regresión: Síntesis de Celdas de Perovskita")

# Sección de instrucciones y ejemplos
# Sección de instrucciones y ejemplos
with st.expander("📖 Instrucciones y Ejemplos de Formato", expanded=True):
    st.markdown("""
    **Formato de entrada:** Debes escribir el símbolo del ion o molécula seguido de su coeficiente estequiométrico (sin dejar espacios). 
    
    **Ejemplos válidos:**
    * `MA1Pb1I3` (Perovskita de yoduro de plomo y metilamonio)
    * `Cs0.05FA0.66MA0.29PbBr0.45I2.55` (Perovskita de triple catión)
    * `FA1Pb1I3`
    """)

st.divider()

# Columna de Entradas
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("1. Tipo de Perovskita")
    material_str = st.text_input(
        "Ingresa la fórmula:", 
        value="MA1Pb1I3",
        placeholder="Ej: Cs0.05FA0.66MA0.29PbBr0.45I2.55"
    )

with col2:
    st.subheader("2. PCE Deseado (%)")
    # Límites ajustados a 10.0 - 21.0
    target_pce_val = st.number_input(
        "Target PCE:", 
        min_value=10.0, 
        max_value=21.0, 
        value=17.0, 
        step=0.1,
        help="El rango permitido es de 10% a 21%"
    )

with col3:
    st.subheader("3. Prospectos")
    n_prospects = st.number_input(
        "N° de prospectos:", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1
    )

st.divider()

# Botón de ejecución
if st.button("Generar Prospectos de Síntesis", type="primary", use_container_width=True):
    if not modelos_cargados:
        st.error("Error crítico: Los modelos de ML no están disponibles.")
    elif not material_str.strip():
        st.warning("Por favor, ingresa una fórmula de material.")
    else:
        with st.spinner('Realizando modelado inverso mediante optimización COBYLA...'):
            # Límites de síntesis definidos en tu investigación
            input_bounds = [                                           
                [0.01, 5.0],   # 'DMF-DMSO-ratio'
                [1.0, 7.0],    # 'ann-thermal-budget'
                [1.1, 2.2],    # 'band-gap'
                [50, 300],     # '1st-ann-temperature'
                [0.01, 10]     # 'area_measured'
            ]
            
            try:
                # Llamada al backend optimizado
                resultados_df = generate_perovskite_samples(
                    material_expression=[material_str], 
                    target_pce=np.array([target_pce_val]), 
                    n_prospects=n_prospects, 
                    input_bounds=input_bounds,
                    loaded_regressor=regressor, 
                    gen_model=generator, 
                    lle_cos=lle_model
                )
                
                if resultados_df.empty:
                    st.warning("La optimización no pudo converger en prospectos válidos para estos parámetros.")
                else:
                    st.success(f"¡Resultados generados para {material_str}!")
                    
                    # Formateo de la tabla para que se vea profesional
                    st.markdown("### Parámetros Experimentales Sugeridos")
                    
                    # Renombrar columnas para la visualización final si lo deseas
                    df_display = resultados_df.copy()
                    
                    st.dataframe(
                        df_display, 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Opción para descargar los resultados
                    csv = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar Prospectos (CSV)",
                        csv,
                        f"prospectos_{material_str}.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
            except Exception as e:
                st.error(f"Error en el proceso de inversión: {str(e)}")