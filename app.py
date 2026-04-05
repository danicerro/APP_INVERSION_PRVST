import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time

# Import the main function from your backend script
from Inversion_experimental_app import generate_perovskite_samples

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Experiment Design Inversion",
    page_icon="☀️",
    layout="wide"
)

# --- HIDE THE RUNNING MAN ANIMATION ---
hide_running_man_style = """
<style>
div[data-testid="stStatusWidget"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
</style>
"""
st.markdown(hide_running_man_style, unsafe_allow_html=True)

# --- CACHE MODELS ---
@st.cache_resource
def load_all_models():
    regressor = joblib.load('final_xgboost_model.joblib')
    generator = joblib.load('gmr_model_5_clusters.joblib')
    lle_model = joblib.load('lle_cos_model.pkl')
    return regressor, generator, lle_model

try:
    regressor, generator, lle_model = load_all_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# --- USER INTERFACE ---
st.title("☀️ Experiment Design Inversion for Perovskite Solar Cells")

# Instructions and Examples Section
with st.expander("📖 Instructions and Input Format", expanded=True):
    st.markdown("""
    **Fast and easy to use!** Just provide these 3 inputs:
    1. **Perovskite Type:** The chemical formula of your material.
    2. **Target PCE (%):** Your desired Power Conversion Efficiency.
    3. **Prospects:** The maximum number of experimental conditions to generate.
    
    **Format rule for Perovskite Type:** You must write the symbol of the element or molecule followed by its stoichiometric coefficient (without leaving spaces). 
    
    You can construct any type of perovskite as long as you use the basic elements and molecules supported by the model.
    
    **5 Valid Examples:**
    * `MA1Pb1I3` (Methylammonium lead iodide)
    * `Cs0.05FA0.66MA0.29PbBr0.45I2.55` (Triple cation perovskite)
    * `FA1Pb1I3` (Formamidinium lead iodide)
    * `Cs1Pb1I2Br1` (Inorganic mixed-halide perovskite)
    * `MA0.5FA0.5Pb1I3` (Mixed cation perovskite)
    """)
    
    # Hidden list of allowed elements
    with st.expander("🔍 View supported elements and molecules"):
        allowed_ions = [
            '((CH3)3S)', '(1.3-Pr(NH3)2)', '(3AMP)', '(3AMPY)', '(4AMP)', '(4AMPY)', '(4ApyH)',
            '(4FPEA)', '(5-AVA)', '(5-AVAI)', '(6-ACA)', '(ALA)', '(APMim)', '(AVA)', '(Ace)',
            '(Ada)', '(Anyl)', '(BDA)', '(BEA)', '(BF4)', '(BIM)', '(BYA)', '(BZA)', '(BdA)',
            '(Br-PEA)', '(BzDA)', '(C6H4NH2)', '(CH3ND3)', '(CHMA)', '(CIEA)', '(CPEA)',
            '(Cl-PEA)', '(DAP)', '(DAT)', '(DMA)', '(EDA)', '(F-PEA)', '(F3EA)', '(GABA)',
            '(H-PEA)', '(HAD)', '(HEA)', '(HdA)', '(IEA)', '(ImEA)', '(NEA)', '(NH4)',
            '(NMA)', '(OdA)', '(PBA)', '(PDA)', '(PDMA)', '(PEA)', '(PEI)', '(PF6)',
            '(PGA)', '(PMA)', '(PTA)', '(PyEA)', '(PyrEA)', '(SCN)', '(TBA)', '(TEA)',
            '(TFEA)', '(THM)', '(TMA)', '(ThMA)', '(f-PEA)', '(iPA)', '(iso-BA)',
            '(mF1PEA)', '(n-C3H7NH3)', '(oF1PEA)', '(pF1PEA)', 'AN', 'Aa', 'Ag', 'BA',
            'BDA', 'BE', 'BU', 'Ba', 'Bi', 'Br', 'Ca', 'Cl', 'Co', 'Cs', 'Cu', 'DI',
            'EA', 'EDA', 'Eu', 'F', 'FA', 'Fe', 'GA', 'GU', 'Ge', 'HA', 'HDA', 'Hg',
            'I', 'IA', 'IM', 'In', 'K', 'La', 'Li', 'MA', 'Mg', 'Mn', 'Na', 'Nb',
            'Ni', 'O', 'PA', 'PDA', 'PF6', 'PN', 'PR', 'Pb', 'Rb', 'S', 'Sb', 'Sm',
            'Sn', 'Sr', 'TN', 'Tb', 'Ti', 'Y', 'Zn'
        ]
        st.write(", ".join(allowed_ions))

st.divider()

# Input Columns
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("1. Perovskite Type")
    material_str = st.text_input(
        "Enter the formula:", 
        value="MA1Pb1I3",
        placeholder="e.g., Cs0.05FA0.66MA0.29PbBr0.45I2.55",
        help="Format: Element or molecule symbol followed by its coefficient (no spaces)."
    )

with col2:
    st.subheader("2. Target PCE (%)")
    target_pce_val = st.number_input(
        "Target PCE:", 
        min_value=10.0, 
        max_value=21.0, 
        value=17.0, 
        step=0.1,
        help="Allowed range is between 10% and 21%"
    )

with col3:
    st.subheader("3. Prospects")
    n_prospects = st.number_input(
        "Maximum number of prospects:", 
        min_value=1, 
        max_value=50, 
        value=5, 
        step=1,
        help="Note: Output may be smaller as prospects failing physical constraints are discarded."
    )

st.divider()

# Execution Button
if st.button("Generate Synthesis Prospects", type="primary", use_container_width=True):
    if not models_loaded:
        st.error("Critical Error: ML models are not available.")
    elif not material_str.strip():
        st.warning("Please enter a material formula.")
    else:
        # Synthesis bounds
        input_bounds = [                                           
            [0.01, 5.0],   # 'DMF-DMSO-ratio'
            [1.0, 7.0],    # 'ann-thermal-budget'
            [1.1, 2.2],    # 'band-gap'
            [50, 300],     # '1st-ann-temperature'
            [0.01, 10]     # 'area_measured'
        ]
        
        # --- PROGRESS BAR IMPLEMENTATION (REAL TIME) ---
        my_bar = st.progress(0, text="Initializing optimization sequence...")
        
        # Esta función será llamada por el backend en cada ciclo
        def update_progress(current_step, total_steps):
            percent = int((current_step / total_steps) * 100)
            my_bar.progress(percent, text=f"Generating prospect {current_step} of {total_steps}... Please wait.")

        try:
            # Call to optimized backend, passing the callback function
            resultados_df = generate_perovskite_samples(
                material_expression=[material_str], 
                target_pce=np.array([target_pce_val]), 
                n_prospects=n_prospects, 
                input_bounds=input_bounds,
                loaded_regressor=regressor, 
                gen_model=generator, 
                lle_cos=lle_model,
                progress_callback=update_progress  # <--- SE PASA EL CALLBACK AQUÍ
            )
            
            my_bar.progress(100, text="Optimization complete!")
            time.sleep(0.5)
            my_bar.empty()
            
            if resultados_df.empty:
                st.warning("The optimization could not converge on valid prospects for these parameters.")
            else:
                st.success(f"Results generated for {material_str}!")
                
                st.markdown("### Suggested Experimental Parameters")
                
                df_display = resultados_df.copy()
                cols_to_drop = ['LLE-1', 'LLE-2', 'LLE-3', 'LLE-4']
                df_display = df_display.drop(columns=[col for col in cols_to_drop if col in df_display.columns])
                
                df_display = df_display.round(3)
                
                st.dataframe(
                    df_display, 
                    use_container_width=True,
                    hide_index=True
                )
                
                with st.expander("📊 Variables Glossary"):
                    st.markdown("""
                    * **`DMF-DMSO-ratio`**: $\chi_{sol}$, DMSO:DMF ratio expressed in logarithmic scale. DMSO and DMF (along with other solvents reported in the Perovskite Project Database) are used in the deposition of the perovskite layer.
                    * **`1st-ann-temperature`**: First temperature during the thermal annealing process.
                    * **`ann-thermal-budget`**: Thermal budget.
                    * **`band-gap`**: Perovskite band gap.
                    * **`area_measured`**: Cell area measured.
                    * **`PCE`**: Power Conversion Efficiency.
                    * **`Abs_Error`**: Absolute error of the predicted PCE with respect to the desired target value.
                    """)
                
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Prospects (CSV)",
                    csv,
                    f"prospects_{material_str}.csv",
                    "text/csv",
                    key='download-csv'
                )
                
        except Exception as e:
            my_bar.empty() 
            st.error(f"Error during the inversion process: {str(e)}")

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True) 
st.divider()

st.markdown("""
**For additional details, please refer to the article:** [https://pubs.acs.org/doi/10.1021/acs.jcim.5c02017](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02017)

<div style="text-align: right; font-size: 0.85em; color: #888888;">
    <b>Authors & Contact:</b><br>
    <a href="https://orcid.org/0000-0002-9643-5193" target="_blank" style="color: #888888; text-decoration: none;">F. Alexander Sepúlveda</a> | <a href="mailto:franklin@e3t.uis.edu.co" style="color: #888888; text-decoration: none;">franklin@e3t.uis.edu.co</a><br>
    <a href="https://orcid.org/0009-0009-0654-4520" target="_blank" style="color: #888888; text-decoration: none;">Daniel Cerro-Ramos</a> | <a href="mailto:daniel2258050@correo.uis.edu.co" style="color: #888888; text-decoration: none;">daniel2258050@correo.uis.edu.co</a>
</div>
""", unsafe_allow_html=True)
