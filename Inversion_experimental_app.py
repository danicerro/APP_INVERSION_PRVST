import sys
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))

import numpy as np
import pandas as pd
import re
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import joblib

# Asumiendo que estos módulos locales están en tu carpeta 'src' o en el mismo directorio
import GMM_utils as GMM
import ABX

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) 

# --- CONSTANTES ---
ION_LIST = [
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

# --- FUNCIONES AUXILIARES DE PARSEO Y REPRESENTACIÓN ---

def get_weight_vector(result_values, ion_list):
    weights = [0.0] * len(ion_list)
    ion_to_index = {ion: i for i, ion in enumerate(ion_list)}
    for site in ['A', 'B', 'X']:
        for symbol, value in result_values[site]:
            if symbol in ion_to_index:
                idx = ion_to_index[symbol]
                weights[idx] += value
            else:
                print(f"Warning: {symbol} not found in ION_LIST")
    return weights

def parse_with_ion_list_refined(formula, ion_list):
    formula = formula.replace(" ", "").replace("\u200b", "")
    sorted_ions = sorted(ion_list, key=len, reverse=True)
    escaped_ions = [re.escape(i) for i in sorted_ions]
    ion_pattern = "|".join(escaped_ions)
    
    full_pattern = fr"({ion_pattern})(\d*\.?\d*)"
    matches = re.findall(full_pattern, formula)
    
    tokens = []
    for sym, num in matches:
        val = float(num) if num else 1.0
        tokens.append((sym, val)) 

    sites = {'A': [], 'B': [], 'X': []}
    targets = [('A', 1.0), ('B', 1.0), ('X', 3.0)]
    
    current_idx = 0
    for site_name, target_sum in targets:
        current_sum = 0.0
        while current_idx < len(tokens) and current_sum < (target_sum - 1e-5):
            sites[site_name].append(tokens[current_idx])
            current_sum += tokens[current_idx][1]
            current_idx += 1
            
    return sites

def sparse_dict_prvskt_materials(name_prvkt, sparse_vectors):
    vector_sums = np.sum(sparse_vectors, axis=1, keepdims=True)  
    normalized_data = sparse_vectors/vector_sums                
    normalized_data_list = normalized_data.tolist()             
    sparse_vectors = normalized_data_list
    prvkt_dict = dict()
    for name, vector in zip(name_prvkt, sparse_vectors):
        if name not in prvkt_dict:
            prvkt_dict[name] = vector
    return prvkt_dict

# --- FUNCIONES DE OPTIMIZACIÓN Y MODELADO INVERSO ---

def objective_function(synthesis_inputs, material, target_PCE, loaded_model, input_bounds):
    for i in range(len(synthesis_inputs)):
        if not (input_bounds[i][0] <= synthesis_inputs[i] <= input_bounds[i][1]):
            return 1e9  

    full_inputs = np.hstack([material, synthesis_inputs])   
    inputs_reshaped = full_inputs.reshape(1, -1)
    pce_prediction = loaded_model.predict(inputs_reshaped)
    cost = np.abs(target_PCE - pce_prediction[0]) * np.abs(target_PCE - pce_prediction[0])   
    return cost

def condition_gmm(gmm, observed_dims, target_dims, z_observed):
    K = gmm.N_mix
    weights = []
    cond_means = []
    cond_covs = []

    for k in range(K):
        mu = gmm.gmm.means_[k]
        cov = gmm.gmm.covariances_[k]

        mu_a = mu[observed_dims]
        mu_b = mu[target_dims]

        Sigma_aa = cov[np.ix_(observed_dims, observed_dims)]
        Sigma_bb = cov[np.ix_(target_dims, target_dims)]
        Sigma_ab = cov[np.ix_(target_dims, observed_dims)]
        Sigma_ba = Sigma_ab.T

        Sigma_aa_inv = np.linalg.pinv(Sigma_aa)

        cond_mu = mu_b + Sigma_ab @ Sigma_aa_inv @ (z_observed - mu_a)
        cond_cov = Sigma_bb - Sigma_ab @ Sigma_aa_inv @ Sigma_ba

        cond_means.append(cond_mu)
        cond_covs.append(cond_cov)

        w_k = gmm.gmm.weights_[k]
        p_z_a = multivariate_normal(mean=mu_a, cov=Sigma_aa).pdf(z_observed)
        weights.append(w_k * p_z_a)

    weights = np.array(weights)
    weights /= np.sum(weights)

    return {
        'weights': weights,
        'means': cond_means,
        'covariances': cond_covs
    }

def sample_from_conditional_gmm(cond_gmm, n_samples=1, random_state=None):
    rng = np.random.default_rng(random_state)
    weights = cond_gmm['weights']
    means = cond_gmm['means']
    covariances = cond_gmm['covariances']
    n_components = len(weights)
    dim = len(means[0])

    component_indices = rng.choice(n_components, size=n_samples, p=weights)

    samples = np.zeros((n_samples, dim))
    for i, comp in enumerate(component_indices):
        samples[i] = rng.multivariate_normal(means[comp], covariances[comp])

    return samples

def array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y):
    z_material = np.array(z_material)
    zeros = np.zeros(len(target_dims))
    z_material = np.append(z_material, zeros)
    target_dims = np.array(target_dims)
    if z_material.ndim == 1:
        z_o = z_material.reshape(1, -1)  
        z_o = scaler_X.transform(z_o)    
        Zo = np.delete(z_o, target_dims, axis=1)  
        y_o = np.array([[z_pce]])               
        y_o = scaler_y.transform(y_o)           
        Zo = np.hstack((Zo, y_o))               
        Zo = Zo.flatten()
    return Zo

def inverse_transform_batch(X_target, scaler, target_dims, observed_dims):
    n_samples, p = X_target.shape
    N = len(target_dims) + len(observed_dims)
    X_full = np.zeros((n_samples, N))
    X_full[:, target_dims] = X_target
    X_full_inv = scaler.inverse_transform(X_full)
    X_target_inv = X_full_inv[:, target_dims]
    return X_target_inv

# OPTIMIZACIÓN: Recibe los modelos cargados en lugar de los archivos
def inferre_exp_conditions(loaded_regressor, gen_model, target_PCE, material, input_bounds):
    Nsamples_generator = 2   
    K = 1                    

    gmm_model = gen_model['model']
    scaler_X  = gen_model['scaler_x']
    scaler_y  = gen_model['scaler_y']

    observed_dims = [0, 1, 2, 3, 9]
    target_dims   = [4, 5, 6, 7, 8]
    z_material = material
    z_pce      = target_PCE

    Zo = array_cond_Zo(z_material, z_pce, target_dims, scaler_X, scaler_y)   
    cond_gmm = condition_gmm(gmm_model, observed_dims, target_dims, Zo)            
    new_samples = sample_from_conditional_gmm(cond_gmm, n_samples=Nsamples_generator, random_state=None)
    
    observed_dims = [0, 1, 2, 3]       
    X_gen = inverse_transform_batch(new_samples, scaler_X, target_dims, observed_dims)

    ii = 0
    optimization_results = {}
    for init_vector in X_gen:
        synthesis_inputs = init_vector
        initial_step_size = 0.05
        result = minimize(
            objective_function,           
            synthesis_inputs,             
            method='COBYLA',
            args=(material, target_PCE, loaded_regressor, input_bounds), 
            options={'maxiter': 50,
                'rhobeg': initial_step_size,
                'tol': 1e-4,                 
                'disp': False
                }
            )
        if result.success:
            optimal_synthesis_inputs = result.x
            final_cost = result.fun
            optimal_full_inputs = np.hstack([material, optimal_synthesis_inputs])
            final_pce_prediction = loaded_regressor.predict(optimal_full_inputs.reshape(1, -1))[0]
            optimization_results[ii] = {
                'status': 'success',
                'optimal_inputs': optimal_full_inputs,
                'predicted_pce': final_pce_prediction,
                'cost': final_cost,
                'initial_guess': init_vector
                }
        else:
            # print(f"  Status: FAILED ({result.message})") # Ocultamos esto para no ensuciar logs en web
            optimization_results[ii] = {
                'status': 'failed',
                'message': result.message,
                'initial_guess': init_vector
                }
        ii += 1

    successful_runs = {k: v for k, v in optimization_results.items() if v['status'] == 'success'}
    best_candidates = sorted(
        successful_runs.items(), 
        key=lambda item: abs(target_PCE - item[1]['predicted_pce'])
        )

    inverted_points = best_candidates[:K]
    optimal_inputs_list = []
    predicted_pce_list = []
    
    for run_index, results in inverted_points:
        if results['status'] == 'success' and results['cost'] < 1e9: 
            optimal_inputs_list.append(results['optimal_inputs'])
            predicted_pce_list.append(results['predicted_pce'])

    return optimal_inputs_list

# OPTIMIZACIÓN: Recibe el modelo cargado directamente
def verify_inferred_inputs(loaded_regressor, material, optimal_synthesis_inputs):
    synthesis_batch = np.array(optimal_synthesis_inputs)
    if synthesis_batch.size == 0:
        return np.array([]) 

    pce_predictions = loaded_regressor.predict(synthesis_batch)
    return pce_predictions

# --- FUNCIÓN PRINCIPAL OPTIMIZADA ---
# OPTIMIZACIÓN: Ahora recibe los modelos pre-cargados como argumentos
def generate_perovskite_samples(material_expression, target_pce, n_prospects, input_bounds, loaded_regressor, gen_model, lle_cos):
    vectores = []
    for mat in material_expression:  
        parsed_result = parse_with_ion_list_refined(mat, ION_LIST)
        vector = get_weight_vector(parsed_result, ION_LIST)
        vectores.append(vector)
        
    vectores = np.vstack(vectores)
    prvkt_dict = sparse_dict_prvskt_materials(material_expression, vectores)
    sparse_vectors = list(prvkt_dict.values())
    data_matrix = np.array(sparse_vectors)

    V_transf = lle_cos.transform(data_matrix)
    material = V_transf[0] 

    rows = []
    for pp in range(0, n_prospects):
        # Pasamos los modelos ya cargados
        opt_input = inferre_exp_conditions(loaded_regressor, gen_model, target_pce[0], material, input_bounds)
        if opt_input:
            pred_pce = verify_inferred_inputs(loaded_regressor, material, opt_input) 
            pce_val = pred_pce[0] if isinstance(pred_pce, (list, np.ndarray)) else pred_pce
            flat_input = np.ravel(opt_input).tolist()
            combined_row = flat_input + [pce_val]
            rows.append(combined_row)

    if not rows:
        return pd.DataFrame() # Retorna df vacío si no hay resultados

    var_names = ['LLE-1', 'LLE-2', 'LLE-3', 'LLE-4', 'DMF-DMSO-ratio', 'ann-thermal-budget', 'band-gap', '1st-ann-temperature', 'area_measured', 'PCE']
    df_prospects = pd.DataFrame(rows, columns=var_names)
    
    df_prospects = df_prospects.apply(pd.to_numeric, errors='coerce')
    df_prospects['Abs_Error'] = (df_prospects['PCE'] - target_pce[0]).abs()
    df_prospects = df_prospects.sort_values(by='Abs_Error', ascending=True)
    df_prospects = df_prospects.reset_index(drop=True)

    return df_prospects

# Bloque de prueba local para confirmar que sigue funcionando
if __name__ == "__main__":
    # Cargar los modelos UNA SOLA VEZ aquí para las pruebas locales
    print("Cargando modelos en memoria...")
    regressor = joblib.load('final_xgboost_model.joblib')
    generator = joblib.load('gmr_model_5_clusters.joblib')
    lle_model = joblib.load('lle_cos_model.pkl')
    print("Modelos cargados con éxito.\n")

    # Prueba 1
    material_expression = ['MA1Pb1I3']  
    pce_targets = np.array([20.5])      
    input_bounds = [                                           
        [0.01, 5.0],   #'DMF-DMSO-ratio'
        [1.0, 7.0],    # 'ann-thermal-budget'
        [1.1, 2.2],    # 'band-gap'
        [50, 300],     # '1st-ann-temperature'
        [0.01, 10]     # 'area_measured'
        ]
    N_runs = 5                                         
    
    print(f"Probando {material_expression[0]}...")
    prospects = generate_perovskite_samples(material_expression, pce_targets, N_runs, input_bounds, regressor, generator, lle_model)
    print(prospects)