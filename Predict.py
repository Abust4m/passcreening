import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
import joblib
import warnings
import os
import shap
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

INPUT_FILE = ''
OUTPUT_FILE = ''
WATERFALL_DIR = ''

BASE_PATH = ''
MODEL_PATH = f'{BASE_PATH}\\'
SCALER_PATH = f'{BASE_PATH}\\'
FEATURE_NAMES_PATH = f'{BASE_PATH}\\'
CORR_FILTERED_PATH = f'{BASE_PATH}\\'
X_SELECTED_PATH = f'{BASE_PATH}\\'

def generate_morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp)

def generate_maccs_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)

def generate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {}
    for name, func in Descriptors._descList:
        try:
            descriptors[name] = func(mol)
        except:
            descriptors[name] = 0
    return descriptors

def generate_features_for_molecules(smiles_list, dft_df, corr_filtered_path):
    morgan_fps = []
    maccs_fps = []
    rdkit_descs = []
    valid_indices = []
    
    for idx, smi in enumerate(smiles_list):
        morgan_fp = generate_morgan_fingerprint(str(smi).strip())
        maccs_fp = generate_maccs_keys(str(smi).strip())
        rdkit_desc = generate_rdkit_descriptors(str(smi).strip())
        
        if morgan_fp is not None and maccs_fp is not None and rdkit_desc is not None:
            morgan_fps.append(morgan_fp)
            maccs_fps.append(maccs_fp)
            rdkit_descs.append(rdkit_desc)
            valid_indices.append(idx)
    
    morgan_df = pd.DataFrame(morgan_fps, columns=[f'morgan_{i}' for i in range(1024)])
    maccs_df = pd.DataFrame(maccs_fps, columns=[f'maccs_{i}' for i in range(167)])
    
    desc_df_all = pd.DataFrame(rdkit_descs)
    desc_df_all = desc_df_all.fillna(0)
    desc_df_all = desc_df_all.replace([np.inf, -np.inf], 0)
    
    corr_filtered = pd.read_csv(corr_filtered_path, index_col=0)
    filtered_feature_names = list(corr_filtered.columns)
    desc_df = desc_df_all[filtered_feature_names]
    
    X_molecular = pd.concat([morgan_df, maccs_df, desc_df], axis=1)
    
    dft_df_valid = dft_df.iloc[valid_indices].reset_index(drop=True)
    X_raw = pd.concat([X_molecular, dft_df_valid], axis=1)
    
    return X_raw, valid_indices

def generate_waterfall_plots(model, scaler, X_selected, X_raw, feature_names, top_indices, original_smiles, waterfall_dir):
    
    os.makedirs(waterfall_dir, exist_ok=True)

    X_train = np.load(X_SELECTED_PATH, allow_pickle=True)
    background = shap.sample(X_train, 50)

    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    all_feature_names = list(X_raw.columns)
    selected_indices = [all_feature_names.index(f) for f in feature_names]
    
    for rank, idx in enumerate(top_indices, 1):
        
        X_single = X_selected[idx:idx+1]
        shap_values = explainer.shap_values(X_single)
        
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1][0]
            base_value = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            shap_values_pos = shap_values[0, :, 1]
            base_value = explainer.expected_value[1]
        else:
            shap_values_pos = shap_values[0]
            base_value = explainer.expected_value
        
        prob = model.predict_proba(X_single)[0, 1]
        
        X_single_original = X_raw.iloc[idx].values
        X_single_original = X_single_original[selected_indices]
        X_single_rounded = np.round(X_single_original, 2)
        
        positive_indices = np.where(shap_values_pos > 0)[0]
        
        if len(positive_indices) == 0:
            print(f"    Warning: No positive SHAP values for candidate #{rank}, skipping waterfall plot")
            continue
        
        sorted_positive_indices = positive_indices[np.argsort(shap_values_pos[positive_indices])[::-1]]
        
        n_top_features = min(20, len(sorted_positive_indices))
        top_indices_shap = sorted_positive_indices[:n_top_features]
        
        top_feature_names = [feature_names[i] for i in top_indices_shap]
        top_shap_values = shap_values_pos[top_indices_shap]
        top_feature_values = X_single_rounded[top_indices_shap]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_feature_names))
        
        colors = ['#66ccff' for _ in top_shap_values]
        
        bars = ax.barh(y_pos, width=top_shap_values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        
        y_labels = []
        for i, (name, val) in enumerate(zip(top_feature_names, top_feature_values)):
            y_labels.append(f"{name} = {val:.2f}")
        ax.set_yticklabels(y_labels, fontsize=12)
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (contribution to probability)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        smiles_short = str(original_smiles[idx])[:30]
        ax.set_title(f"Top {rank}: {smiles_short}... | Prob: {prob:.4f}\nPositive Feature Contributions", 
                     fontsize=13, fontweight='bold')
        
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        waterfall_path = os.path.join(waterfall_dir, f'waterfall_plot_rank{rank:02d}.png')
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
        plt.close()

def predict_batch(input_file, output_file, model_path, scaler_path, feature_names_path, corr_filtered_path):

    df_input = pd.read_excel(input_file)
    
    required_cols = ['SMILES', 'HOMO', 'Gap', 'DM', 'is_ionic']
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Found {len(df_input)} molecules in input file")
    
    smiles_list = df_input['SMILES'].tolist()
    dft_df = df_input[['HOMO', 'Gap', 'DM', 'is_ionic']].copy()

    X_raw, valid_indices = generate_features_for_molecules(smiles_list, dft_df, corr_filtered_path)
    print(f"Generated {X_raw.shape[1]} total features")

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_raw)

    lasso_feature_names = np.load(feature_names_path, allow_pickle=True)
    lasso_feature_names = list(lasso_feature_names)
    
    all_feature_names = list(X_raw.columns)
    selected_indices = [all_feature_names.index(f) for f in lasso_feature_names]
    X_selected = X_scaled[:, selected_indices]

    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_selected)[:, 1]
    
    results = df_input.copy()
    results['passivator_probability'] = np.nan
    results['prediction'] = 'INVALID_SMILES'
    
    for i, idx in enumerate(valid_indices):
        prob = probabilities[i]
        results.loc[idx, 'passivator_probability'] = prob
        results.loc[idx, 'prediction'] = 'PASSIVATOR' if prob >= 0.5 else 'NON-PASSIVATOR'
    
    results_sorted = results.sort_values('passivator_probability', ascending=False)
    
    print(f"\nSaving results to: {output_file}")
    results_sorted.to_csv(output_file, index=False)

    print(f"Total molecules: {len(df_input)}")
    print(f"Successfully predicted: {len(valid_indices)}")
    print(f"Invalid SMILES: {len(df_input) - len(valid_indices)}")
    
    if len(valid_indices) > 0:
        top5 = results_sorted.head(5)
        for i, row in top5.iterrows():
            print(f"  {str(row['SMILES'])[:50]:<50} | Prob: {row['passivator_probability']:.4f} ({row['passivator_probability']*100:.2f}%)")
    
    if len(valid_indices) >= 10:
        top10 = results_sorted.head(10)
        top10_original_indices = top10.index.tolist()
        
        valid_to_selected = {orig_idx: sel_idx for sel_idx, orig_idx in enumerate(valid_indices)}
        top10_selected_indices = [valid_to_selected[idx] for idx in top10_original_indices if idx in valid_to_selected]
        
        original_smiles = df_input['SMILES'].tolist()
        
        generate_waterfall_plots(
            model=model,
            scaler=scaler,
            X_selected=X_selected,
            X_raw=X_raw,
            feature_names=lasso_feature_names,
            top_indices=top10_selected_indices,
            original_smiles=original_smiles,
            waterfall_dir=WATERFALL_DIR
        )
    else:
        print("\nNot enough valid molecules for force plots (need at least 10)")
    
    return results_sorted

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
    else:
        predict_batch(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
            corr_filtered_path=CORR_FILTERED_PATH
        )
