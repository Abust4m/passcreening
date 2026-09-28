import numpy as np
import pandas as pd
import shap
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

scaler = joblib.load('d:\\demo1\\demo1\\scaler_lite.pkl')
feature_names = np.load('d:\\demo1\\demo1\\feature_names_lite.npy', allow_pickle=True).tolist()
original_feature_names = np.load('d:\\demo1\\demo1\\original_feature_names_lite.npy', allow_pickle=True).tolist()
selected_mask = np.load('d:\\demo1\\demo1\\selected_mask_lite.npy')
X_selected = np.load('d:\\demo1\\demo1\\X_selected_lite.npy')
y_class20 = np.load('d:\\demo1\\demo1\\y_class20_lite.npy')

with open('d:\\demo1\\demo1\\top3_models_class2.0_info.json', 'r') as f:
    top3_info = json.load(f)

print(f"\nFeature dimension: {X_selected.shape}")
print(f"Sample count: {X_selected.shape[0]}")

for model_info in top3_info['models']:
    print(f"  {model_info['rank']}. {model_info['name']}: AUC={model_info['auc_mean']:.4f}, F1={model_info['f1_mean']:.4f}")

models = []
model_names = []
for i in range(1, 4):
    model_path = f'd:\\demo1\\demo1\\top3_model_class2.0_{i}.pkl'
    model = joblib.load(model_path)
    models.append(model)
    model_names.append(top3_info['models'][i-1]['name'])
    print(f"  Loaded: {model_path}")

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X_selected, y_class20))

X_train = X_selected[train_idx]
X_test = X_selected[test_idx]

def get_shap_explainer(model, X_train, model_name):
    if 'LR' in model_name or 'LogisticRegression' in str(type(model)):
        return shap.LinearExplainer(model, X_train), 'linear'
    elif hasattr(model, 'predict_proba'):
        return shap.KernelExplainer(model.predict_proba, X_train[:50]), 'kernel'
    else:
        return shap.KernelExplainer(model.predict, X_train[:50]), 'kernel'

def extract_shap_values(shap_values, model_type):
    if model_type == 'linear':
        if isinstance(shap_values, list):
            return shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return shap_values
    else:
        if isinstance(shap_values, list):
            return shap_values[1]
        else:
            if len(shap_values.shape) == 3:
                return shap_values[:, :, 1]
            return shap_values

all_shap_results = []

for i, (model, model_name) in enumerate(zip(models, model_names)):
    
    explainer, explainer_type = get_shap_explainer(model, X_train, model_name)
    
    if explainer_type == 'kernel':
        shap_values = explainer.shap_values(X_test, nsamples=100)
    else:
        shap_values = explainer.shap_values(X_test)
    
    shap_values_class1 = extract_shap_values(shap_values, explainer_type)
    
    if len(shap_values_class1.shape) > 1:
        shap_importance = np.abs(shap_values_class1).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values_class1).flatten()
    
    shap_importance = np.array(shap_importance).flatten()
    
    n_features = min(len(feature_names), len(shap_importance))
    shap_df = pd.DataFrame({
        'feature': feature_names[:n_features],
        'shap_importance': shap_importance[:n_features]
    }).sort_values('shap_importance', ascending=False)
    
    print(f"\n  Top 10 SHAP Features:")
    for idx, row in shap_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['shap_importance']:.6f}")
    
    shap_df.to_csv(f'd:\\demo1\\demo1\\shap_importance_class2.0_model{i+1}.csv', index=False)
    
    all_shap_results.append({
        'model_name': model_name,
        'shap_values': shap_values_class1,
        'shap_df': shap_df,
        'explainer_type': explainer_type
    })

y_test = y_class20[test_idx]

for i, (model, model_name) in enumerate(zip(models, model_names)):
    print(f"\n>>> Model {i+1}: {model_name}")
    print("-"*40)
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    cm_ratio = cm.astype('float') / cm.sum()
    
    print(f"  Confusion Matrix :")
    print(f"    TN={tn/total:.4f}  FP={fp/total:.4f}")
    print(f"    FN={fn/total:.4f}  TP={tp/total:.4f}")
    print(f"  Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    annot = np.array([[f'{cm_ratio[0,0]*100:.1f}%', f'{cm_ratio[0,1]*100:.1f}%'],
                      [f'{cm_ratio[1,0]*100:.1f}%', f'{cm_ratio[1,1]*100:.1f}%']])
    
    sns.heatmap(cm_ratio, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Proportion'},
                ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - class2.0\n{model_name}', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)
    
    textstr = f'Accuracy:  {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1 Score:  {f1:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy')
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props, family='monospace')
    
    safe_model_name = model_name.replace('.', '_').replace(' ', '_')
    fig_path = f'd:\\demo1\\demo1\\confusion_matrix_class2.0_model{i+1}_{safe_model_name}.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

for idx, result in enumerate(all_shap_results):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    
    ax1 = axes[0]
    top10_df = result['shap_df'].head(10).sort_values('shap_importance', ascending=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top10_df)))
    
    ax1.barh(range(len(top10_df)), top10_df['shap_importance'].values, color=colors)
    ax1.set_yticks(range(len(top10_df)))
    ax1.set_yticklabels(top10_df['feature'].values, fontsize=9)
    ax1.set_xlabel('SHAP Importance', fontsize=11)
    ax1.set_title(f'Top 10 Feature Importance', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(top10_df['shap_importance'].values):
        ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=6)
    
    ax2 = axes[1]
    plt.sca(ax2)
    shap.summary_plot(
        result['shap_values'], 
        X_test, 
        feature_names=feature_names,
        show=False, 
        max_display=10
    )
    ax2.set_title(f'SHAP Summary Plot', fontsize=12)
    ax2.set_xlabel('SHAP Value', fontsize=11)
    
    plt.tight_layout()
    
    safe_model_name = result['model_name'].replace('.', '_').replace(' ', '_')
    fig_path = f'd:\\demo1\\demo1\\shap_analysis_class2.0_model{idx+1}_{safe_model_name}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close(fig)

for i, result in enumerate(all_shap_results):
    np.save(f'd:\\demo1\\demo1\\shap_values_class2.0_model{i+1}.npy', result['shap_values'])
    print(f"  Saved: shap_values_class2.0_model{i+1}.npy")

np.save('d:\\demo1\\demo1\\X_test_class2.0.npy', X_test)
print(f"  Saved: X_test_class2.0.npy")

top_n = 20
all_features = set()
for result in all_shap_results:
    top_features = result['shap_df'].head(top_n)['feature'].tolist()
    all_features.update(top_features)

feature_rankings = {}
for feat in all_features:
    ranks = []
    for result in all_shap_results:
        df = result['shap_df']
        if feat in df['feature'].values:
            rank = df[df['feature'] == feat].index[0] + 1
        else:
            rank = len(df) + 1
        ranks.append(rank)
    feature_rankings[feat] = np.mean(ranks)

sorted_features = sorted(feature_rankings.items(), key=lambda x: x[1])
consensus_features = [f[0] for f in sorted_features[:top_n]]

consensus_df = pd.DataFrame({
    'feature': consensus_features,
    'avg_rank': [feature_rankings[f] for f in consensus_features]
})
consensus_df.to_csv('d:\\demo1\\demo1\\shap_consensus_features_class2.0.csv', index=False)

best_model_result = all_shap_results[1]
best_model_name = best_model_result['model_name']

shap_values = best_model_result['shap_values']

for j, feature_name in enumerate(feature_names):
    if 'morgan' in feature_name or 'maccs' in feature_name:
        continue
    
    if len(shap_values.shape) == 2:
        feature_shap = shap_values[:, j]
    else:
        feature_shap = shap_values[j]
    
    n_samples = X_test.shape[0]
    n_original_features = len(original_feature_names)
    temp_array = np.zeros((n_samples, n_original_features))
    
    temp_array[:, selected_mask] = X_test
    
    temp_array_original = scaler.inverse_transform(temp_array)
    
    original_feature_idx = original_feature_names.index(feature_name)
    feature_values = temp_array_original[:, original_feature_idx]
    
    plt.figure(figsize=(5, 5))
    
    mask_neg = feature_shap <= 0
    mask_pos = feature_shap > 0
    
    if np.any(mask_neg):
        plt.scatter(feature_values[mask_neg], feature_shap[mask_neg], 
                   c='gray', alpha=0.7, s=50, label='SHAP < 0')
    if np.any(mask_pos):
        plt.scatter(feature_values[mask_pos], feature_shap[mask_pos], 
                   c='#00CED1', alpha=0.7, s=50, label='SHAP >= 0')
    
    plt.xlabel(f'Feature Value', fontsize=12, fontweight='bold')
    plt.ylabel('SHAP Value', fontsize=12, fontweight='bold')
    plt.title(f'Feature Name: {feature_name}', fontsize=14, fontweight='bold')
    
    safe_feature_name = feature_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_model_name = best_model_name.replace('.', '_').replace(' ', '_')
    fig_path = f'd:\\demo1\\demo1\\shap_scatter_best_model_{safe_model_name}_{safe_feature_name}.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
