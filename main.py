import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import json
import gc
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

df = pd.read_csv('')
print(f"Dataset size: {df.shape[0]} samples")

df_train, df_test = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['class2.0']
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
print(f"Training set size: {df_train.shape[0]} samples")
print(f"Test set size: {df_test.shape[0]} samples")

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

def compute_sample_weights(deltaPCE, threshold, buffer_width=0.1, min_weight=0.3):
    distance = np.abs(deltaPCE - threshold)
    sigma = buffer_width / 2
    normalization = 1 - np.exp(-(buffer_width ** 2) / (2 * sigma ** 2))
    weights = np.where(
        distance <= buffer_width,
        min_weight + (1 - min_weight) * (1 - np.exp(-(distance ** 2) / (2 * sigma ** 2))) / normalization,
        1.0
    )
    return weights

def build_features(data):
    morgan_fps = []
    valid_indices = []
    for idx, smi in enumerate(data['smiles']):
        fp = generate_morgan_fingerprint(str(smi).strip())
        if fp is not None:
            morgan_fps.append(fp)
            valid_indices.append(idx)

    data_valid = data.iloc[valid_indices].reset_index(drop=True)
    morgan_df = pd.DataFrame(morgan_fps, columns=[f'morgan_{i}' for i in range(morgan_fps[0].shape[0])])

    maccs_fps = []
    for smi in data_valid['smiles']:
        fp = generate_maccs_keys(str(smi).strip())
        if fp is not None:
            maccs_fps.append(fp)

    maccs_df = pd.DataFrame(maccs_fps, columns=[f'maccs_{i}' for i in range(167)])
    print(f"MACCS Keys dimension: {maccs_df.shape[1]}")

    rdkit_descriptors = []
    for smi in data_valid['smiles']:
        desc = generate_rdkit_descriptors(str(smi).strip())
        if desc is not None:
            rdkit_descriptors.append(desc)

    desc_df_all = pd.DataFrame(rdkit_descriptors)
    desc_df_all = desc_df_all.fillna(0)
    desc_df_all = desc_df_all.replace([np.inf, -np.inf], 0)

    corr_filtered = pd.read_csv('d:\\demo1\\demo1\\correlation_matrix_filtered.csv', index_col=0)
    filtered_feature_names = list(corr_filtered.columns)
    desc_df = desc_df_all[filtered_feature_names]

    print(f"Original RDKit descriptors: {desc_df_all.shape[1]}")
    print(f"After correlation filtering: {desc_df.shape[1]}")

    dft_cols = ['HOMO', 'Gap', 'DM', 'is_ionic']
    dft_df = data_valid[dft_cols].copy()

    X_molecular = pd.concat([morgan_df, maccs_df, desc_df], axis=1)
    X_raw = pd.concat([X_molecular, dft_df], axis=1)

    return X_raw, data_valid


X_raw, df_train_valid = build_features(df_train)
X_test_raw, df_test_valid = build_features(df_test)

y_class18 = df_train_valid['class1.8']
y_class20 = df_train_valid['class2.0']
y_test_class18 = df_test_valid['class1.8']
y_test_class20 = df_test_valid['class2.0']

deltaPCE_train = df_train_valid['deltaPCE'].values

print(f"Total features: {X_raw.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_test_scaled = scaler.transform(X_test_raw)

lasso = LogisticRegressionCV(cv=5, random_state=42, max_iter=5000, penalty='l1', solver='saga', Cs=[0.01, 0.1, 1.0])
lasso.fit(X_scaled, y_class18)

coef_abs = np.abs(lasso.coef_[0])
nonzero_count = np.sum(coef_abs > 0)
print(f"LASSO non-zero coefficients: {nonzero_count}")

if nonzero_count > 50:
    threshold = np.percentile(coef_abs[coef_abs > 0], 50)
else:
    threshold = 0
    
selected_mask = coef_abs > threshold
X_selected = X_scaled[:, selected_mask]
X_test_selected = X_test_scaled[:, selected_mask]
feature_names = X_raw.columns[selected_mask].tolist()

print(f"Features after LASSO: {X_selected.shape[1]}")

sample_weights_18 = compute_sample_weights(deltaPCE_train, threshold=1.8, buffer_width=0.1, min_weight=0.3)
sample_weights_20 = compute_sample_weights(deltaPCE_train, threshold=2.0, buffer_width=0.1, min_weight=0.3)

print(f"class1.8 weight range: [{sample_weights_18.min():.3f}, {sample_weights_18.max():.3f}]")
print(f"class2.0 weight range: [{sample_weights_20.min():.3f}, {sample_weights_20.max():.3f}]")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_with_auc_weighted(model, X, y, cv, sample_weights=None, use_weight=True):
    acc_scores = []
    auc_scores = []
    f1_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if sample_weights is not None and use_weight:
            weights_train = sample_weights[train_idx]
            try:
                model.fit(X_train, y_train, sample_weight=weights_train)
            except (TypeError, ValueError):
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_scores.append(roc_auc_score(y_test, y_prob))
        else:
            auc_scores.append(np.nan)
    
    return np.array(acc_scores), np.array(auc_scores), np.array(f1_scores)

from xgboost import XGBClassifier

from bayes_opt import BayesianOptimization

rf_pbounds = {
    'n_estimators': (100, 500),
    'max_depth': (2, 20),
    'min_samples_split': (2, 5),
    'min_samples_leaf': (1, 5),
    'max_features': (0.3, 1.0)
}
xgb_pbounds = {
    'n_estimators': (100, 500),
    'max_depth': (2, 7),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 10),
    'gamma': (0, 5),
    'reg_alpha': (0, 1),
    'reg_lambda': (0.1, 10)
}
svm_pbounds = {
    'C': (0.1, 10),
    'gamma': (0.001, 1)
}
mlp_pbounds = {
    'hidden_layer_size_1': (50, 200),
    'hidden_layer_size_2': (20, 100),
    'alpha': (0.0001, 0.01),
    'learning_rate_init': (0.0001, 0.01)
}
kernel_names = ['RBF', 'RBF*Matern1.5', 'Matern1.5', 'Matern2.5', 'RationalQuadratic', 'Const*RBF']

y_class2 = y_class20
weights2 = sample_weights_20
print(f"Class distribution: 0={sum(y_class2==0)}, 1={sum(y_class2==1)}")

base_models_2 = {}

lr2 = LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)
lr_acc2, lr_auc2, lr_f1_2 = cv_with_auc_weighted(lr2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {lr_acc2.mean():.4f}±{lr_acc2.std():.4f}, AUC = {np.nanmean(lr_auc2):.4f}±{np.nanstd(lr_auc2):.4f}, F1 = {np.nanmean(lr_f1_2):.4f}±{np.nanstd(lr_f1_2):.4f}")
base_models_2['LR'] = (lr_acc2.mean(), lr_acc2.std(), np.nanmean(lr_auc2), np.nanstd(lr_auc2), np.nanmean(lr_f1_2), np.nanstd(lr_f1_2))

rf2 = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1)
rf_acc2, rf_auc2, rf_f1_2 = cv_with_auc_weighted(rf2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {rf_acc2.mean():.4f}±{rf_acc2.std():.4f}, AUC = {np.nanmean(rf_auc2):.4f}±{np.nanstd(rf_auc2):.4f}, F1 = {np.nanmean(rf_f1_2):.4f}±{np.nanstd(rf_f1_2):.4f}")
base_models_2['RF'] = (rf_acc2.mean(), rf_acc2.std(), np.nanmean(rf_auc2), np.nanstd(rf_auc2), np.nanmean(rf_f1_2), np.nanstd(rf_f1_2))

svm2 = SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, random_state=42)
svm_acc2, svm_auc2, svm_f1_2 = cv_with_auc_weighted(svm2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {svm_acc2.mean():.4f}±{svm_acc2.std():.4f}, AUC = {np.nanmean(svm_auc2):.4f}±{np.nanstd(svm_auc2):.4f}, F1 = {np.nanmean(svm_f1_2):.4f}±{np.nanstd(svm_f1_2):.4f}")
base_models_2['SVM'] = (svm_acc2.mean(), svm_acc2.std(), np.nanmean(svm_auc2), np.nanstd(svm_auc2), np.nanmean(svm_f1_2), np.nanstd(svm_f1_2))

mlp2 = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True)
mlp_acc2, mlp_auc2, mlp_f1_2 = cv_with_auc_weighted(mlp2, X_selected, y_class2, skf, weights2, use_weight=False)
print(f"Accuracy = {mlp_acc2.mean():.4f}±{mlp_acc2.std():.4f}, AUC = {np.nanmean(mlp_auc2):.4f}±{np.nanstd(mlp_auc2):.4f}, F1 = {np.nanmean(mlp_f1_2):.4f}±{np.nanstd(mlp_f1_2):.4f}")
base_models_2['MLP'] = (mlp_acc2.mean(), mlp_acc2.std(), np.nanmean(mlp_auc2), np.nanstd(mlp_auc2), np.nanmean(mlp_f1_2), np.nanstd(mlp_f1_2))

xgb2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_acc2, xgb_auc2, xgb_f1_2 = cv_with_auc_weighted(xgb2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {xgb_acc2.mean():.4f}±{xgb_acc2.std():.4f}, AUC = {np.nanmean(xgb_auc2):.4f}±{np.nanstd(xgb_auc2):.4f}, F1 = {np.nanmean(xgb_f1_2):.4f}±{np.nanstd(xgb_f1_2):.4f}")
base_models_2['XGB'] = (xgb_acc2.mean(), xgb_acc2.std(), np.nanmean(xgb_auc2), np.nanstd(xgb_auc2), np.nanmean(xgb_f1_2), np.nanstd(xgb_f1_2))

gpc2 = GaussianProcessClassifier(kernel=RBF(), random_state=42)
gpc_acc2, gpc_auc2, gpc_f1_2 = cv_with_auc_weighted(gpc2, X_selected, y_class2, skf, weights2, use_weight=False)
print(f"Accuracy = {gpc_acc2.mean():.4f}±{gpc_acc2.std():.4f}, AUC = {np.nanmean(gpc_auc2):.4f}±{np.nanstd(gpc_auc2):.4f}, F1 = {np.nanmean(gpc_f1_2):.4f}±{np.nanstd(gpc_f1_2):.4f}")
base_models_2['GPC'] = (gpc_acc2.mean(), gpc_acc2.std(), np.nanmean(gpc_auc2), np.nanstd(gpc_auc2), np.nanmean(gpc_f1_2), np.nanstd(gpc_f1_2))

try:
    from bayes_opt import BayesianOptimization

    rf_best2 = {
        'n_estimators': 300,
        'max_depth': 8,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 0.5,
    }
    print(f"Empirical params: n_estimators={rf_best2['n_estimators']}, max_depth={rf_best2['max_depth']}")
    
    rf_opt2 = RandomForestClassifier(
        n_estimators=rf_best2['n_estimators'],
        max_depth=rf_best2['max_depth'],
        min_samples_split=rf_best2['min_samples_split'],
        min_samples_leaf=rf_best2['min_samples_leaf'],
        max_features=rf_best2['max_features'],
        random_state=42,
        n_jobs=-1
    )
    rf_opt_acc2, rf_opt_auc2, rf_opt_f1_2 = cv_with_auc_weighted(rf_opt2, X_selected, y_class2, skf, weights2, use_weight=True)
    print(f"Accuracy = {rf_opt_acc2.mean():.4f}±{rf_opt_acc2.std():.4f}, AUC = {np.nanmean(rf_opt_auc2):.4f}±{np.nanstd(rf_opt_auc2):.4f}, F1 = {np.nanmean(rf_opt_f1_2):.4f}±{np.nanstd(rf_opt_f1_2):.4f}")
    base_models_2['RF_Bayes'] = (rf_opt_acc2.mean(), rf_opt_acc2.std(), np.nanmean(rf_opt_auc2), np.nanstd(rf_opt_auc2), np.nanmean(rf_opt_f1_2), np.nanstd(rf_opt_f1_2))

    def xgb_evaluate2(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda):
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        acc, auc, f1 = cv_with_auc_weighted(model, X_selected, y_class2, skf, weights2, use_weight=True)
        return np.nanmean(auc)
    
    xgb_optimizer2 = BayesianOptimization(f=xgb_evaluate2, pbounds=xgb_pbounds, random_state=42, verbose=0)
    xgb_optimizer2.maximize(init_points=10, n_iter=100)
    
    xgb_best2 = xgb_optimizer2.max['params']
    print(f"Best AUC: {xgb_optimizer2.max['target']:.4f}")
    
    xgb_opt2 = XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_opt_acc2, xgb_opt_auc2, xgb_opt_f1_2 = cv_with_auc_weighted(xgb_opt2, X_selected, y_class2, skf, weights2, use_weight=True)
    print(f"Optimized Accuracy = {xgb_opt_acc2.mean():.4f}±{xgb_opt_acc2.std():.4f}, AUC = {np.nanmean(xgb_opt_auc2):.4f}±{np.nanstd(xgb_opt_auc2):.4f}, F1 = {np.nanmean(xgb_opt_f1_2):.4f}±{np.nanstd(xgb_opt_f1_2):.4f}")
    base_models_2['XGB_Bayes'] = (xgb_opt_acc2.mean(), xgb_opt_acc2.std(), np.nanmean(xgb_opt_auc2), np.nanstd(xgb_opt_auc2), np.nanmean(xgb_opt_f1_2), np.nanstd(xgb_opt_f1_2))

    def svm_evaluate2(C, gamma):
        model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
        acc, auc, f1 = cv_with_auc_weighted(model, X_selected, y_class2, skf, weights2, use_weight=True)
        return np.nanmean(auc)
    
    svm_optimizer2 = BayesianOptimization(f=svm_evaluate2, pbounds=svm_pbounds, random_state=42, verbose=0)
    svm_optimizer2.maximize(init_points=10, n_iter=100)
    
    svm_best2 = svm_optimizer2.max['params']
    print(f"Best AUC: {svm_optimizer2.max['target']:.4f}")
    
    svm_opt2 = SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)
    svm_opt_acc2, svm_opt_auc2, svm_opt_f1_2 = cv_with_auc_weighted(svm_opt2, X_selected, y_class2, skf, weights2, use_weight=True)
    print(f"Optimized Accuracy = {svm_opt_acc2.mean():.4f}±{svm_opt_acc2.std():.4f}, AUC = {np.nanmean(svm_opt_auc2):.4f}±{np.nanstd(svm_opt_auc2):.4f}, F1 = {np.nanmean(svm_opt_f1_2):.4f}±{np.nanstd(svm_opt_f1_2):.4f}")
    base_models_2['SVM_Bayes'] = (svm_opt_acc2.mean(), svm_opt_acc2.std(), np.nanmean(svm_opt_auc2), np.nanstd(svm_opt_auc2), np.nanmean(svm_opt_f1_2), np.nanstd(svm_opt_f1_2))

    def mlp_evaluate2(hidden_layer_size_1, hidden_layer_size_2, alpha, learning_rate_init):
        model = MLPClassifier(
            hidden_layer_sizes=(int(hidden_layer_size_1), int(hidden_layer_size_2)),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        acc, auc, f1 = cv_with_auc_weighted(model, X_selected, y_class2, skf, weights2, use_weight=False)
        return np.nanmean(auc)
    
    mlp_optimizer2 = BayesianOptimization(f=mlp_evaluate2, pbounds=mlp_pbounds, random_state=42, verbose=0)
    mlp_optimizer2.maximize(init_points=10, n_iter=100)
    
    mlp_best2 = mlp_optimizer2.max['params']
    print(f"Best AUC: {mlp_optimizer2.max['target']:.4f}")
    
    mlp_opt2 = MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    mlp_opt_acc2, mlp_opt_auc2, mlp_opt_f1_2 = cv_with_auc_weighted(mlp_opt2, X_selected, y_class2, skf, weights2, use_weight=False)
    print(f"Optimized Accuracy = {mlp_opt_acc2.mean():.4f}±{mlp_opt_acc2.std():.4f}, AUC = {np.nanmean(mlp_opt_auc2):.4f}±{np.nanstd(mlp_opt_auc2):.4f}, F1 = {np.nanmean(mlp_opt_f1_2):.4f}±{np.nanstd(mlp_opt_f1_2):.4f}")
    base_models_2['MLP_Bayes'] = (mlp_opt_acc2.mean(), mlp_opt_acc2.std(), np.nanmean(mlp_opt_auc2), np.nanstd(mlp_opt_auc2), np.nanmean(mlp_opt_f1_2), np.nanstd(mlp_opt_f1_2))

    gpc_kernels_2 = [
        RBF(length_scale=1.0),
        RBF(length_scale=1.0) * Matern(length_scale=1.0, nu=1.5),
        Matern(length_scale=1.0, nu=1.5),
        Matern(length_scale=1.0, nu=2.5),
        RationalQuadratic(length_scale=1.0, alpha=1.0),
        ConstantKernel(1.0) * RBF(length_scale=1.0),
    ]
    
    gpc_best_score_2 = -np.inf
    gpc_best_kernel_2 = None
    gpc_best_kernel_name_2 = None
    
    for kernel, name in zip(gpc_kernels_2, kernel_names):
        gpc_temp = GaussianProcessClassifier(kernel=kernel, random_state=42)
        acc, auc, f1 = cv_with_auc_weighted(gpc_temp, X_selected, y_class2, skf, weights2, use_weight=False)
        mean_auc = np.nanmean(auc)
        print(f"  {name}: AUC = {mean_auc:.4f}")
        if mean_auc > gpc_best_score_2:
            gpc_best_score_2 = mean_auc
            gpc_best_kernel_2 = kernel
            gpc_best_kernel_name_2 = name
    
    print(f"\nBest kernel: {gpc_best_kernel_name_2}, AUC: {gpc_best_score_2:.4f}")
    
    gpc_opt2 = GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42)
    gpc_opt_acc2, gpc_opt_auc2, gpc_opt_f1_2 = cv_with_auc_weighted(gpc_opt2, X_selected, y_class2, skf, weights2, use_weight=False)
    print(f"Optimized Accuracy = {gpc_opt_acc2.mean():.4f}±{gpc_opt_acc2.std():.4f}, AUC = {np.nanmean(gpc_opt_auc2):.4f}±{np.nanstd(gpc_opt_auc2):.4f}, F1 = {np.nanmean(gpc_opt_f1_2):.4f}±{np.nanstd(gpc_opt_f1_2):.4f}")
    base_models_2['GPC_GridSearch'] = (gpc_opt_acc2.mean(), gpc_opt_acc2.std(), np.nanmean(gpc_opt_auc2), np.nanstd(gpc_opt_auc2), np.nanmean(gpc_opt_f1_2), np.nanstd(gpc_opt_f1_2))
    
except ImportError:
    print("\nNote: bayes_opt not installed, skipping Bayesian optimization")
    rf_best2 = {'n_estimators': 300, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 1}
    xgb_best2 = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
    svm_best2 = {'C': 1.0, 'gamma': 0.1}
    mlp_best2 = {'hidden_layer_size_1': 100, 'hidden_layer_size_2': 50, 'alpha': 0.001, 'learning_rate_init': 0.001}
    gpc_best_kernel_2 = RBF()

voting_clf2 = VotingClassifier(
    estimators=[
        ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
        ('rf', RandomForestClassifier(
            n_estimators=int(rf_best2['n_estimators']),
            max_depth=int(rf_best2['max_depth']),
            min_samples_split=int(rf_best2['min_samples_split']),
            min_samples_leaf=int(rf_best2['min_samples_leaf']),
            max_features=rf_best2['max_features'],
            random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(
            n_estimators=int(xgb_best2['n_estimators']),
            max_depth=int(xgb_best2['max_depth']),
            learning_rate=xgb_best2['learning_rate'],
            subsample=xgb_best2['subsample'],
            colsample_bytree=xgb_best2['colsample_bytree'],
            min_child_weight=xgb_best2['min_child_weight'],
            gamma=xgb_best2['gamma'],
            reg_alpha=xgb_best2['reg_alpha'],
            reg_lambda=xgb_best2['reg_lambda'],
            random_state=42, n_jobs=-1, eval_metric='logloss')),
        ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
            alpha=mlp_best2['alpha'],
            learning_rate_init=mlp_best2['learning_rate_init'],
            max_iter=500, random_state=42, early_stopping=True)),
        ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
    ],
    voting='soft'
)
soft_acc2, soft_auc2, soft_f1_2 = cv_with_auc_weighted(voting_clf2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {soft_acc2.mean():.4f}±{soft_acc2.std():.4f}, AUC = {np.nanmean(soft_auc2):.4f}±{np.nanstd(soft_auc2):.4f}, F1 = {np.nanmean(soft_f1_2):.4f}±{np.nanstd(soft_f1_2):.4f}")
base_models_2['Soft_Voting'] = (soft_acc2.mean(), soft_acc2.std(), np.nanmean(soft_auc2), np.nanstd(soft_auc2), np.nanmean(soft_f1_2), np.nanstd(soft_f1_2))

voting_clf2_v2 = VotingClassifier(
    estimators=[
        ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42)),
        ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
        ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
            alpha=mlp_best2['alpha'],
            learning_rate_init=mlp_best2['learning_rate_init'],
            max_iter=500, random_state=42, early_stopping=True)),
        ('xgb', XGBClassifier(
            n_estimators=int(xgb_best2['n_estimators']),
            max_depth=int(xgb_best2['max_depth']),
            learning_rate=xgb_best2['learning_rate'],
            subsample=xgb_best2['subsample'],
            colsample_bytree=xgb_best2['colsample_bytree'],
            min_child_weight=xgb_best2['min_child_weight'],
            gamma=xgb_best2['gamma'],
            reg_alpha=xgb_best2['reg_alpha'],
            reg_lambda=xgb_best2['reg_lambda'],
            random_state=42, n_jobs=-1, eval_metric='logloss')),
    ],
    voting='soft'
)
soft_acc2_v2, soft_auc2_v2, soft_f1_2_v2 = cv_with_auc_weighted(voting_clf2_v2, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {soft_acc2_v2.mean():.4f}±{soft_acc2_v2.std():.4f}, AUC = {np.nanmean(soft_auc2_v2):.4f}±{np.nanstd(soft_auc2_v2):.4f}, F1 = {np.nanmean(soft_f1_2_v2):.4f}±{np.nanstd(soft_f1_2_v2):.4f}")
base_models_2['Soft_Voting_V2'] = (soft_acc2_v2.mean(), soft_acc2_v2.std(), np.nanmean(soft_auc2_v2), np.nanstd(soft_auc2_v2), np.nanmean(soft_f1_2_v2), np.nanstd(soft_f1_2_v2))

from sklearn.ensemble import StackingClassifier
estimators_v1_1 = [
    ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500, random_state=42, early_stopping=True)),
    ('xgb', XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss')),
    ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
]
stacking_v1_1 = StackingClassifier(
    estimators=estimators_v1_1,
    final_estimator=LogisticRegressionCV(cv=5, random_state=42, max_iter=1000),
    cv=5,
    n_jobs=-1
)
stack_acc_v1_1, stack_auc_v1_1, stack_f1_v1_1 = cv_with_auc_weighted(stacking_v1_1, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {stack_acc_v1_1.mean():.4f}±{stack_acc_v1_1.std():.4f}, AUC = {np.nanmean(stack_auc_v1_1):.4f}±{np.nanstd(stack_auc_v1_1):.4f}, F1 = {np.nanmean(stack_f1_v1_1):.4f}±{np.nanstd(stack_f1_v1_1):.4f}")
base_models_2['Stacking_V1.1'] = (stack_acc_v1_1.mean(), stack_acc_v1_1.std(), np.nanmean(stack_auc_v1_1), np.nanstd(stack_auc_v1_1), np.nanmean(stack_f1_v1_1), np.nanstd(stack_f1_v1_1))

estimators_v2_1 = [
    ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
    ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500, random_state=42, early_stopping=True)),
    ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
]
stacking_v2_1 = StackingClassifier(
    estimators=estimators_v2_1,
    final_estimator=XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
    cv=5,
    n_jobs=-1
)
stack_acc_v2_1, stack_auc_v2_1, stack_f1_v2_1 = cv_with_auc_weighted(stacking_v2_1, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {stack_acc_v2_1.mean():.4f}±{stack_acc_v2_1.std():.4f}, AUC = {np.nanmean(stack_auc_v2_1):.4f}±{np.nanstd(stack_auc_v2_1):.4f}, F1 = {np.nanmean(stack_f1_v2_1):.4f}±{np.nanstd(stack_f1_v2_1):.4f}")
base_models_2['Stacking_V2.1'] = (stack_acc_v2_1.mean(), stack_acc_v2_1.std(), np.nanmean(stack_auc_v2_1), np.nanstd(stack_auc_v2_1), np.nanmean(stack_f1_v2_1), np.nanstd(stack_f1_v2_1))

estimators_v4_1 = [
    ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
    ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss')),
    ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
]
stacking_v4_1 = StackingClassifier(
    estimators=estimators_v4_1,
    final_estimator=MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True),
    cv=5,
    n_jobs=-1
)
stack_acc_v4_1, stack_auc_v4_1, stack_f1_v4_1 = cv_with_auc_weighted(stacking_v4_1, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {stack_acc_v4_1.mean():.4f}±{stack_acc_v4_1.std():.4f}, AUC = {np.nanmean(stack_auc_v4_1):.4f}±{np.nanstd(stack_auc_v4_1):.4f}, F1 = {np.nanmean(stack_f1_v4_1):.4f}±{np.nanstd(stack_f1_v4_1):.4f}")
base_models_2['Stacking_V4.1'] = (stack_acc_v4_1.mean(), stack_acc_v4_1.std(), np.nanmean(stack_auc_v4_1), np.nanstd(stack_auc_v4_1), np.nanmean(stack_f1_v4_1), np.nanstd(stack_f1_v4_1))

estimators_v5_1 = [
    ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
    ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500, random_state=42, early_stopping=True)),
    ('xgb', XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss'))
]
stacking_v5_1 = StackingClassifier(
    estimators=estimators_v5_1,
    final_estimator=GaussianProcessClassifier(kernel=RBF(), random_state=42),
    cv=5,
    n_jobs=-1
)
stack_acc_v5_1, stack_auc_v5_1, stack_f1_v5_1 = cv_with_auc_weighted(stacking_v5_1, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {stack_acc_v5_1.mean():.4f}±{stack_acc_v5_1.std():.4f}, AUC = {np.nanmean(stack_auc_v5_1):.4f}±{np.nanstd(stack_auc_v5_1):.4f}, F1 = {np.nanmean(stack_f1_v5_1):.4f}±{np.nanstd(stack_f1_v5_1):.4f}")
base_models_2['Stacking_V5.1'] = (stack_acc_v5_1.mean(), stack_acc_v5_1.std(), np.nanmean(stack_auc_v5_1), np.nanstd(stack_auc_v5_1), np.nanmean(stack_f1_v5_1), np.nanstd(stack_f1_v5_1))

estimators_v6_1 = [
    ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500, random_state=42, early_stopping=True)),
    ('xgb', XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss')),
    ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
]
stacking_v6_1 = StackingClassifier(
    estimators=estimators_v6_1,
    final_estimator=SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, random_state=42),
    cv=5,
    n_jobs=-1
)
stack_acc_v6_1, stack_auc_v6_1, stack_f1_v6_1 = cv_with_auc_weighted(stacking_v6_1, X_selected, y_class2, skf, weights2, use_weight=True)
print(f"Accuracy = {stack_acc_v6_1.mean():.4f}±{stack_acc_v6_1.std():.4f}, AUC = {np.nanmean(stack_auc_v6_1):.4f}±{np.nanstd(stack_auc_v6_1):.4f}, F1 = {np.nanmean(stack_f1_v6_1):.4f}±{np.nanstd(stack_f1_v6_1):.4f}")
base_models_2['Stacking_V6.1'] = (stack_acc_v6_1.mean(), stack_acc_v6_1.std(), np.nanmean(stack_auc_v6_1), np.nanstd(stack_auc_v6_1), np.nanmean(stack_f1_v6_1), np.nanstd(stack_f1_v6_1))

print("\n" + "="*60)
print("class2.0 Final Ranking (by AUC-ROC):")
print("="*60)
sorted_models_2 = sorted(base_models_2.items(), key=lambda x: x[1][2], reverse=True)
for i, (name, (acc_mean, acc_std, auc_mean, auc_std, f1_mean, f1_std)) in enumerate(sorted_models_2, 1):
    print(f"{i}. {name}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")

print("\n" + "="*60)
print("Saving Top 3 Models for class2.0 SHAP Analysis...")
print("="*60)

top3_models_2 = sorted_models_2[:3]
top3_model_names = [name for name, _ in top3_models_2]
print(f"\nTop 3 models: {top3_model_names}")

model_builders = {
    'LR': lambda: LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000),
    'RF': lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1),
    'SVM': lambda: SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, random_state=42),
    'MLP': lambda: MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True),
    'XGB': lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
    'GPC': lambda: GaussianProcessClassifier(kernel=RBF(), random_state=42),
    'RF_Bayes': lambda: RandomForestClassifier(
        n_estimators=int(rf_best2['n_estimators']),
        max_depth=int(rf_best2['max_depth']),
        min_samples_split=int(rf_best2['min_samples_split']),
        min_samples_leaf=int(rf_best2['min_samples_leaf']),
        max_features=rf_best2['max_features'],
        random_state=42, n_jobs=-1),
    'XGB_Bayes': lambda: XGBClassifier(
        n_estimators=int(xgb_best2['n_estimators']),
        max_depth=int(xgb_best2['max_depth']),
        learning_rate=xgb_best2['learning_rate'],
        subsample=xgb_best2['subsample'],
        colsample_bytree=xgb_best2['colsample_bytree'],
        min_child_weight=xgb_best2['min_child_weight'],
        gamma=xgb_best2['gamma'],
        reg_alpha=xgb_best2['reg_alpha'],
        reg_lambda=xgb_best2['reg_lambda'],
        random_state=42, n_jobs=-1, eval_metric='logloss'),
    'SVM_Bayes': lambda: SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42),
    'MLP_Bayes': lambda: MLPClassifier(
        hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
        alpha=mlp_best2['alpha'],
        learning_rate_init=mlp_best2['learning_rate_init'],
        max_iter=500, random_state=42, early_stopping=True),
    'GPC_GridSearch': lambda: GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42),
    'Soft_Voting': lambda: VotingClassifier(
        estimators=[
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('rf', RandomForestClassifier(
                n_estimators=int(rf_best2['n_estimators']),
                max_depth=int(rf_best2['max_depth']),
                min_samples_split=int(rf_best2['min_samples_split']),
                min_samples_leaf=int(rf_best2['min_samples_leaf']),
                max_features=rf_best2['max_features'],
                random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(
                n_estimators=int(xgb_best2['n_estimators']),
                max_depth=int(xgb_best2['max_depth']),
                learning_rate=xgb_best2['learning_rate'],
                subsample=xgb_best2['subsample'],
                colsample_bytree=xgb_best2['colsample_bytree'],
                min_child_weight=xgb_best2['min_child_weight'],
                gamma=xgb_best2['gamma'],
                reg_alpha=xgb_best2['reg_alpha'],
                reg_lambda=xgb_best2['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True)),
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
        ],
        voting='soft'),
    'Soft_Voting_V2': lambda: VotingClassifier(
        estimators=[
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42)),
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True))
        ],
        voting='soft'),
    'Stacking_V1.1': lambda: StackingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True)),
            ('xgb', XGBClassifier(
                n_estimators=int(xgb_best2['n_estimators']),
                max_depth=int(xgb_best2['max_depth']),
                learning_rate=xgb_best2['learning_rate'],
                subsample=xgb_best2['subsample'],
                colsample_bytree=xgb_best2['colsample_bytree'],
                min_child_weight=xgb_best2['min_child_weight'],
                gamma=xgb_best2['gamma'],
                reg_alpha=xgb_best2['reg_alpha'],
                reg_lambda=xgb_best2['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
        ],
        final_estimator=LogisticRegressionCV(cv=5, random_state=42, max_iter=1000),
        cv=5, n_jobs=-1),
    'Stacking_V2.1': lambda: StackingClassifier(
        estimators=[
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True)),
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
        ],
        final_estimator=XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
        cv=5, n_jobs=-1),
    'Stacking_V4.1': lambda: StackingClassifier(
        estimators=[
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('xgb', XGBClassifier(
                n_estimators=int(xgb_best2['n_estimators']),
                max_depth=int(xgb_best2['max_depth']),
                learning_rate=xgb_best2['learning_rate'],
                subsample=xgb_best2['subsample'],
                colsample_bytree=xgb_best2['colsample_bytree'],
                min_child_weight=xgb_best2['min_child_weight'],
                gamma=xgb_best2['gamma'],
                reg_alpha=xgb_best2['reg_alpha'],
                reg_lambda=xgb_best2['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
        ],
        final_estimator=MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True),
        cv=5, n_jobs=-1),
    'Stacking_V5.1': lambda: StackingClassifier(
        estimators=[
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('svm', SVC(kernel='rbf', C=svm_best2['C'], gamma=svm_best2['gamma'], probability=True, random_state=42)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True)),
            ('xgb', XGBClassifier(
                n_estimators=int(xgb_best2['n_estimators']),
                max_depth=int(xgb_best2['max_depth']),
                learning_rate=xgb_best2['learning_rate'],
                subsample=xgb_best2['subsample'],
                colsample_bytree=xgb_best2['colsample_bytree'],
                min_child_weight=xgb_best2['min_child_weight'],
                gamma=xgb_best2['gamma'],
                reg_alpha=xgb_best2['reg_alpha'],
                reg_lambda=xgb_best2['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss'))
        ],
        final_estimator=GaussianProcessClassifier(kernel=RBF(), random_state=42),
        cv=5, n_jobs=-1),
    'Stacking_V6.1': lambda: StackingClassifier(
        estimators=[
            ('lr', LogisticRegressionCV(cv=skf, random_state=42, max_iter=2000)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(int(mlp_best2['hidden_layer_size_1']), int(mlp_best2['hidden_layer_size_2'])),
                alpha=mlp_best2['alpha'],
                learning_rate_init=mlp_best2['learning_rate_init'],
                max_iter=500, random_state=42, early_stopping=True)),
            ('xgb', XGBClassifier(
                n_estimators=int(xgb_best2['n_estimators']),
                max_depth=int(xgb_best2['max_depth']),
                learning_rate=xgb_best2['learning_rate'],
                subsample=xgb_best2['subsample'],
                colsample_bytree=xgb_best2['colsample_bytree'],
                min_child_weight=xgb_best2['min_child_weight'],
                gamma=xgb_best2['gamma'],
                reg_alpha=xgb_best2['reg_alpha'],
                reg_lambda=xgb_best2['reg_lambda'],
                random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('gpc', GaussianProcessClassifier(kernel=gpc_best_kernel_2, random_state=42))
        ],
        final_estimator=SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, random_state=42),
        cv=5, n_jobs=-1),
}

for i, (model_name, (acc_mean, acc_std, auc_mean, auc_std, f1_mean, f1_std)) in enumerate(top3_models_2, 1):
    print(f"\n[{i}/3] Training and saving {model_name}...")
    model = model_builders[model_name]()
    model.fit(X_selected, y_class2)
    model_path = f'd:\\demo1\\demo1\\top3_model_class2.0_{i}.pkl'
    joblib.dump(model, model_path)
    print(f"  Saved: {model_path}")
    print(f"  AUC: {auc_mean:.4f}±{auc_std:.4f}, F1: {f1_mean:.4f}±{f1_std:.4f}")

top3_info = {
    'models': [
        {'rank': i+1, 'name': name, 'acc_mean': float(acc_mean), 'acc_std': float(acc_std), 
         'auc_mean': float(auc_mean), 'auc_std': float(auc_std), 
         'f1_mean': float(f1_mean), 'f1_std': float(f1_std)}
        for i, (name, (acc_mean, acc_std, auc_mean, auc_std, f1_mean, f1_std)) in enumerate(top3_models_2)
    ]
}
with open('d:\\demo1\\demo1\\top3_models_class2.0_info.json', 'w') as f:
    json.dump(top3_info, f, indent=2)


joblib.dump(scaler, 'd:\\demo1\\demo1\\scaler_lite.pkl')

feature_names_array = np.array(feature_names)
np.save('d:\\demo1\\demo1\\feature_names_lite.npy', feature_names_array)

original_feature_names = np.array(X_raw.columns.tolist())
np.save('d:\\demo1\\demo1\\original_feature_names_lite.npy', original_feature_names)
np.save('d:\\demo1\\demo1\\selected_mask_lite.npy', selected_mask)
np.save('d:\\demo1\\demo1\\X_selected_lite.npy', X_selected)
y_class18_arr = y_class18.values
y_class20_arr = y_class20.values
np.save('d:\\demo1\\demo1\\y_class18_lite.npy', y_class18_arr)
np.save('d:\\demo1\\demo1\\y_class20_lite.npy', y_class20_arr)

print("\n" + "="*60)
print("Test Set Evaluation (10% Hold-out Set)")
print("="*60)

def fit_and_test(model, X_train, y_train, sample_weights, X_test, y_test):
    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except (TypeError, ValueError):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auc = np.nan
    return acc, auc, f1

test_results = {}
for model_name, _ in sorted_models_2:
    model = model_builders[model_name]()
    acc, auc, f1 = fit_and_test(model, X_selected, y_class2, weights2, X_test_selected, y_test_class20)
    test_results[model_name] = (acc, auc, f1)

print(f"Test set size: {X_test_selected.shape[0]} samples")
print("\nclass2.0 Test Ranking (by AUC-ROC):")
for i, (name, (acc, auc, f1)) in enumerate(sorted(test_results.items(), key=lambda x: x[1][1], reverse=True), 1):
    print(f"{i}. {name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")

gc.collect()
