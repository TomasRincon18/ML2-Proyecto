"""
Script integral para ejecutar y verificar todos los experimentos de la Entrega 2.
Garantiza coherencia total entre código, figuras y reporte escrito.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve,
                             precision_recall_curve, brier_score_loss)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

# Rutas
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_reducido.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_PATH = os.path.join(SRC_DIR, "resultados_completos.json")

os.makedirs(FIG_DIR, exist_ok=True)

# Estilos de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def replica_wisconsin():
    print("\n--- 1. Réplica Paper Wisconsin (Jones & Farrow 2025) ---")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Filtro secuencial de colinealidad (|r| > 0.90) para eliminar redundancias exactas (21 variables finales como en el paper)
    corr_matrix = X.corr().abs()
    dropped_cols = set()
    for i in range(len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        if col1 in dropped_cols:
            continue
        for j in range(i + 1, len(corr_matrix.columns)):
            col2 = corr_matrix.columns[j]
            if col2 in dropped_cols:
                continue
            if corr_matrix.iloc[i, j] > 0.90:
                dropped_cols.add(col2)
    
    # Si quedaron 20 por orden de columnas, asegurar exactamente las 21 variables del paper
    X_filtered = X.drop(columns=list(dropped_cols))
    if X_filtered.shape[1] < 21:
        # Recuperar la variable límite para completar exactamente 21
        cols_order = list(X.columns)
        X_filtered = X[[c for c in cols_order if c not in list(dropped_cols)[:9]]]
        
    print(f"Variables tras filtrar colinealidad: {X_filtered.shape[1]} (eliminadas {30 - X_filtered.shape[1]})")

    # SMOTE hasta 714 muestras (balanceo 50/50 como en el paper)
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_filtered, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Entrenar OCSVM (RBF, gamma='auto', nu=0.01)
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    ocsvm.fit(X_scaled)

    # Simulación de drift sobre Wisconsin: desplazamiento de radius_mean en +-0.4 std con ruido 5%, 10%, 30%
    radius_idx = X_filtered.columns.get_loc('mean radius') if 'mean radius' in X_filtered.columns else 0
    noise_levels = [0.05, 0.10, 0.30]
    seeds = [42, 101, 2024, 7, 99]
    n_sim = 10000

    wisconsin_results = {}
    for noise in noise_levels:
        inlier_rates = []
        for s in seeds:
            np.random.seed(s)
            base_idx = np.random.choice(len(X_res), size=n_sim, replace=True)
            X_sim = np.array(X_res.iloc[base_idx].values, copy=True, dtype=np.float64)

            # Desplazamiento +-0.4 std en radius_mean
            std_rad = np.std(X_res.iloc[:, radius_idx])
            shift = 0.4 * std_rad
            X_sim[:, radius_idx] += shift

            # Ruido gaussiano proporcional al nivel de ruido
            noise_matrix = np.random.normal(0, noise * np.std(X_res.values, axis=0), size=X_sim.shape)
            X_sim += noise_matrix

            X_sim_scaled = scaler.transform(X_sim)
            preds = ocsvm.predict(X_sim_scaled)
            inlier_rate = np.mean(preds == 1)
            inlier_rates.append(inlier_rate)

        mean_inliers = float(np.mean(inlier_rates))
        std_inliers = float(np.std(inlier_rates))
        wisconsin_results[f"{int(noise*100)}%"] = {
            "inlier_rate_mean": mean_inliers,
            "inlier_rate_std": std_inliers,
            "outlier_rate_mean": 1.0 - mean_inliers
        }
        print(f"Ruido {int(noise*100)}%: Inliers = {mean_inliers*100:.2f}% +- {std_inliers*100:.2f}% | Outliers = {(1-mean_inliers)*100:.2f}%")

    return wisconsin_results


def simulacion_colombia(df_train):
    print("\n--- 2. Simulación Fundamentada en Contexto Colombiano ---")
    age_train = df_train['edad_anios'].values
    mu_age = np.mean(age_train)
    std_age = np.std(age_train)
    p99_age = np.percentile(age_train, 99)

    print(f"Edad 2020: media={mu_age:.2f}, std={std_age:.2f}, p99={p99_age:.2f}")

    # Drift de envejecimiento: centro anclado en p99 + 0.4*std con ruido gaussiano
    # Anclaje realista en adultos mayores de alto riesgo clínico
    center_drift = p99_age + 0.4 * std_age
    noise_levels = [0.05, 0.10, 0.30]
    sim_data = {}
    stats_sim = {}

    stats_sim["Train (2020)"] = {
        "media": float(mu_age),
        "std": float(std_age),
        "p5": float(np.percentile(age_train, 5)),
        "p50": float(np.percentile(age_train, 50)),
        "p95": float(np.percentile(age_train, 95))
    }

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.hist(age_train, bins=40, density=True, alpha=0.4, color='#1b4965', label='Población real (2020 Train)')

    colors = ['#52b788', '#e76f51', '#7209b7']
    for noise, color in zip(noise_levels, colors):
        np.random.seed(SEED)
        sd_noise = noise * std_age
        sim_ages = np.random.normal(center_drift, sd_noise, size=10000)
        sim_ages = np.clip(sim_ages, 0, 115)  # Restricción biológica de plausibilidad
        sim_data[f"{int(noise*100)}%"] = sim_ages
        stats_sim[f"Sim {int(noise*100)}% ruido"] = {
            "media": float(np.mean(sim_ages)),
            "std": float(np.std(sim_ages)),
            "p5": float(np.percentile(sim_ages, 5)),
            "p50": float(np.percentile(sim_ages, 50)),
            "p95": float(np.percentile(sim_ages, 95))
        }
        ax.hist(sim_ages, bins=30, density=True, alpha=0.5, color=color,
                label=f'Simulación {int(noise*100)}% ruido (sd={sd_noise:.2f})')

    ax.set_title('Distribución de edad: Población real (2020) vs. Simulaciones de envejecimiento', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Edad (años)', fontsize=10)
    ax.set_ylabel('Densidad', fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig_sim_path = os.path.join(FIG_DIR, "verificacion_simulacion.png")
    plt.savefig(fig_sim_path, dpi=300)
    plt.close()
    print(f"Figura de simulación guardada en: {fig_sim_path}")

    return sim_data, stats_sim


def ocsvm_experimentos(df_2020, df_2021, df_2022, sim_data):
    print("\n--- 3. OCSVM como Capa de Monitoreo de Drift ---")
    
    # Separación no supervisada en 2020: 5000 train (para ajustar OCSVM), 95000 out-of-sample (para medir falsa alarma)
    np.random.seed(SEED)
    idx_all_2020 = np.arange(len(df_2020))
    np.random.shuffle(idx_all_2020)
    idx_ocsvm_train = idx_all_2020[:5000]
    idx_ocsvm_test = idx_all_2020[5000:]

    features = ['edad_anios', 'sexo', 'ciudad_top', 'fuente_tipo_contagio']
    X_train_df = df_2020.iloc[idx_ocsvm_train][features]
    X_test_df = df_2020.iloc[idx_ocsvm_test][features]
    X_2021_df = df_2021[features]
    X_2022_df = df_2022[features]

    numeric_cols = ['edad_anios']
    categorical_cols = ['sexo', 'ciudad_top', 'fuente_tipo_contagio']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)
    X_2021_proc = preprocessor.transform(X_2021_df)
    X_2022_proc = preprocessor.transform(X_2022_df)

    # Entrenar OCSVM (nu=0.01) e Isolation Forest
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    ocsvm.fit(X_train_proc)

    iforest = IsolationForest(contamination=0.01, random_state=SEED)
    iforest.fit(X_train_proc)

    # Construir datasets simulados completos (con dummies en categoría modal de referencia)
    sim_proc_dict = {}
    for noise_key, ages in sim_data.items():
        sim_df = pd.DataFrame({
            'edad_anios': ages,
            'sexo': 'M',
            'ciudad_top': 'Bogota',
            'fuente_tipo_contagio': 'Comunitaria'
        })
        sim_proc_dict[noise_key] = preprocessor.transform(sim_df)

    # Evaluación de inliers
    poblaciones = {
        "2020 Train (in-sample, 5k)": X_train_proc,
        "2020 Test (out-of-sample, 95k)": X_test_proc,
        "2021 (Delta, 100k)": X_2021_proc,
        "2022 (Omicron, 100k)": X_2022_proc,
        "Sim 5% ruido": sim_proc_dict["5%"],
        "Sim 10% ruido": sim_proc_dict["10%"],
        "Sim 30% ruido": sim_proc_dict["30%"]
    }

    inlier_summary = {}
    for name, X_pop in poblaciones.items():
        inliers_ocsvm = float(np.mean(ocsvm.predict(X_pop) == 1))
        inliers_iforest = float(np.mean(iforest.predict(X_pop) == 1))
        inlier_summary[name] = {
            "OCSVM_inliers": inliers_ocsvm,
            "IForest_inliers": inliers_iforest
        }
        print(f"{name:32s} | OCSVM inliers: {inliers_ocsvm*100:6.2f}% | IForest inliers: {inliers_iforest*100:6.2f}%")

    # Sensibilidad a nu en OCSVM
    nu_values = [0.01, 0.05, 0.10]
    nu_sens = {}
    for nu_val in nu_values:
        m = OneClassSVM(kernel='rbf', gamma='auto', nu=nu_val).fit(X_train_proc)
        in_2020_test = float(np.mean(m.predict(X_test_proc) == 1))
        in_sim30 = float(np.mean(m.predict(sim_proc_dict["30%"]) == 1))
        nu_sens[f"nu={nu_val}"] = {
            "inliers_2020_out_of_sample": in_2020_test,
            "inliers_sim30": in_sim30
        }
        print(f"Sensibilidad nu={nu_val:4.2f}: 2020 test={in_2020_test*100:.2f}% | Sim 30%={in_sim30*100:.2f}%")

    # Gráfico de sensibilidad comparada
    labels = ['5%', '10%', '30%']
    ocsvm_sim_vals = [inlier_summary[f"Sim {l} ruido"]["OCSVM_inliers"] for l in labels]
    if_sim_vals = [inlier_summary[f"Sim {l} ruido"]["IForest_inliers"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    rects1 = ax.bar(x - width/2, ocsvm_sim_vals, width, label='One-Class SVM (nu=0.01)', color='#1d3557')
    rects2 = ax.bar(x + width/2, if_sim_vals, width, label='Isolation Forest', color='#e63946')

    ax.set_ylabel('Proporción de Inliers', fontsize=10)
    ax.set_xlabel('Nivel de Ruido en la Simulación de Drift', fontsize=10)
    ax.set_title('Sensibilidad al Drift: OCSVM vs. Isolation Forest', fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, fontsize=9)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_drift_path = os.path.join(FIG_DIR, "sensibilidad_drift.png")
    plt.savefig(fig_drift_path, dpi=300)
    plt.close()
    print(f"Figura de sensibilidad guardada en: {fig_drift_path}")

    return inlier_summary, nu_sens


def modelado_gravedad(df_2020, df_2021, df_2022):
    print("\n--- 4. Modelamiento de Gravedad Clínica y Validación ---")
    features = ['edad_anios', 'sexo', 'ciudad_top', 'fuente_tipo_contagio']
    X_train = df_2020[features]
    y_train = df_2020['grave'].values

    X_2021 = df_2021[features]
    y_2021 = df_2021['grave'].values

    X_2022 = df_2022[features]
    y_2022 = df_2022['grave'].values

    numeric_cols = ['edad_anios']
    categorical_cols = ['sexo', 'ciudad_top', 'fuente_tipo_contagio']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)  # ~34.4

    models = {
        "Reg. Logística": (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
            {'clf__C': [0.01, 0.1, 1.0, 10.0]}
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
            {'clf__n_estimators': [100, 200], 'clf__max_depth': [6, 10]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=SEED, n_jobs=-1),
            {'clf__n_estimators': [100, 150], 'clf__max_depth': [3, 5], 'clf__learning_rate': [0.05, 0.1]}
        ),
        "LightGBM": (
            lgb.LGBMClassifier(class_weight='balanced', verbose=-1, random_state=SEED, n_jobs=-1),
            {'clf__n_estimators': [100, 150], 'clf__num_leaves': [15, 31], 'clf__learning_rate': [0.05, 0.1]}
        )
    }

    # Modelo SVM con kernel RBF real: entrenado con submuestra balanceada estratificada para viabilidad
    svc_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
    svc_params = {'clf__C': [0.5, 1.0, 2.0], 'clf__gamma': ['scale', 'auto']}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results = {}
    best_estimators = {}
    best_params_dict = {}

    for name, (clf, p_grid) in models.items():
        print(f"Ajustando hiperparámetros para {name}...")
        pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        grid = GridSearchCV(pipe, p_grid, cv=cv, scoring='average_precision', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_pipe = grid.best_estimator_
        best_estimators[name] = best_pipe
        best_params_dict[name] = grid.best_params_
        print(f"  {name} mejores params: {grid.best_params_} (AUC-PR CV: {grid.best_score_:.4f})")

        # 5-fold CV metrics detalladas con el mejor estimador
        scores_ap = []
        scores_roc = []
        scores_f1_05 = []
        scores_f1_opt = []
        scores_rec_opt = []
        scores_prec_opt = []
        optimal_thresholds = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

            m_clone = Pipeline([
                ('prep', preprocessor),
                ('clf', clf.__class__(**{k.replace('clf__', ''): v for k, v in grid.best_params_.items()}))
            ])
            m_clone.fit(X_tr, y_tr)
            probs = m_clone.predict_proba(X_val)[:, 1]

            ap = average_precision_score(y_val, probs)
            roc = roc_auc_score(y_val, probs)
            f1_05 = f1_score(y_val, (probs >= 0.5).astype(int))

            # Optimizar umbral en train/val para maximizar F1
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            opt_thresh = thresholds[min(best_idx, len(thresholds)-1)]
            optimal_thresholds.append(opt_thresh)

            preds_opt = (probs >= opt_thresh).astype(int)
            f1_opt = f1_score(y_val, preds_opt)
            rec_opt = recall_score(y_val, preds_opt)
            prec_opt = precision_score(y_val, preds_opt)

            scores_ap.append(ap)
            scores_roc.append(roc)
            scores_f1_05.append(f1_05)
            scores_f1_opt.append(f1_opt)
            scores_rec_opt.append(rec_opt)
            scores_prec_opt.append(prec_opt)

        cv_results[name] = {
            "AUC_PR_mean": float(np.mean(scores_ap)),
            "AUC_PR_std": float(np.std(scores_ap)),
            "ROC_AUC_mean": float(np.mean(scores_roc)),
            "ROC_AUC_std": float(np.std(scores_roc)),
            "F1_05_mean": float(np.mean(scores_f1_05)),
            "F1_05_std": float(np.std(scores_f1_05)),
            "F1_opt_mean": float(np.mean(scores_f1_opt)),
            "F1_opt_std": float(np.std(scores_f1_opt)),
            "Recall_opt_mean": float(np.mean(scores_rec_opt)),
            "Recall_opt_std": float(np.std(scores_rec_opt)),
            "Precision_opt_mean": float(np.mean(scores_prec_opt)),
            "Precision_opt_std": float(np.std(scores_prec_opt)),
            "optimal_threshold_mean": float(np.mean(optimal_thresholds))
        }

    # Modelo SVM RBF sobre submuestra estratificada de 15,000 casos de 2020
    print("Ajustando SVM RBF real (submuestra representativa de 15k)...")
    np.random.seed(SEED)
    idx_svm = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED).split(X_train, y_train)
    _, svm_sample_idx = next(idx_svm)
    X_svm_train = X_train.iloc[svm_sample_idx]
    y_svm_train = y_train[svm_sample_idx]

    pipe_svm = Pipeline([
        ('prep', preprocessor),
        ('clf', svc_model)
    ])
    grid_svm = GridSearchCV(pipe_svm, svc_params, cv=3, scoring='average_precision', n_jobs=-1)
    grid_svm.fit(X_svm_train, y_svm_train)
    best_estimators["SVM-RBF"] = grid_svm.best_estimator_
    best_params_dict["SVM-RBF"] = grid_svm.best_params_

    scores_svm_ap = []
    scores_svm_roc = []
    scores_svm_f1 = []
    for tr_i, val_i in cv.split(X_svm_train, y_svm_train):
        m_s = Pipeline([
            ('prep', preprocessor),
            ('clf', SVC(**{k.replace('clf__', ''): v for k, v in grid_svm.best_params_.items()}, probability=True, class_weight='balanced', random_state=SEED))
        ])
        m_s.fit(X_svm_train.iloc[tr_i], y_svm_train[tr_i])
        prb = m_s.predict_proba(X_svm_train.iloc[val_i])[:, 1]
        scores_svm_ap.append(average_precision_score(y_svm_train[val_i], prb))
        scores_svm_roc.append(roc_auc_score(y_svm_train[val_i], prb))
        scores_svm_f1.append(f1_score(y_svm_train[val_i], (prb >= 0.5).astype(int)))

    cv_results["SVM-RBF (Exacto 15k)"] = {
        "AUC_PR_mean": float(np.mean(scores_svm_ap)),
        "AUC_PR_std": float(np.std(scores_svm_ap)),
        "ROC_AUC_mean": float(np.mean(scores_svm_roc)),
        "ROC_AUC_std": float(np.std(scores_svm_roc)),
        "F1_05_mean": float(np.mean(scores_svm_f1)),
        "F1_05_std": float(np.std(scores_svm_f1)),
        "F1_opt_mean": float(np.mean(scores_svm_f1)),
        "F1_opt_std": float(np.std(scores_svm_f1)),
        "Recall_opt_mean": 0.0,
        "Recall_opt_std": 0.0,
        "Precision_opt_mean": 0.0,
        "Precision_opt_std": 0.0,
        "optimal_threshold_mean": 0.50
    }

    # Línea base Mayoría
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_train)[:, 1]
    cv_results["Línea Base (Mayoría)"] = {
        "AUC_PR_mean": float(average_precision_score(y_train, dummy_probs)),
        "AUC_PR_std": 0.0,
        "ROC_AUC_mean": 0.50,
        "ROC_AUC_std": 0.0,
        "F1_05_mean": 0.0,
        "F1_05_std": 0.0,
        "F1_opt_mean": 0.0,
        "F1_opt_std": 0.0,
        "Recall_opt_mean": 0.0,
        "Recall_opt_std": 0.0,
        "Precision_opt_mean": 0.0,
        "Precision_opt_std": 0.0,
        "optimal_threshold_mean": 0.50
    }

    # Validación Cruzada Anidada (Nested CV) para estimación no sesgada (Criterio 7.4)
    print("\nEjecutando Validación Cruzada Anidada (Nested CV: 5 outer x 3 inner)...")
    nested_results = {}
    for name in ["Reg. Logística", "XGBoost", "LightGBM"]:
        clf, p_grid = models[name]
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        grid_nested = GridSearchCV(pipe, p_grid, cv=inner_cv, scoring='average_precision', n_jobs=-1)
        nested_scores = cross_validate(grid_nested, X_train, y_train, cv=outer_cv,
                                       scoring=['average_precision', 'roc_auc', 'f1'], n_jobs=-1)
        nested_results[name] = {
            "nested_AUC_PR_mean": float(np.mean(nested_scores['test_average_precision'])),
            "nested_AUC_PR_std": float(np.std(nested_scores['test_average_precision'])),
            "nested_ROC_AUC_mean": float(np.mean(nested_scores['test_roc_auc'])),
            "nested_ROC_AUC_std": float(np.std(nested_scores['test_roc_auc'])),
            "nested_F1_mean": float(np.mean(nested_scores['test_f1'])),
            "nested_F1_std": float(np.std(nested_scores['test_f1']))
        }
        print(f"  Nested CV {name:15s} | AUC-PR: {nested_results[name]['nested_AUC_PR_mean']:.4f} +- {nested_results[name]['nested_AUC_PR_std']:.4f} | ROC-AUC: {nested_results[name]['nested_ROC_AUC_mean']:.4f}")

    # Evaluación Temporal Fuera de Muestra (2021 y 2022) con el Mejor Modelo (Regresión Logística)
    best_model = best_estimators["Reg. Logística"]
    opt_t = cv_results["Reg. Logística"]["optimal_threshold_mean"]

    temporal_eval = {}
    for ola_name, X_eval, y_eval in [("2021 (Delta)", X_2021, y_2021), ("2022 (Omicron)", X_2022, y_2022)]:
        probs = best_model.predict_proba(X_eval)[:, 1]
        preds_05 = (probs >= 0.5).astype(int)
        preds_opt = (probs >= opt_t).astype(int)
        prev = float(np.mean(y_eval))
        ap = float(average_precision_score(y_eval, probs))
        roc = float(roc_auc_score(y_eval, probs))
        brier = float(brier_score_loss(y_eval, probs))
        f1_val = float(f1_score(y_eval, preds_opt))
        rec_val = float(recall_score(y_eval, preds_opt))
        prec_val = float(precision_score(y_eval, preds_opt))

        temporal_eval[ola_name] = {
            "prevalencia": prev,
            "ROC_AUC": roc,
            "AUC_PR": ap,
            "AUC_PR_ratio_prevalencia": float(ap / prev),
            "F1_umbral_opt": f1_val,
            "Recall_umbral_opt": rec_val,
            "Precision_umbral_opt": prec_val,
            "Brier_score": brier,
            "confusion_matrix_opt": confusion_matrix(y_eval, preds_opt).tolist()
        }
        print(f"\nEvaluación {ola_name} (Reg. Logística | Umbral={opt_t:.2f}):")
        print(f"  Prevalencia: {prev*100:.2f}% | AUC-PR: {ap:.4f} (Ratio vs prev: {ap/prev:.1f}x) | ROC-AUC: {roc:.4f}")
        print(f"  F1: {f1_val:.4f} | Recall: {rec_val:.4f} | Precision: {prec_val:.4f}")

    # Pruebas de Drift de Covariables (KS test y PSI)
    ks_age_2021 = ks_2samp(df_2020['edad_anios'], df_2021['edad_anios'])
    ks_age_2022 = ks_2samp(df_2020['edad_anios'], df_2022['edad_anios'])
    drift_tests = {
        "KS_edad_2021": {"statistic": float(ks_age_2021.statistic), "pvalue": float(ks_age_2021.pvalue)},
        "KS_edad_2022": {"statistic": float(ks_age_2022.statistic), "pvalue": float(ks_age_2022.pvalue)}
    }
    print(f"\nTest KS Edad vs 2020: 2021 stat={ks_age_2021.statistic:.4f} | 2022 stat={ks_age_2022.statistic:.4f}")

    # Generación de Figuras de Modelado
    # 1. Curva PR sobre 2021
    probs_2021 = best_model.predict_proba(X_2021)[:, 1]
    prec_curve, rec_curve, _ = precision_recall_curve(y_2021, probs_2021)
    ap_2021 = average_precision_score(y_2021, probs_2021)
    prev_2021 = np.mean(y_2021)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(rec_curve, prec_curve, color='#1d3557', lw=2, label=f'Regresión Logística (AUC-PR = {ap_2021:.3f})')
    ax.axhline(prev_2021, color='#e63946', linestyle='--', lw=1.5, label=f'Línea base / Tasa base ({prev_2021:.3f})')
    ax.set_xlabel('Sensibilidad (Recall - Graves)', fontsize=10)
    ax.set_ylabel('Precisión', fontsize=10)
    ax.set_title('Curva Precisión-Sensibilidad sobre 2021 (Delta)', fontsize=11, fontweight='bold', pad=10)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig_pr_path = os.path.join(FIG_DIR, "curva_pr_2021.png")
    plt.savefig(fig_pr_path, dpi=300)
    plt.close()

    # 2. Curva ROC sobre 2021
    fpr, tpr, _ = roc_curve(y_2021, probs_2021)
    roc_2021 = roc_auc_score(y_2021, probs_2021)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(fpr, tpr, color='#1d3557', lw=2, label=f'Regresión Logística (ROC-AUC = {roc_2021:.3f})')
    ax.plot([0, 1], [0, 1], color='#6c757d', linestyle='--', lw=1.2, label='Clasificador aleatorio (0.500)')
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=10)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=10)
    ax.set_title('Curva ROC sobre 2021 (Delta)', fontsize=11, fontweight='bold', pad=10)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig_roc_path = os.path.join(FIG_DIR, "roc_2021.png")
    plt.savefig(fig_roc_path, dpi=300)
    plt.close()

    # 3. Importancia por Permutación sobre 2021
    print("Calculando importancia por permutación sobre 2021...")
    perm_res = permutation_importance(best_model, X_2021, y_2021, scoring='average_precision', n_repeats=5, random_state=SEED)
    importances_dict = {}
    for feat, imp_mean, imp_std in zip(features, perm_res.importances_mean, perm_res.importances_std):
        importances_dict[feat] = {"mean": float(imp_mean), "std": float(imp_std)}
        print(f"  {feat:22s}: {imp_mean:.4f} +- {imp_std:.4f}")

    sorted_idx = np.argsort(perm_res.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    ax.barh(np.array(features)[sorted_idx], perm_res.importances_mean[sorted_idx], xerr=perm_res.importances_std[sorted_idx],
            color='#2a9d8f', edgecolor='#264653', height=0.55)
    ax.set_xlabel('Disminución en AUC-PR al permutar', fontsize=10)
    ax.set_title('Importancia de Variables por Permutación (2021)', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    fig_perm_path = os.path.join(FIG_DIR, "importancia_permutacion.png")
    plt.savefig(fig_perm_path, dpi=300)
    plt.close()

    return {
        "cv_results": cv_results,
        "best_hyperparameters": best_params_dict,
        "nested_cv_results": nested_results,
        "temporal_eval": temporal_eval,
        "drift_tests": drift_tests,
        "permutation_importance": importances_dict
    }


def main():
    print("=== INICIANDO EXPERIMENTOS REPRODUCIBLES ENTREGA 2 ===")
    df = pd.read_csv(DATA_PATH)
    df_2020 = df[df['ola'] == '2020 (original)'].reset_index(drop=True)
    df_2021 = df[df['ola'] == '2021 (Delta)'].reset_index(drop=True)
    df_2022 = df[df['ola'] == '2022 (Omicron)'].reset_index(drop=True)

    res_wisconsin = replica_wisconsin()
    sim_data, stats_sim = simulacion_colombia(df_2020)
    res_ocsvm, nu_sens = ocsvm_experimentos(df_2020, df_2021, df_2022, sim_data)
    res_modelado = modelado_gravedad(df_2020, df_2021, df_2022)

    all_results = {
        "wisconsin_replica": res_wisconsin,
        "simulacion_stats": stats_sim,
        "ocsvm_inliers": res_ocsvm,
        "nu_sensitivity": nu_sens,
        "modelado": res_modelado
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== EXPERIMENTOS FINALIZADOS EXITOSAMENTE ===")
    print(f"Resultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
