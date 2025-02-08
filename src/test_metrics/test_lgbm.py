import lightgbm as lgb
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score
)

def get_next_file(base_name, extension):
    """Busca el siguiente número disponible para el archivo."""
    i = 1
    while os.path.exists(f"{base_name}_{i}{extension}"):
        i += 1
    return f"{base_name}_{i}{extension}"

def best_results(opt, X_train, X_test, y_train, y_test):
    """Evalúa los 15 mejores modelos de la optimización bayesiana y guarda los resultados."""
    
    # Verificar que cv_results_ tiene las claves necesarias
    if 'params' not in opt.cv_results_ or 'mean_test_accuracy' not in opt.cv_results_:
        raise ValueError("Las claves 'params' o 'mean_test_accuracy' no están en opt.cv_results_.")

    # Obtener los 15 mejores modelos y sus parámetros
    best_models = opt.cv_results_['params'][:15]
    best_scores = opt.cv_results_['mean_test_accuracy'][:15]

    # Almacenar resultados en una lista
    results = []
    
    for i, params in enumerate(best_models):
        # Crear y entrenar el modelo con los mejores parámetros
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Predecir en el conjunto de prueba
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva

        # Calcular métricas de evaluación
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Guardar los resultados
        results.append({
            'params': params,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_accuracy': best_scores[i]
        })

    # Convertir a DataFrame
    results_df = pd.DataFrame(results)

    # Mostrar los resultados
    print(results_df[['params', 'mean_cv_accuracy', 'accuracy', 'roc_auc', 'precision', 'recall', 'f1']])

    # Definir la carpeta de guardado
    output_path = "/home/tomas-mur/Documentos/Codigos/payment/reports/"
    os.makedirs(output_path, exist_ok=True)  # Crear carpeta si no existe

    # Guardar resultados de la optimización bayesiana
    filename1 = get_next_file(os.path.join(output_path, "lgbm_bayesian_optimization_2P"), ".csv")
    pd.DataFrame(opt.cv_results_).to_csv(filename1, index=False)

    # Guardar resultados de la validación en train-test
    filename2 = get_next_file(os.path.join(output_path, "lgbm_traintest_res_2P"), ".csv")
    results_df.to_csv(filename2, index=False)

    print(f"Guardado: {filename1}")
    print(f"Guardado: {filename2}")
