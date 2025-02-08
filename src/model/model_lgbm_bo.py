import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer


def model_lgbm_bo (X_train,y_train):
    
    # Definir el modelo
    lgbm = lgb.LGBMClassifier()
    # Calcular el número total de muestras para el ajuste de parámetros
    total_samples = len(X_train)
    # Definir el espacio de búsqueda basado en porcentajes
    param_space = {
        'num_leaves': Integer(200, 550),
        'max_depth': Integer(2, 4),
        'learning_rate': Real(0.05, 0.1, prior='log-uniform'),
        'n_estimators': Integer(150, 300),
        'min_data_in_leaf': Integer(int(0.01 * total_samples), int(0.05 * total_samples)),  # 1% a 5% del total de muestras
    }
    # Configurar la búsqueda bayesiana con múltiples métricas
    opt = BayesSearchCV(
        lgbm,
        search_spaces=param_space,
        n_iter=50,
        cv=5,
        scoring={
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        },
        refit='accuracy',  # Refit para optimizar por 'accuracy' (puedes cambiarlo)
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    # Entrenar el modelo
    opt.fit(X_train, y_train)
    # Visualizar los mejores parámetros encontrados
    print("Mejores parámetros encontrados:", opt.best_params_)

    return opt