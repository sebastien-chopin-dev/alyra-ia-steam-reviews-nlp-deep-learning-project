import time


def hyperparameter_optimization_pipeline(X_train, y_train, X_val, y_val):
    """
    Pipeline complet d'optimisation des hyperparamètres
    """
    # 1. Recherche grossière avec Random Search
    print("Phase 1: Recherche grossière")
    coarse_search = RandomizedSearchCV(
        estimator=create_model_wrapper(),
        param_distributions=get_coarse_param_grid(),
        n_iter=20,
        cv=3,
        scoring="accuracy",
    )
    coarse_result = coarse_search.fit(X_train, y_train)

    # 2. Recherche fine autour des meilleurs paramètres
    print("Phase 2: Recherche fine")
    best_params = coarse_result.best_params_
    fine_param_grid = create_fine_grid_around_best(best_params)
    fine_search = GridSearchCV(
        estimator=create_model_wrapper(),
        param_grid=fine_param_grid,
        cv=5,
        scoring="accuracy",
    )
    fine_result = fine_search.fit(X_train, y_train)

    # 3. Validation finale avec Bayesian Optimization
    print("Phase 3: Optimisation Bayésienne")
    final_params = bayesian_fine_tuning(fine_result.best_params_)
    return final_params


def get_coarse_param_grid():
    return {
        "learning_rate": uniform(0.0001, 0.1),
        "batch_size": [16, 32, 64, 128],
        "n_layers": randint(1, 6),
        "n_units": randint(32, 512),
    }


def create_fine_grid_around_best(best_params):
    # Créer une grille fine autour des meilleurs paramètres
    lr = best_params["learning_rate"]

    return {
        "learning_rate": [lr * 0.5, lr, lr * 2],
        "batch_size": [best_params["batch_size"]],
        "n_layers": [
            best_params["n_layers"] - 1,
            best_params["n_layers"],
            best_params["n_layers"] + 1,
        ],
        "n_units": [
            int(best_params["n_units"] * 0.75),
            best_params["n_units"],
            int(best_params["n_units"] * 1),
        ],
    }


#  Métriques d'évaluation pour l'optimisation


def comprehensive_model_evaluation(model, X_test, y_test, hyperparams):
    """
    Évaluation complète incluant performance et efficacité
    """
    # Performance
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Temps d'inférence
    start_time = time.time()
    predictions = model.predict(X_test[:1000])
    inference_time = (time.time() - start_time) / 1000  # Temps par échantillon

    # Taille du modèle
    model_size_mb = get_model_size_mb(model)

    # Score composite
    efficiency_score = test_accuracy / (inference_time * model_size_mb)
    results = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "inference_time_ms": inference_time * 1000,
        "model_size_mb": model_size_mb,
        "efficiency_score": efficiency_score,
        "hyperparameters": hyperparams,
    }
    return results


def get_model_size_mb(model):
    """Calcule la taille du modèle en MB"""
    total_params = model.count_params()
    return (total_params * 4) / (1024 * 1024)  # 4 bytes par paramètre (float32)
