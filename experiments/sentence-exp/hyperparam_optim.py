import optuna

def objective(trial: optuna.Trial) -> float:
    trial.suggest_int('embedding_dim', 16, 1024)
    trial.suggest_float('lr', 5e-4, 5e-1, log=True)
    
