import optuna

exp_dir = 'universal-exp'
test_num = 0
output_db_path = 'sqlite:////home/juan/Research/research-f2020/experiments/hyperparam-optim.db'

def objective(trial: optuna.Trial) -> float:
    s_weight = trial.suggest_float('s_weight', 0, 1)

    if s_weight != 0:
        trial.suggest_int('s_embedding_dim', 16, 512)
        trial.suggest_float('s_dropout', 0.0, 0.9)
    
    if s_weight != 1:
        trial.suggest_int('v_embedding_dim', 16, 512)
        trial.suggest_float('v_dropout', 0.0, 0.9)

    trial.suggest_float('lr', 5e-4, 5e-2, log=True)
    
    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,
        config_file=f"./{exp_dir}/training_config/config.jsonnet",  # path to jsonnet
        serialization_dir=f"./{exp_dir}/results/optuna/test-{test_num}/{trial.number}",
        metrics="best_validation_accuracy",
        include_package=f'{exp_dir}.architecture'
    )
    return executor.run()

if __name__ == '__main__':
    study = optuna.create_study(
        storage=output_db_path,  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name=f"{exp_dir}{test_num}",
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner()
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=100,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
