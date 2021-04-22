import optuna

exp_dir = 'sentence-exp'
test_num = 8
output_db_path = 'sqlite:////home/juan/Research/research-f2020/experiments/test.db'

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
        serialization_dir=f"./{exp_dir}/results/optuna{test_num}/{trial.number}",
        metrics="best_validation_accuracy"
    )
    return executor.run()

if __name__ == '__main__':
    study = optuna.create_study(
        storage=output_db_path,  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name=f"{exp_dir}-exp{test_num}",
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

    dump_best_config(CONFIG_PATH, BEST_CONFIG_PATH, study)
    print("\nCreated optimized AllenNLP config to `{}`.".format(BEST_CONFIG_PATH))

    shutil.rmtree(MODEL_DIR)

