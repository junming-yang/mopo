default_config = {
    "transition_params": {
        "model_batch_size": 256,
        "use_weight_decay": True,
        "optimizer_class": "Adam",
        "learning_rate": 0.001,
        "holdout_ratio": 0.2,
        "inc_var_loss": True,
        "model": {
            "hidden_dims": [200, 200, 200, 200],
            "decay_weights": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
            "act_fn": "swish",
            "out_act_fn": "identity",
            "num_elite": 5,
            "ensemble_size": 7
        }
    },
    "mopo_params": {
        "max_epoch": 125,
        "rollout_batch_size": 50000,
        "rollout_mini_batch_size": 10000,
        "model_retain_epochs": 1,
        "num_env_steps_per_epoch": 1000,
        "train_model_interval": 250,
        "max_trajectory_length": 1000,
        "eval_interval": 1000,
        "num_eval_trajectories": 10,
        "snapshot_interval": 2000,
        "model_env_ratio": 0.95,
        "max_model_update_epochs_to_improve": 5,
        "max_model_train_iterations": "None"
    }
}
