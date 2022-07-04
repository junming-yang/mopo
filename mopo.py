import os

import numpy as np
import torch

from common.functional import dict_batch_generator
from models.tf_dynamics_models.fake_env import FakeEnv
from models.tf_dynamics_models.constructor import format_samples_for_training
from common import util
from common.logger import Logger
import tqdm


class MOPO():
    def __init__(
            self,
            policy,
            dynamics_model,
            static_fns,
            offline_buffer,
            model_buffer,
            reward_penalty_coef,
            rollout_length,
            batch_size,
            real_ratio,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
            rollout_mini_batch_size=1000,
            model_retain_epochs=1,
            num_env_steps_per_epoch=1000,
            train_model_interval=250,
            train_agent_interval=1,
            num_agent_updates_per_env_step=20,  # G
            max_epoch=100000,
            max_agent_updates_per_env_step=5,
            max_model_update_epochs_to_improve=5,
            max_model_train_iterations="None",
            warmup_timesteps=5000,
            model_env_ratio=0.8,
            hold_out_ratio=0.1,
            load_dir="",
            **kwargs
    ):
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.static_fns = static_fns
        self.fake_env = FakeEnv(
            self.dynamics_model,
            self.static_fns,
            penalty_coeff=reward_penalty_coef,
            penalty_learned_var=True
        )
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self.agent_batch_size = agent_batch_size
        self.model_batch_size = model_batch_size
        self.rollout_batch_size = rollout_batch_size
        self.rollout_mini_batch_size = rollout_mini_batch_size
        self.model_retain_epochs = model_retain_epochs
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.train_agent_interval = train_agent_interval
        self.train_model_interval = train_model_interval
        self.num_agent_updates_per_env_step = num_agent_updates_per_env_step
        self.max_agent_updates_per_env_step = max_agent_updates_per_env_step
        self.max_model_update_epochs_to_improve = max_model_update_epochs_to_improve
        if max_model_train_iterations == "None":
            self.max_model_train_iterations = np.inf
        else:
            self.max_model_train_iterations = max_model_train_iterations
        self.max_epoch = max_epoch
        self.warmup_timesteps = warmup_timesteps
        self.model_env_ratio = model_env_ratio
        self.hold_out_ratio = hold_out_ratio
        if load_dir != "":
            self.agent.load_snapshot(load_dir)
        self.model_tot_train_timesteps = 0
        self.logger = Logger('results/runs', 'Hopper', 1)

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self):
        init_transitions = self._sample_initial_transitions()
        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            actions = self.policy.sample_action(observations)
            next_observations, rewards, terminals, infos = self.dynamics_model.predict(observations, actions)
            rew_len = len(rewards)
            rewards = rewards.reshape(rew_len, 1)
            terminals = terminals.reshape(rew_len, 1)
            self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break
            observations = next_observations[nonterm_mask]

    def learn_dynamics(self):
        # get train and eval data
        max_sample_size = self.offline_buffer.get_size
        num_train_data = int(max_sample_size * (1.0 - self.hold_out_ratio))
        env_data = self.offline_buffer.sample_all()
        train_data, eval_data = {}, {}
        for key in env_data.keys():
            train_data[key] = env_data[key][:num_train_data]
            eval_data[key] = env_data[key][num_train_data:]
        self.dynamics_model.reset_normalizers()
        self.dynamics_model.update_normalizer(train_data['observations'], train_data['actions'])

        # train model
        model_train_iters = 0
        model_train_epochs = 0
        num_epochs_since_prev_best = 0
        break_training = False
        self.dynamics_model.reset_best_snapshots()

        # init eval_mse_losses
        eval_mse_losses, _ = self.dynamics_model.eval_data(eval_data, update_elite_models=False)
        self.logger.log_var("loss/model_eval_mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
        updated = self.dynamics_model.update_best_snapshots(eval_mse_losses)
        while not break_training:
            # Todo: add processing bar
            for train_data_batch in dict_batch_generator(train_data, self.model_batch_size):
                model_log_infos = self.dynamics_model.update(train_data_batch)
                model_train_iters += 1
                self.model_tot_train_timesteps += 1

            eval_mse_losses, _ = self.dynamics_model.eval_data(eval_data, update_elite_models=False)
            self.logger.log_var("loss/model_eval_mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
            updated = self.dynamics_model.update_best_snapshots(eval_mse_losses)
            num_epochs_since_prev_best += 1
            if updated:
                model_train_epochs += num_epochs_since_prev_best
                num_epochs_since_prev_best = 0
            if num_epochs_since_prev_best >= self.max_model_update_epochs_to_improve or model_train_iters > self.max_model_train_iterations:
                break
            # Debug
            #break
        self.dynamics_model.load_best_snapshots()

        # evaluate data to update the elite models
        self.dynamics_model.eval_data(eval_data, update_elite_models=True)
        model_log_infos['misc/norm_obs_mean'] = torch.mean(torch.Tensor(self.dynamics_model.obs_normalizer.mean)).item()
        model_log_infos['misc/norm_obs_var'] = torch.mean(torch.Tensor(self.dynamics_model.obs_normalizer.var)).item()
        model_log_infos['misc/norm_act_mean'] = torch.mean(torch.Tensor(self.dynamics_model.act_normalizer.mean)).item()
        model_log_infos['misc/norm_act_var'] = torch.mean(torch.Tensor(self.dynamics_model.act_normalizer.var)).item()
        model_log_infos['misc/model_train_epochs'] = model_train_epochs
        model_log_infos['misc/model_train_train_steps'] = model_train_iters
        return model_log_infos

    """
    def learn_dynamics(self):
        data = self.offline_buffer.sample_all()
        train_inputs, train_outputs = format_samples_for_training(data)
        max_epochs = 1 if self.dynamics_model.model_loaded else None
        loss = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self._batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2
        )
        return loss
    """

    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]],
                                                axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss

    def save_dynamics_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dynamics_model.save_model(save_path, timestep=0)
