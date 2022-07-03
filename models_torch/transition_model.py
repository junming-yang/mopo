import numpy as np
import torch
import os
from common import util, functional
from models_torch.ensemble_dynamics import EnsembleModel
from operator import itemgetter
from common.normalizer import StandardNormalizer
from copy import deepcopy


class TransitionModel:
    def __init__(self,
                 obs_space,
                 action_space,
                 env_name,
                 holdout_ratio=0.1,
                 inc_var_loss=False,
                 use_weight_decay=False,
                 **kwargs):

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        hidden_dim = [200, 200]

        # fix hidden_dims
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dim, device=util.device)
        self.env_name = env_name
        # print("params", type(self.model.parameters()))
        # for i, p in enumerate(self.model.parameters()):
        #     print(i, p.shape)
        # exit(0)

        # fix lr
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.networks = {
            "model": self.model
        }
        self.obs_space = obs_space
        self.holdout_ratio = holdout_ratio
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        self.model_train_timesteps = 0

    def _termination_fn(self, env_name, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if env_name in ["Hopper-v2", "Hopper-v3"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
        elif env_name in ["Walker2d-v2", "Walker2d-v3"]:
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            return done
        elif env_name == "HumanoidTruncatedObs-v2":
            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)
        elif env_name == "InvertedDoublePendulum-v2":
            sin1, cos1 = next_obs[:, 1], next_obs[:, 3]
            sin2, cos2 = next_obs[:, 2], next_obs[:, 4]
            theta_1 = np.arctan2(sin1, cos1)
            theta_2 = np.arctan2(sin2, cos2)
            y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

            done = y <= 1
            return done
        elif env_name == "InvertedPendulum-v2":
            notdone = np.isfinite(next_obs).all(axis=-1) \
                      * (np.abs(next_obs[:, 1]) <= .2)
            done = ~notdone
        elif env_name == "AntTruncatedObs-v2":
            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done
        elif "Swimmer" in env_name or "HalfCheetah" in env_name:  # No done for these two envs
            return np.array([False for _ in obs])
        else:
            raise NotImplementedError

        return done

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("obs", 'action', 'next_obs', 'reward')(data)
        delta_obs_list = next_obs_list - obs_list
        obs_list, action_list = self.transform_obs_action(obs_list, action_list)
        model_input = torch.cat([obs_list, action_list], dim=-1)
        predictions = functional.minibatch_inference(args=[model_input], rollout_fn=self.model.predict,
                                                     batch_size=10000,
                                                     cat_dim=1)  # the inference size grows as model buffer increases
        groundtruths = torch.cat((delta_obs_list, reward_list), dim=1)
        eval_mse_losses, _ = self.model_loss(predictions, groundtruths, mse_only=True)
        if update_elite_models:
            elite_idx = np.argsort(eval_mse_losses.cpu().numpy())
            self.model.elite_model_idxes = elite_idx[:self.model.num_elite]
        return eval_mse_losses.detach().cpu().numpy(), None

    def reset_normalizers(self):
        self.obs_normalizer.reset()
        self.act_normalizer.reset()

    def update_normalizer(self, obs, action):
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(action)

    def transform_obs_action(self, obs, action):
        obs = self.obs_normalizer.transform(obs)
        action = self.act_normalizer.transform(action)
        return obs, action

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("obs", 'action', 'next_obs', 'reward')(data_batch)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch = self.transform_obs_action(obs_batch, action_batch)

        # predict with model
        model_input = torch.cat([obs_batch, action_batch], dim=-1)
        predictions = self.model.predict(model_input)
        # compute training loss
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1)
        train_mse_losses, train_var_losses = self.model_loss(predictions, groundtruths)
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)
        train_transition_loss = train_mse_loss + train_var_loss
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(
            self.model.min_logvar)  # why
        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss += decay_loss
        else:
            decay_loss = None
        # update transition model
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()

        # compute test loss for elite model

        return {
            "loss/train_model_loss_mse": train_mse_loss.item(),
            "loss/train_model_loss_var": train_var_loss.item(),
            "loss/train_model_loss": train_var_loss.item(),
            "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_std": self.model.max_logvar.mean().item(),
            "misc/min_std": self.model.min_logvar.mean().item()
        }

    def model_loss(self, predictions, groundtruths, mse_only=False):
        pred_means, pred_logvars = predictions
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            assert 0
        return mse_losses, var_losses

    @torch.no_grad()
    def predict(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None,]
            act = act[None,]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(util.device)

        scaled_obs, scaled_act = self.transform_obs_action(obs, act)

        model_input = torch.cat([scaled_obs, scaled_act], dim=-1)
        pred_diff_means, pred_diff_logvars = self.model.predict(model_input)
        pred_diff_means = pred_diff_means.detach().cpu().numpy()
        # add curr obs for next obs
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()
        if deterministic:
            pred_diff_means = pred_diff_means
        else:
            pred_diff_means = pred_diff_means + np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = pred_diff_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

        next_obs, rewards = pred_diff_samples[:, :-1] + obs, pred_diff_samples[:, -1]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        assert (type(next_obs) == np.ndarray)

        return next_obs, rewards, terminals

    def update_best_snapshots(self, val_losses):
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True
                improvement = (best_loss - current_loss) / best_loss
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        return updated

    def reset_best_snapshots(self):
        self.model_best_snapshots = [deepcopy(self.model.ensemble_models[idx].state_dict()) for idx in
                                     range(self.model.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.model.ensemble_size)]

    def save_model_snapshot(self, idx):
        self.model_best_snapshots[idx] = deepcopy(self.model.ensemble_models[idx].state_dict())

    def load_best_snapshots(self):
        self.model.load_state_dicts(self.model_best_snapshots)

    def save_model(self, info):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)

    def load_model(self, info):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)