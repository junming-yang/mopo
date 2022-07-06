import numpy as np
import torch
import os
from common import util, functional
from models.ensemble_dynamics import EnsembleModel
from operator import itemgetter
from common.normalizer import StandardNormalizer
from copy import deepcopy


class TransitionModel:
    def __init__(self,
                 obs_space,
                 action_space,
                 static_fns,
                 holdout_ratio=0.1,
                 inc_var_loss=False,
                 use_weight_decay=False,
                 **kwargs):

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        # fix hidden_dims
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, device=util.device, **kwargs['model'])
        self.static_fns = static_fns
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

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data)
        obs_list = torch.Tensor(obs_list)
        action_list = torch.Tensor(action_list)
        next_obs_list = torch.Tensor(next_obs_list)
        reward_list = torch.Tensor(reward_list)
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
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data_batch)
        obs_batch = torch.Tensor(obs_batch)
        action_batch = torch.Tensor(action_batch)
        next_obs_batch = torch.Tensor(next_obs_batch)
        reward_batch = torch.Tensor(reward_batch)

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
        # Todo: add discriminator
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
        """
        predict next_obs and rew
        """
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
        terminals = self.static_fns.termination_fn(obs, act, next_obs)

        # penalty rewards
        penalty_coeff = 1
        penalty_learned_var = True
        if penalty_coeff != 0:
            if not penalty_learned_var:
                ensemble_means_obs = pred_diff_means[:, :, 1:]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)  # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0, :obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
                penalty = np.max(dists, axis=0)  # max distances over models
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
            penalized_rewards = rewards - penalty_coeff * penalty
        else:
            penalized_rewards = rewards

        assert (type(next_obs) == np.ndarray)
        info = {'penalty': penalty, 'penalized_rewards': penalized_rewards}
        return next_obs, penalized_rewards, terminals, info

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