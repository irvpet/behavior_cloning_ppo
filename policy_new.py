import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from _dril_distrib import DiagGaussian, init, AddBias


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))


    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

        self.clip_param = 0.2


    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor


class Policy(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(Policy, self).__init__()
        self.epsilon = 0.1
        self.dril = True
        self.base = MLPBase(num_inputs, num_hidden, num_outputs)
        self.dist = DiagGaussian(num_hidden, num_outputs)


    def get_action(self, inputs):
        value, actor_features = self.base(inputs)
        return self.dist.get_mean(actor_features)


    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs


    def get_value(self, inputs, rnn_hxs, masks):
        value, _ = self.base(inputs)
        return value


    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class BC():
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = optim.Adam(params=policy.actor.parameters(), lr=0.0002)
        self.obs = None
        self.acs = None
        self.load_expertdata()

    def load_expertdata(self):
        obs_file = '/home/giovani/dril/dril/dril/demo_data/obs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
        acs_file = '/home/giovani/dril/dril/dril/demo_data/acs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'

        obs = np.load(obs_file)
        acs = np.load(acs_file)

        device = 'cuda:0'
        obs = torch.tensor(obs, dtype=torch.float)
        acs = torch.tensor(acs, dtype=torch.float)

        perm = torch.randperm(obs.size(0))
        obs = obs[perm]
        acs = acs[perm]

        self.obs = obs
        self.acs = acs


    def bc_update(self):
        batch_size = 100

        self.optimizer.zero_grad()

        perm = torch.randperm(self.obs.size(0))
        obs = self.obs[perm]
        acs = self.acs[perm]

        obs_batch = obs[0: batch_size, :]
        acs_batch = acs[0: batch_size, :]

        pred_action = self.policy.get_action(obs_batch).view(batch_size, -1)

        loss = F.mse_loss(pred_action, acs_batch)
        loss.backward()
        self.optimizer.step()


class PPO():
    def __init__(self, actor_critic, bc_model):
        self.actor_critic = actor_critic
        self.bc = bc_model
        self.optimizer = optim.Adam(params=actor_critic.policy.parameters(), lr=1e-5)
        self.max_grad_norm = 0.1
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.clip_param = 0.2
        self.use_clipped_value_loss = True
        self.num_mini_batch = 64
        self.ppo_epoch = 10


    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for layer in self.actor_critic.policy.state_dict().keys():
            print(layer)

        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.policy.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                self.bc.bc_update()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates


        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


def evaluate_agent(agent):
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('LunarLanderContinuous-v2')

    n_games = 100
    score_history = []

    for game in range(n_games):
        obs = env.reset()
        score_episode = 0
        done = False

        while not done:
            obs_pth = torch.Tensor(obs).unsqueeze(0)
            _, action_pth, _ = agent.policy.act(obs_pth, deterministic=True)
            action = action_pth.squeeze().detach().numpy()

            obs_, reward, done, info = env.step(action)
            score_episode += reward
            obs = obs_
            #env.render()

        score_history.append(score_episode)

    env.close()

    GAME_THRESHOLD = 200.

    score_history = np.array(score_history)
    avg_score = np.ones_like(score_history)*score_history.mean()
    solved = np.ones_like(score_history) * GAME_THRESHOLD

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(score_history, label='score per episode', marker='o', linestyle='-')
    ax.plot(avg_score, label='Average Agent Score with BC training')
    ax.plot(solved, label='Score threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_ylim([-50, 350])
    plt.legend()
    plt.show()

    return score_history.mean()
