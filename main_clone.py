import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import storage_ground
from baselines.common.running_mean_std import RunningMeanStd
from ensemble_variance import calculate_variance
from collections import deque
from policy_new import PPO, evaluate_agent

class Args():
    def __init__(self):
        self.use_gae = True
        self.gamma = 0.95
        self.gae_lambda = 0.99
        self.use_proper_time_limits = True
        self.num_steps = 2048
        self.num_updates = 20
        self.num_processes = 1



class TimeLimitMask(gym.Wrapper):
    def __init__(self, env):
        super(TimeLimitMask, self).__init__(env)
        self.env.total_steps = 0
        self.env._max_episode_steps = 2048/4 -1

        self.env._current_ep_reward = []

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.env.total_steps += 1
        self.env._current_ep_reward.append(rew)

        if done and self.env._max_episode_steps == self.env.total_steps:
            info['bad_transition'] = True

        if done and self.env._max_episode_steps != self.env.total_steps:
            info['masks'] = True
            info['episode'] = {'r':sum(self.env._current_ep_reward), 'l':len(self.env._current_ep_reward)}
            self.env._current_ep_reward = []

            done = False
            obs = self.env.reset()

        if self.env._max_episode_steps == self.env.total_steps:
            done = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def main():
    np.set_printoptions(linewidth=400)

    env = gym.make('LunarLanderContinuous-v2')
    env = TimeLimitMask(env)

    agt_file = '/home/giovani/ground_clone/NEW_a2c_agent_v0_ntraj=3.pth'
    ens_file = '/home/giovani/ground_clone/ensemble_512_v0_ntraj=3.pth'

    bc_model = torch.load(agt_file)
    ensemble = torch.load(ens_file)

    ppo_model = PPO(actor_critic=bc_model, bc_model=bc_model)
    dril = DRIL(bc_model, ensemble)

    num_steps = 2048
    num_processes = 1
    obs_shape = [8]
    action_space = env.action_space
    recurrent_hidden_state_size = 1
    recurrent_hidden_states = torch.tensor([[0]])

    rollouts = storage_ground.RolloutStorage(num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size)

    obs = env.reset()
    obs = torch.Tensor(obs).unsqueeze(0)
    rollouts.obs[0].copy_(obs)

    args = Args()

    episode_rewards = deque(maxlen=10)
    episode_uncertainty_rewards = deque(maxlen=10)
    running_uncertainty_reward = 0


    episode_rewards = []

    for j in range(args.num_updates):
        print(f'Update {j}')

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action_tf, action_log_prob = ppo_model.actor_critic.policy.act(obs, deterministic=False)

            action = action_tf.squeeze().detach().cpu().numpy()
            obs_, reward, done, infos = env.step(action)

            reward = dril.predict_reward(action, obs)
            running_uncertainty_reward += reward.view(-1).numpy()

            obs_ = torch.Tensor(obs_).unsqueeze(0)
            action = torch.Tensor(action).unsqueeze(0)


            if 'episode' in infos.keys():
                episode_rewards.append(infos['episode']['r'])
                episode_uncertainty_rewards.append(running_uncertainty_reward / infos['episode']['l'])
                print(f"Ep reward: {infos['episode']['r']:.1f} | running uncertainty = {running_uncertainty_reward} \
                | Episode length: {infos['episode']['l']:.1f}")
                running_uncertainty_reward = 0

            masks = torch.FloatTensor([0.0] if done else [1.0])
            bad_masks = torch.FloatTensor([0.0] if 'bad_transition' in infos.keys() else [1.0])

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

            obs = obs_

        # ---------- Rollout loop end ----------

        with torch.no_grad():
            next_value = ppo_model.actor_critic.policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        for step in range(args.num_steps):
            rollouts.rewards[step] = dril.normalize_reward(
                rollouts.obs[step], rollouts.actions[step], args.gamma,
                rollouts.masks[step], rollouts.rewards[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = ppo_model.update(rollouts)

        rollouts.after_update()

    agent_score = evaluate_agent(ppo_model.actor_critic)
    print(f'DRIL agent score in 10 episodes = {agent_score}')

    env.close()


class DRIL():
    ''' DRIL reward function '''
    def __init__(self, model, ensemble):
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.actor_critic = model
        self.bc = model
        self.ensemble = ensemble
        self.variance_threshold = calculate_variance()

    def predict_reward(self, action, state):
        # get the ensemble actions for the state
        ensemble_actions = []
        for agent in self.ensemble:
            action_ens = agent(state)
            ensemble_actions.append(action_ens.detach().squeeze().cpu().numpy())
        ensemble_actions = np.array(ensemble_actions)
        cov = np.cov(ensemble_actions, rowvar=False)
        variance = np.matmul(np.matmul(action, cov), action.T)
        if variance >= self.variance_threshold:
            output = -1
        else:
            output = 1

        return torch.Tensor([output])


    def normalize_reward(self, state, action, gamma, masks, reward, update_rms=True):
        if self.returns is None:
            self.returns = reward.clone()

        if update_rms:
            self.returns = self.returns * masks * gamma + reward
            self.ret_rms.update(self.returns.cpu().numpy())

        return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


def normalization_verifier():
    PATH_before = '/home/giovani/ground_clone/rollout_before_norm'
    rollouts_before = torch.load(PATH_before, map_location=torch.device('cpu'))

    PATH_after = '/home/giovani/ground_clone/rollout_after_norm'
    rollouts_after = torch.load(PATH_after, map_location=torch.device('cpu'))

    episode_mask = rollouts_before.masks
    rewards_un = rollouts_before.rewards
    rewards_norm = rollouts_after.rewards

    episode_mask = episode_mask.squeeze().squeeze().cpu().numpy()
    rewards_un = rewards_un.squeeze().squeeze().cpu().numpy()
    rewards_norm = rewards_norm.squeeze().squeeze().cpu().numpy()

    rewards_calc = torch.zeros_like(rollouts_before.rewards)

    dril = DRIL(None, None, None)

    for step in range(rollouts_before.rewards.size()[0]):
        rewards_calc[step] = dril.normalize_reward(
            rollouts_before.obs[step], rollouts_before.actions[step], 0.99,
            rollouts_before.masks[step], rollouts_before.rewards[step])

    rewards_calc = rewards_calc.squeeze().squeeze().cpu().numpy()

    def curves():
        fig, ax = plt.subplots(4, 1, figsize=(16, 8))
        ax[0].plot(episode_mask)
        ax[0].set_ylabel('End of episode == 0')

        ax[1].plot(rewards_un)
        ax[1].set_ylabel('Rewards')

        ax[2].plot(rewards_norm)
        ax[2].set_ylabel('Normalized Rewards')
        ax[2].set_ylim([-0.1, 0.1])

        ax[3].plot(rewards_calc)
        ax[3].set_ylabel('Calc norm rewards')
        ax[3].set_ylim([-0.1, 0.1])

    agt_file = '/home/giovani/ground_clone/a2c_agent_v0_ntraj=3.pth'

    actor_critic = torch.load(agt_file, map_location=torch.device('cpu'))

    rollouts = rollouts_after
    rollouts.to('cpu')


    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()


    args = Args()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

    value_loss_epoch, action_loss_epoch, dist_entropy_epoch = actor_critic.update(rollouts)

    print(f'value loss {value_loss_epoch}')
    print(f'action_loss epoch {action_loss_epoch}')
    print(f'dist entropy {dist_entropy_epoch}')


    rollouts.after_update()

    computed_returns = rollouts.returns.detach().squeeze().squeeze().numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(computed_returns)
    #plt.show()

if __name__=='__main__':
    main()
