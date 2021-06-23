import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from a2c_policy import Agent_a2c

def main():
    np.set_printoptions(linewidth=400)

    n_games = 100
    env = gym.make('LunarLanderContinuous-v2')
    reward_history = []
    agt_file = '/home/giovani/ground_clone/a2c_agent_v0_ntraj=3.pth'
    ens_file = '/home/giovani/ground_clone/ensemble_512_v0_ntraj=3.pth'
    rollout_file = '/home/giovani/ground_clone/rollout_512_v0_ntraj=3.pth'
    reward_file = '/home/giovani/ground_clone/reward_512_sample_v0_ntraj=3.pth'

    print(f'... Loading agent @ {agt_file}')
    agent = torch.load(agt_file)
    print(f'... Loading ensemble @ {ens_file}')

    ensemble = torch.load(ens_file)

    obs_rollout = []
    agt_rollout = []
    ens_rollout = []
    rew_rollout = []
    rollout = []

    print(f'... Starting main loop')
    for i in range(n_games):
        obs = env.reset()
        episode_reward = 0
        done = 0
        while not done:
            obs_pth = torch.Tensor(obs)
            obs_pth = obs_pth.unsqueeze(0)
            _ ,action_pth, _ = agent.act(obs_pth)

            clip_action = torch.clamp(action_pth, float(env.action_space.low[0]), \
                                      float(env.action_space.high[0]))

            action = action_pth.detach().squeeze().cpu().numpy()

            acs_ens_agents = []
            for j in range(5):
                acs_ens_agent = ensemble[j](obs)
                acs_ens_agents.append(acs_ens_agent.detach().cpu().numpy())

            obs_, reward, done, info = env.step(action)
            episode_reward += reward
            #env.render()

            obs_rollout.append(obs)
            agt_rollout.append(action)
            ens_rollout.append(acs_ens_agents)
            rew_rollout.append(reward)

            obs = obs_
        print(f'Game {i}, Score = {episode_reward}')

        reward_history.append(episode_reward)

    plt.plot(reward_history)
    plt.show()


    rollout.append(obs_rollout)
    rollout.append(agt_rollout)
    rollout.append(ens_rollout)
    
    print(f'... Saving rollouts @ {rollout_file}')
    torch.save(rollout, rollout_file)
    torch.save(reward_history, reward_file)

    env.close()

if __name__ == '__main__':
    main()