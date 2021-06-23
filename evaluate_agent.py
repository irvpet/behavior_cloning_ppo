import torch
from policy_new import BC, Policy
import numpy as np
import matplotlib.pyplot as plt
import gym

agt_file = '/home/giovani/ground_clone/OLD_a2c_agent_v0_ntraj=3.pth'

agent = torch.load(agt_file)

env = gym.make('LunarLanderContinuous-v2')

n_games = 10
score_history = []

for game in range(n_games):
    obs = env.reset()
    score_episode = 0
    done = False

    while not done:
        obs_pth = torch.Tensor(obs).unsqueeze(0)
        _, action_pth, _ = agent.act(obs_pth, deterministic=True)
        action = action_pth.squeeze().detach().numpy()

        obs_, reward, done, info = env.step(action)
        score_episode += reward
        obs = obs_
        env.render()

    score_history.append(score_episode)

env.close()
plt.plot(score_history)
plt.show()
