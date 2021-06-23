import torch
from policy_new import BC, Policy
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

agt_file = '/home/giovani/ground_clone/NEW_a2c_agent_v0_ntraj=3.pth'
loss_file = '/home/giovani/ground_clone/NEW_a2c_loss_v0_ntraj=3.pth'
obs_file = '/home/giovani/dril/dril/dril/demo_data/obs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
acs_file = '/home/giovani/dril/dril/dril/demo_data/acs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'

obs = np.load(obs_file)
acs = np.load(acs_file)

device = 'cuda:0'
obs = torch.tensor(obs, dtype=torch.float)
acs = torch.tensor(acs, dtype=torch.float)

bc_model = Policy(8, 64, 2)

agent = BC(bc_model)

print(agent.policy)

epochs = 20
batch_size = 32
print(f'length of observations {agent.obs.size(0)}')
iter_per_epoch = int(2001/4)

print(f'iter_per_epoch {iter_per_epoch}')

loss_hist = []

for epoch in range(epochs):
    print(f'epoch {epoch}')
    for step in range(iter_per_epoch):
        obs_batch = obs[step:step+batch_size, :]
        acs_batch = acs[step:step+batch_size, :]

        pred_action = agent.policy.get_action(obs_batch)

        loss = F.mse_loss(pred_action, acs_batch)
        loss_hist.append(loss)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()


torch.save(agent, agt_file)
torch.save(loss_hist, loss_file)

plt.plot(loss_hist)
plt.xlabel('Time steps')
plt.ylabel('Loss')
plt.show()

