import torch
import numpy as np
from ensemble_policy import Policy


def train():
    obs_file = '/home/giovani/dril/dril/dril/demo_data/obs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
    acs_file = '/home/giovani/dril/dril/dril/demo_data/acs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
    ens_file = '/home/giovani/ground_clone/ensemble_512_v0_ntraj=3.pth'


    obs = np.load(obs_file)
    acs = np.load(acs_file)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    obs = torch.tensor(obs, dtype=torch.float).to(device)
    acs = torch.tensor(acs, dtype=torch.float).to(device)

    def randomize_samples():
        print(f'obs size {obs.size()}, acs size {acs.size()}')
        print(f'obs size {obs.size()[0]}, acs size {acs.size()[0]}')

        length_set = obs.size()[0]

        randomizer = torch.randint(length_set, (length_set,))
        acs_rdm_0 = acs[randomizer, 0]


    randomize_samples()

    epochs = 20
    batch_size = 64
    input_dims = 8
    hidden_units = 512
    n_actions = 2
    learning_rate = 0.002
    iter_per_epoch = int(2001/4)

    loss_fn = torch.nn.MSELoss()
    ensemble = []

    for i in range(5):
        print(f'training agent {i}')
        agent = Policy(input_dims, hidden_units, n_actions, learning_rate)

        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for step in range(iter_per_epoch):
                obs_batch = obs[step:step+batch_size, :]
                acs_batch = acs[step:step+batch_size, :]

                pred_action = agent(obs_batch)

                loss = loss_fn(pred_action, acs_batch)
                agent.optim.zero_grad()
                loss.backward()
                agent.optim.step()

        ensemble.append(agent)

    torch.save(ensemble, ens_file)


if __name__ == '__main__':
    train()