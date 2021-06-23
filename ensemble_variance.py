import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_variance():
    obs_file = '/home/giovani/dril/dril/dril/demo_data/obs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
    acs_file = '/home/giovani/dril/dril/dril/demo_data/acs_LunarLanderContinuous-v2_seed=0_ntraj=3.npy'
    ens_file = '/home/giovani/ground_clone/ensemble_512_v0_ntraj=3.pth'

    obs = np.load(obs_file)
    acs = np.load(acs_file)

    obs = torch.Tensor(obs)

    ensemble = torch.load(ens_file)

    variance_hist = []

    for i in range(len(obs)):
        ensemble_actions = []
        for agent in ensemble:
            action_ens = agent(obs[i])
            ensemble_actions.append(action_ens.detach().cpu().numpy())

        ensemble_actions = np.array(ensemble_actions)
        cov = np.cov(ensemble_actions, rowvar=False)
        action = acs[i]

        variance = np.matmul(np.matmul(action, cov), action.T)
        variance_hist.append(variance)

    variance_hist = np.array(variance_hist)
    q = np.quantile(variance_hist, 0.98, axis=0)

    lenght_var = variance_hist.shape[0]
    t = np.linspace(0, lenght_var-1, lenght_var)
    q0 = np.ones_like(t)*q

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t, variance_hist)
    ax.plot(t, q0)
    #plt.show()

    return q


if __name__ == '__main__':
    calculate_variance()
