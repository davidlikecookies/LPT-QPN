import numpy as np

def prep_clf(sim, obs, threshold=0.1):
    obs = np.asarray(obs.cpu().detach().numpy())
    sim = np.asarray(sim.cpu().detach().numpy())
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (sim == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (sim == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (sim == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (sim == 0))

    return hits, misses, falsealarms, correctnegatives


def CSI(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)

    results = (hits / (hits + misses + falsealarms)).mean()

    return results