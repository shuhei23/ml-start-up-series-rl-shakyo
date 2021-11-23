import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # cm は colormapを作るためのライブラリ
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})

def show_q_value(Q):
    
    env = gym.make("FrozenLake-v0")
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))
    
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            state_exist = False
            if isinstance(Q, dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True
            
            if state_exist:
                
    