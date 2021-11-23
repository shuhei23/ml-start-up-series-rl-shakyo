import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # cm は colormapを作るためのライブラリ
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})

def show_q_value(Q):
    """
    +---+---+---+
    |   | u |   |  u: up value
    | l | m | r |  l: left value, r: right value, m: mean value
    |   | d |   |  d: down value
    +---+---+---+
    """
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
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]: # Q.shape[0] はQの縦幅
                state_exist = True
            
            if state_exist:
                
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c + state_size #(_r,_c) = 状態を指定したときの，3*3のマスの真ん中の座標
                # 行動に割り振られる値の定義は↓で定義されている
                # https://github.com/openai/gym/blob/590f2504a76fa98f3a734a4d8d45d536e86eb5d5/gym/envs/toy_text/frozen_lake.py#L10
                reward_map[_r][_c - 1] = Q[s][0] # 左に行くとき
                reward_map[_r - 1][_c] = Q[s][1] # 下に行くとき
                reward_map[_r][_c + 1] = Q[s][2] # 右に行くとき
                reward_map[_r + 1][_c] = Q[s][3] # 上に行くとき
                reward_map[_r][_c] = np.mean(Q[s]) # 各行動を取った時の平均値
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(reward_map, cmap = cm.RdYlGn, interpolation="bilinear", 
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()