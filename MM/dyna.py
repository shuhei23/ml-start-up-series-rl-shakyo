import argparse
from importlib_metadata import entry_points
import numpy as np
from collections import defaultdict, Counter
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_points="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})

class DynaAgent():
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.actions = []
        self.value = None
        
    def policy(self, state):
        if np.random.random < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            if sum(self.value[state]) == 0:
                return np.random.randint(len(self.actions))
            else:
                return np.argmax(self.value[state])
    
    def learn(self, env, episode_count=3000, gamma=0.9, learning_rate=0.1,
              steps_in_model=-1, report_interval=100):
        self.actions = list(range(env.action_space.n))
        self.value = defaultdict(lambda: [0] * len(self.actions))
        model = Model(self.actions)
        
        rewards = []
        for e in range(episode_count):
            s = env.reset()
            done = False
            goal_reward = 0
            while not done:
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                
                # 実環境から学習する
                gain = reward + gamma * max(self.value[n_state])
                estimated = self.value[s][a]
                self.value[s][a] += learning_rate * (gain - estimated) # (1/3)実環境で1回学習
                
                # モデルから学習する
                if steps_in_model > 0: # steps_in_model:モデルを使って何回学習させるか
                    model.update(s, a, reward, n_state) # (2/3)行動結果でモデルを学習
                    for s, a, r, n_s in model.simulate(steps_in_model):
                        gain = r + gamma * max(self.value[n_s])
                        estimated = self.value[s][a]
                        self.value[s][a] += learning_rate * (gain - estimated) # (3/3)学習させたモデルでさらに学習
    
                s = n_state            
            else:
                goal_reward = reward

            rewards.append(goal_reward)
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(e, recent.mean()))
                
class Model():

    def __init__(self, actions):
        self.num_actions = len(actions)
        self.transit_count = defaultdict(lambda: [Counter() for a in actions])
        # lambda: [Counter(0), Counter(1), ... ]
        # func(a_vec) = [func(a_vec(0)), func(a_vec(1)), ...]
        self.total_reward = defaultdict(lambda: [0]*self.num_actions)
        # lambda: [0, 0, ...]
        self.history = defaultdict(Counter)
        # defaultdictの引数には、「初期化時に実行する関数」を記述します。ToDO:変数の使用先の処理内容を確認する
        # actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        '''
        self.history[state][action] += 1
        キーがstate, actionの値を+1する。
        そんなキーがなかった場合は、キー生成して値はCounter()で初期化してから+1する。
        
        中身は以下のようになる。 
        history = {
            '0': {       # state1
                '0' : 1, #  LEFT
                '1' : 0, #  DOWN
                '2' : 0, #  RIGHT
                '3' : 0  #  UP
            },
            '1': {
                '0' : 0,
                '1' : 0,
                '2' : 0,
                '3' : 0
            },
        }
        '''

    def update(self, state, action, reward, next_state):
        self.transit_count[state][action][next_state] += 1
        self.total_reward[state][action] += reward
        self.history[state][action] += 1 # 何回その状況(state(string型文字列))で行動(action)をとったか
        print("-- update method is called -- ") # デバッグ用に書いた
        print(state)
        print(action)
        print(self.history)
        # self.history[0][1] += 1 

    def transit(self, state, action):
        counter = self.transit_count[state][action]
        states = []
        counts = []
        for s, c in counter.most_common(): # (要素, 出現回数)という形のタプルを出現回数順に並べたリストを返す
            states.append(s)
            counts.append(c)
        probs = np.array(counts) / sum(counts)
        return np.random.choice(states, p=probs)
    
    def reward(self):
        pass
    def simulate():
        pass