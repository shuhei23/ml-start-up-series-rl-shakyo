from locale import normalize
import os
import argparse
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gym

# Disable TensorFlow GPU for parallel excecution
if os.name == "nt":
    os.eviron["CUDA_VISIBLE_DEVICE"] = "-1"
else:
        os.environ["CUDA_VISIBLE_DEVICE"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
"""
TF_CPP_MIN_LOG_LEVEL: Tensorflow実行時のwarningを消す環境変数
0: 全てのメッセージが出力される（デフォルト）。
1: INFOメッセージが出ない。
2: INFOとWARNINGが出ない。
3: INFOとWARNINGとERRORが出ない。
"""

from tensorflow.python import keras as K

class EvolutionalAgent():

    def __init__(self, actions):
        self.actions = actions
        self.model = None
    
    def save(self, model_path):
        self.model.save(model_path, overwrite = True, include_optimizer=False )
    
    @classmethod # EvolutionalAgent.load(env_shokupan, model_path_currypan) -> 最初の引数は self みたいなもの，入力しない
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent
    
    def initialize(self, state, weights=()):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(3, kernel_size=5, strides=3,
            input_shape=state.shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten()) # 複数の画像を一列にならべてベクトルにする
        model.add(K.layers.Dense(len(self.actions), activation="softmax"))
        
        self.model = model
        if len(weights) > 0:
            self.model.set_weights(weights)
    
    def policy(self, state):
        """
        状態遷移確率に基づいて行動
        """
        action_probs = self.model.predict(np.array([state]))[0] # 2次元配列扱いのベクトルを1次元配列にする
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]
        return action

        """
        ・末尾[0]の考察
        A = np.random.rand(1,5)
        A
        Out[9]: array([[0.43766051, 0.64880078, 0.91807693, 0.7268641 , 0.19006856]]) #2次元arrays
        A[0]
        Out[10]: array([0.43766051, 0.64880078, 0.91807693, 0.7268641 , 0.19006856]) #array

        np.random.choice([3,1,2], size=1, p=[0.99,0.0,0.01])[0]
        Out[12]: 3
        np.random.choice([3,1,2], size=1, p=[0.99,0.0,0.01])
        Out[13]: array([3])
        """
    
    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}".format(episode_reward))

class CatcherObserser():
    
    def __init__(self, width, height, frame_count):
        import gym_ple
        self._env = gym.make("Catcher-v0")
        self.width = width
        self.height = height
        
    @property # クラスメソッドと異なり，インスタンスを作ってからでないと使えない
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.observation_space
    
    def reset(self):
        return self.transform(self._env.reset()) # 初期状態を与えている
    
    def render(self):
        self._env.render(mode="human")
    
    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info
    
    def transform(self, state):
        grayed = Image.fromarray(state).convert("L") # L: 8bit グレイスケール
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=2)
        # https://qiita.com/yoya/items/96c36b069e74398796f3 -> (高さ) * (幅) = (高さ) * (幅) * (スカラー値の色情報)
        return normalized

        """
        expnad_dims()の考察
        b
        Out[23]: 
        array([[0.83710896, 0.06835604], # -> 0.83710896 が [0.83710896, 0.06835604] に置き換わった
               [0.26780558, 0.71478435],
               [0.11456884, 0.60180898],
               [0.05760326, 0.03725969],
               [0.48048988, 0.84207342]])

        b = np.expand_dims(b, axis=2)

        b
        Out[25]: 
        array([[[0.83710896],
                [0.06835604]],
        # 上のが b[0,:,:]
               [[0.26780558],
                [0.71478435]],

               [[0.11456884],
                [0.60180898]],

               [[0.05760326],
                [0.03725969]],

               [[0.48048988],
                [0.84207342]]])


            [[0.11456884, 0.60180898],
             [0.05760326, 0.03725969]]
            [[R:1 G:2 B:3, 0.60180898]]
        """



class EvolutionalTrainer():
    
    def __init__(self, population_size = 20, sigma = 0.5, learning_rate = 0.1, 
                report_interval = 10):
        self.population_size = population_size
        self.sigma = sigma 
        self.learning_rate = learning_rate
        self.weights = ()
        self.reward_log = []
    
    def train(self, epoch=100, episode_per_agent = 1, render = False):
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()

        # with Paralell(n_jobs=-1) as parallel: # -1 は自動設定の意味，コア数に応じて勝手にジョブ割り振る
        #    for e in range(epoch):
        for e in range(epoch):
            experiment = delayed(EvolutionalTrainer.run_agent)
            results = Parallel(n_jobs=-1)(delayed(EvolutionalTrainer.run_agent)(episode_per_agent, self.weights, self.sigma) for p in range(self.population_size))
            self.update(results)
            self.log()
        
        agent.model.set_weights(self.weights)
        return agent