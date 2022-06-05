from locale import normalize
import os
import argparse
from matplotlib import colors
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gym

# Disable TensorFlow GPU for parallel excecution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
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
        self.weights = agent.model.get_weights() # NNの各層の重みパラメタ

        # with Paralell(n_jobs=-1) as parallel: # -1 は自動設定の意味，コア数に応じて勝手にジョブ割り振る
        #    for e in range(epoch):
        for e in range(epoch):
            #experiment = delayed(EvolutionalTrainer.run_agent)
            results = Parallel(n_jobs=-1)(delayed(EvolutionalTrainer.run_agent)(episode_per_agent, self.weights, self.sigma) for p in range(self.population_size))
            self.update(results)
            self.log()
        
        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        return CatcherObserser(width=50, height=50, frame_count=5)

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        """
        sigma：weightを更新するパラメータ
        """
        env = cls.make_env()
        actions = list(range(env.action_space.n))
        agent = EvolutionalAgent(actions)

        noises = []
        new_weights = []

        # 1. Make weight.
        for w in base_weights:
            noise = np.random.randn(*w.shape) # ndarray.shape はndarrayの行数、*はListのアンパック
            new_weights.append(w + sigma * noise) 
            noises.append(noise)

        # 2. Test Play.
        total_reward = 0

        for e in range(episode_per_agent):
            s = env.reset()
            if agent.model is None:
                agent.initialize(s, new_weights)
            done = False
            step = 0
            while not done and step < max_step:
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                total_reward += reward
                s = n_state
                step += 1

        reward = total_reward / episode_per_agent
        return reward, noises

    def update(self, agent_results):
        """
        agent_results = [[reward_0, noises_0],[reward_1, noises_1], ... [reward_n, noises_n]]
        """
        rewards = np.array([r[0] for r in agent_results])
        noises = np.array([r[1] for r in agent_results]) # 角層の重み* population_size (=20) 分のノイズがまとめて入っている
        normalized_rs = (rewards - rewards.mean()) / rewards.std() # 変化率が大きいものを貢献度が高いとする
        # print("normalized_rs = ",normalized_rs)
        # print("normalized_rs.shape = ",normalized_rs.shape)
        # normalized_rs は　20要素のベクトル
        # 3. Update base weights.
        new_weights = []
        # print("self.weights = ",self.weights)
        
        for i, w in enumerate(self.weights):    # enumerate()は要素とインデックスを返す
        # 層ごとにweightを更新
        # w は横ベクトル (1 \times 3) 
        # normalized_rs は 行列 (N \times 3)，各population に対する重み
            # print("update i = ",i," STEP")
            noise_at_i  = np.array([n[i] for n in noises]) # i層目のノイズ[0:19]
            rate = self.learning_rate / (self.population_size * self.sigma)
            w = w + rate * np.dot(noise_at_i.T, normalized_rs).T    #np.dotは内積
            # normalized_rs は 各population の感度のようなもの，rewardが大きく変化しそうな パラメータを
            # 大きく変化させようということ，noize_at_i はガウシアンノイズなので，方向は考えていない
            # rate: スカラー
            # 
            # print("w = ",w)
            new_weights.append(w)
        # print("new_weights = ",new_weights)
        self.weights = new_weights
        self.reward_log.append(rewards)
   
    def log(self):
        rewards = self.reward_log[-1]
        print("Epoch {}: reward {:.3}(max:{}, min:{})".format(len(self.reward_log), rewards.mean(), rewards.max(), rewards.min()))
        #    ., print("reward = {:.3}".format(12345.6789)) -> reward = 1.23e+04

    def plot_rewards(self):
        np.indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        plt.figure()
        plt.title("Reward History")
        plt.grid()
        plt.fill_bitween(np.indices, means - stds, means + stds, alpha=0.1 , color = "g")
        plt.plot(np.indices, means, "o-", color = "g", label = "reward")
        plt.legend(loc = "best")
        plt.show()       
    
def main(play):
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")

    
    if play:    
        env = EvolutionalTrainer.make_env()    
        agent = EvolutionalAgent.load(env, model_path)    
        agent.play(env, episode_count=5, render=True)
    else:    
        trainer = EvolutionalTrainer()    
        trained = trainer.train()    
        trained.save(model_path)    
        trainer.plot_rewards()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolurional Agent")
    parser.add_argument("--play", action="store_true",help="play with trained model")
    args = parser.parse_args()
    main(args.play)