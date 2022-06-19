import imp
from multiprocessing.spawn import import_main_path, old_main_modules
import os
import argparse
import warnings
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor, MLPClassifer
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
 kwargs={"is_slippery": False})


class TeacherAgent():

    def __init__(self, env, epsilon=0.1):
        self.actions = list(range(env.action_space.n))
        self.epsilon = epsilon
        self.model = None
    
    def save(self, model_path):
        joblib.dump(self.model, model_path)
    
    @classmethod
    def load(cls, env, model_path, epsilon=0.1):
        agent = cls(env, epsilon)
        agent.model = joblib.load(model_path)
        return agent
    
    def initialize(self, state):
        self.model = MLPRegressor(hidden_layer_sizes=(), max_iter=1) # 隠れ層無し
        # predict methodを使うための準備
        dummy_label = [np.random.uniform(size=len(self.actions))] #本来はモデルを更新するための推測値を入れる
        self.model.partial_fit([state], dummy_label) # ダミーのデータでモデルをフィッティングした
        return self
    
    def estimate(self, state):
        """
        状態stateから、状態価値qを返す
        """
        q = self.model.predict([state])[0] # ベクトルだけど，行列(横ベクトル)で返ってくるので，ベクトルにする
        print("q:{}",format(q))
        return q

    def policy(self, state): 
        """
        確率epsilonでランダムなactionを選択
        確率(1-epsilon)でベクトルqを最大化するようなactionを選択
        """
        if np.random.random() < self.epsilon: 
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.estimate(state))    

    @classmethod
    def train(cls, env, episode_count = 3000, gamma = 0.9, 
            initial_epsilon = 1.0, final_epsilon = 0.1, report_interval = 100):
        """
        Teacherの学習、学習済みのTeacherAgentのインスタンスを返す
        """
        agent = cls(env, initial_epsilon).initialize(env.reset())
        rewards = []
        decay = [initial_epsilon - final_epsilon] / episode_count
        for e in range(episode_count): 
            s = env.reset()
            done = False
            goal_reward = 0
            while not done: 
                a = agent.policy(s)
                estimated = agent.estimate(s) # q の値

                n_state, reward, done, info = env.step(a)
                gain = reward + gamma * max(agent.estimate(n_state))

                estimated[a] = gain
                agent.model.partial_fit([s], [estimated])# qは状態価値関数??
                # agentt 内　の method LPRegressor は [s]
                # #-> [estimated] とするNNモデル，
                # estimated は 各 a:action で得られる gain をためたベクトル，上下左右で4要素のベクトル
                s = n_state
            else:
                goal_reward = reward

            rewards.append(goal_reward)
            if e != 0 and e % report_interval ==0: # 

                recent = np.array([rewards[-report_interval:]]) # matlabでいう a[end-10:end]
                print("At episode {}, reward is {}".format(e, recent.mean()))
            agent.epsilon -= decay 

        return agent 


