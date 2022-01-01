"""
--------オブジェクト指向のインスタンスとクラスの関係について--------
epsilon1 = 0.001
actions1 = ["up", "down"]
agent1   =      FNAgent(epsilon1, actions1)
->インスタンス  ->クラス
 ここでは agent をインスタンスという，変数とは言わない(数字じゃないんで)
 インスタンスは実体化の意味，いろいろな情報を持った，変数の拡張のような感じ


epsilon2 = 0.002
actions2 = ["left", "right"]
agent2   =       FNAgent(epsilon, actions2)
agent1 と agent2は同じように振る舞うけれど，持っている値は違う，
epsilonも値も違うし，estimate の値とかも違ってくる

agent1 で 100更新 (x[100])，agent2で1更新 (x[1])したら，当然状態(制御でいうと状態変数)は変わってくる
agent1 と agent2 はやんわり一緒のもの(一緒のクラスからできている)だけど，中身は違っている

e.g., 買い物サイトのかご，かごの中身は違っている，かごクラスを定義しておいて，
各人のインスタンスごとに，各人の買いたいものの中身が入ってる

agent1.policy()  ○
FNAgent.policy() x <- コンパイラに怒られる

クラスメソッド(@classmethodを付けたメソッド)
agent1 = FNAgent.load(EpsilonGreedyAgent, env, "/workspace/model", epsilon=0.2)

"""

import os # ファイル名やディレクトリ名をとるライブラリ ??? 
import io 
import re # 正規表現 regular expression 
from collections import namedtuple
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from PIL import Image #画像処理ライブラリ Pillow
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend_config import epsilon


Experience = namedtuple("Experince",
                        ["s", "a", "r", "n_s", "d"])#Cの構造体のような，tuple は中身変更できない


class FNAgent():

    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)


    @classmethod    # インスタンス化していないクラスからクラス.メソッド()でload()を呼び出せる
    def load(cls, env, model_path, epsilon=0.0001):
        """
        cls: FNAgentを継承したエージェントクラス
        """
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions) #デザインパターンでいうFactory Method的な感じ
        # clsはクラスと思われる，agent はインスタンスになる
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError("You have to implemant initialize method.")

    def estimate(self, s):
        raise NotImplementedError("You have to implemant estimate method.")

    def update(self, experiences, gamma):
        raise NotImplementedError("You have to implemant update method.")
    # e.g., class IMAIAgent(FNAgent): <- "FNAgentを継承した"IMAIAgentなるクラス
    #           def update() ... と具体的に実装することになる
    
    def policy(self, s):
        """
        epsilonの確率でランダム行動(探索)
        1-epsilonの確率では、
        　予測するのが行動確率の場合は、その確立に従って行動
        　それ以外は価値最大の行動をとる（活用）
        """
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

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
                print("Get reward {}.".format(episode_reward))

class Trainer():
    
    def __init__(self, buffer_size = 1024, batch_size = 32, 
        gamma = 0.9, report_interval = 10, log_dir = ""):
        self.buffer_size = buffer_size  # Agentの行動履歴を保持するバッファサイズ
        self.batch_size = batch_size    # 1回の学習で取り出すデータ数
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size) # Agentの行動履歴。ためて書き出す感じ
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property #propertyをつけるとこのメソッドを変数みたいに扱える trainer.trainer_name <-○ / trainer.trainer_name() <-× / trainer.trainer_name = ”hoge” <- × 
    def trainer_name(self):
        class_name = self.__class__.__name # 予約語
        snaked = re.sub("(.)([A-Z][a-z]+)",r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])",r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked
    
    def train_loop(self, env, agent, episode = 200, initial_count = -1, 
                    render = False, observe_interval = 0): 
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                if render:
                    env.render()
                if self.training and observe_interval > 0 and \
                 (self.trainging_count == 1 or self.training_count % observe_interval == 0):
                    frames.append(s)
            
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)
                if not self.training and \
                    len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True
            
                self.step(i, step_count, agent, e)
                
            else:   # done == True;
            # https://www.javadrive.jp/python/for/index1.html#section2
                self.episode_end(i, step_count, agent)

                if not self.training and \
                    initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True
                
                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)
                        frames = []
                    self.training_count += 1
                
    def episode_begin(self, episode, agent):
        pass
    
    def begin_train(self, episode, agent):
        pass
    
    def step(self, episode, step_count, agent, experience):
        pass

    def episode_end(self, episode, step_count, agent):
        pass
    
    def is_event(self, count, interval):
        return True if count != 0 and count % interval == 0 else False
                
    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]

