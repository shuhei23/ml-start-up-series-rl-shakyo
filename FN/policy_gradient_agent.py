import os
import random
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Observer, Experience
tf.compat.v1.disable_eager_execution()

# 題材は"CartPole-v0"
# scalar (温度とかスカラー量のスカラーは lar)
# scaler (スケールする人，正規化器)

class PolicyGradientAgent(FNAgent): # クラスFNAgentを"継承"している

    def __init__(self, actions):
        # PolicyGradientAgent uses self policy (epsilon-greedy じゃない)
        super().__init__(epsilon=0.0, actions=actions)
        # FNAgent の コンストラクタ(__init__()の処理のこと)を呼んでいる 
        # actions は [1,2]: 左右みたいなもの
        self.estimate_prob = True
        self.scaler = StandardScaler()
        self._updater = None

    def save(self, model_path):
        """
        model_pathにモデルとスケーラーを保存する
        """
        super().save(model_path) # self.model.save(model_path, overwrite=True, include_optimizer=False) をやった (モデルを保存)
        joblib.dump(self.scaler, self.scaler_path(model_path)) # npzファイルで保存するらしい
        #  scaler (scikit-learn製) estimator (keras, tensorflowに統合されてる) でメーカー違うので Pipeline コマンド使えない

    @classmethod
    def load(cls, env, model_path): 
        """
        model_pathにあるモデルとスケーラーをagentに保存して、agentを返す。
        """
        actions = list(range(env.action_space.n))
        # agent は PolicyGradientAgentのインスタンス
        agent = cls(actions) # cls(actions) は PolicyGradientAgent.__init__(actions) と同義
        agent.model = K.models.loadMmodel(model_path)
        agent.initialized = True
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent
    
    def scaler_path(self, model_path):
        fname, _ = os.path.splittext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):
        states = np.vstack([e.s for e in experiences])
        feature_size = states.shape[1] # stateの列数 = 状態のパラメータ数
        self.model = K.modes.Sequential([
            K.layers.Dense(10,activation = "relu", input_shape=(feature_size,)),
            K.layers.Dense(10,activation = "relu"),
            K.layers.Dense(len(self.action), activation = "softmax")
        ]) 
        self.set_updater(optimizer)
        self.scaler.fit(states) # 正規化
        self.initialized = True
        print("Done initialization. From now, begin training!")    
    
    def set_updater(self, optimizer):
        """
        パラメータ更新するメソッドself._updater()作成
        シンボリックで記述して、実行時は後ろから解析していくように実行する？
        """
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32") # いつ値入る？ -> シンボリックな変数
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1) # action が 1 か 2 かで違いはないこと（名義尺度である）を記述
        # 解釈は
        # fmincon(func) と関数ハンドルを関数におくるとき， 
        # func = @(actions, rewards)(actions^2 + rewards)
        # 行動確率の計算
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis = 1) # 行動確立

        # MATLABのsumコマンドみたいなもの ?? 
        # あとで action_probs を見ましょう，tf.reduce_sumが何らかの変換をしている
        print(action_probs) # 2/13にデバック用に書いておいた
        
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0) # selected_action_probsを最小値1e-10, 最大値1.0で丸め込み
        loss = - tf.math.log(clipped) * rewards # -logπΘ*Q
        loss = tf.reduce_mean(loss)

        # Off-Policy Actor-Critic だったらlossに重み(πΘ/βΘ)をかける必要がある？

        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
        self._updater = K.backend.function(
            inputs = [self.model.input, actions, rewards], 
            ouput = [loss], 
            updates = updates # self._updater の上の updates にデータが溜まっていくっぽい
        )

    def estimate(self, s):
        normalized = self.scaler.transform(s)
        action_probs = self.model.predict(normalized)[0]
        return action_probs

    def update(self, states, actions, rewards):
        """
        戦略πΘのパラメータ更新
        """
        normalizeds = self.scaler.transform(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        self._updater([normalizeds, actions, rewards]) # インスタンスであるoptimizerのローカル関数ハンドルupdatesで勾配の計算を実行


class CartPoleObserver(Observer):

    def transform(self, state):
        return np.array(state).reshape((1,-1)) # -1 は自動で要素数から計算する意味，横ベクトル

class PolicyGradientTrainer(Trainer):

    def __init__(self, buffer_size=256, batch_size=32, gamma=0.9, 
                report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma, 
                        report_interval, log_dir)


    def train(self, env, episode_count = 220, initial_count=-1, render=False):
        actions = list(range(env.action_space.n))
        agent = PolicyGradientAgent(actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent):
        if agent.initialized:
            self.experiences = []

    def make_batch(self, policy_experiences):
        """
        戦略の更新に使うバッチを作る
        """
        # batch_sizeかexperiencesの短いほうを長さにした、バッチを作る。
        length = min(self.batch_size, len(policy_experiences))
        # バッチサイズ分だけ経験をサンプリング（同じ戦略で行動したため、1エピソードの中からランダムに取得）
        batch = random.sample(policy_experiences, length)
        # サンプリングした経験を並べる
        states = np.vstack([e.s for e in batch])
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1,1)) # 縦ベクトルに変換
        rewards = scaler.fit_transform(rewards).flatten() #正規化
        return states, actions, rewards

    def episode_end(self, episode, step_count, agent):
        """
        train_loop()中、1エピソード終わるごとに呼ばれる
        """
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if not agent.initialized:
            # 初期化するまではランダムに動いてデータをためる（policy()参照）
            if len(self.experiences) == self.buffer_size:
                optimizer = K.optimizers.Adam(lr=0.01)
                agent.initialize(self.experiences, optimizer)
                self.training = True

        else:
            policy_experiences = []  # 同じ戦略に基づいたデータで学習したいので1エピソードごとに捨てる
            for t, e in enumerate(self.experiences): 
                # Experience = namedtuple("Experince", ["s", "a", "r", "n_s", "d"])
                s, a, r, n_s, d = e # s = e.s, a = e.a, ... の省略. 取り出す順番はどう決まっているの？
                
                # 割引現在価値を計算
                d_r = [_r * (self.gamma ** i) for i, _r in
                        enumerate(rewards[t:])]
                d_r = sum(d_r)
                d_e = Experience(s, a, d_r, n_s, d) # rewaedをd_rに置き換え
                policy_experiences.append(d_e)

            agent.update(*self.make_batch(policy_experiences)) # *でリストのアンパック、中身を分解してくれる

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)

            
