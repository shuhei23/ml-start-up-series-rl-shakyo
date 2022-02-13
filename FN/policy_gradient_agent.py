import os
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

    def initialize(self, experiences, opimizer):
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
    
    def set_updater(self, optimizer)
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1) # action が 1 か 2 かで違いはないこと（名義尺度である）を記述
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis = 1)
        # MATLABのsumコマンドみたいなもの ?? 
        # あとで action_probs を見ましょう，tf.reduce_sumが何らかの変換をしている
        print(action_probs) # 2/13にデバック用に書いておいた
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0) # selected_action_probsを最小値1e-10, 最大値1.0で丸め込み
        loss = - tf.math.log(clipped) * rewards 
        loss = tf.reduce_mean(loss)
        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
        self._updater = K.backend.function(
            inputs = [self.model.input, actions, rewards], 
            ouput = [loss], 
            updates = updates
        )

    def estimate(self, s):
        pass

    def update(self, states, actions, rewards):
        pass
