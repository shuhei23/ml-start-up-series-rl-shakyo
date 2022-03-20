import argparse
from collections import deque
import numpy as np
import sklearn
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python import keras as K
from PIL import Image
import gym 
import gym_ple
from fn_framework import FNAgent, Trainer, Observer
tf.compat.v1.disable_eager_execution()

class ActorCriticAgent(FNAgent):
    
    def __init__(self, actions):
        # ActorCriticAgentはepsilon-greedyじゃないからepsilonを使わない
        super().__init__(epsilon = 0.0, actions=actions)
        self._updater = None
        
    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path, custom_objects={
                "SampleLayer" : SampleLayer}) # 食品サンプルのサンプルでない，サンプリング用途のレイヤー
        agent.initialized = True
        return agent
    
    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialization. From now, begin training!")
    
    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        # ↓ここから、Actor、Critic共用
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape,
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                activation="relu"))
        # ↑ここまでActor、Critic共用
        
        # ↓ここから、Actor用
        action_layer = K.layers.Dense(len(self.actions),
                                      kernel_initializer=normal)
        action_evals = action_layer(model.output) # Q 計算
        actions = SampleLayer()(action_evals) # actionをサンプリング
        # ↑は次の処理を一行で書いている
        # sample_layer = SampleLayer()
        # sample_layer(action_evals), インスタンスを関数のように使っている
        # ↑ここまでActor用
        
        # ↓ここから、Critic用
        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output) # V 計算
        # ↑ここまで、Critic用
        
        self.model = K.Model(inputs = model.input,
                             outputs=[actions, action_evals, values])
        
    def set_updater(self, optimizer, value_loss_weight=1.0, entoropy_weight = 0.1):
        actions = tf.compat.v1.placeholder(shape=(None),dtype = "int32") # actionだからint 
        values = tf.compat.v1.placeholder(shape = (None), dtype = "float32")
        # MATLAB でいう syms actions シンボリック変数定義, pythonでいう sympy sym 

        _, action_evals, estimateds = self.model.output
        
        # negative logの略。- log πΘを求める
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = action_evals, labels=actions)  
        # 行動評価action_evalsから行動確率を計算softmax()
        #   softmax(xi) := exp(xi)/(\sum_{k=1}^{K} exp(xk)) ・・・行動価値からsoft_max関数を求めることで行動確率にする
        # 実際に取った行動actionsの確率をとる
        # 対数を取ってマイナスをかける -> エントロピーが小さいと確率分布がほとんど等しい -> エントロピーを大きくしたい -> マイナス符号つけて最小化
        #   行動確率とactionsのクロスエントロピーを取って
        #   crossentropy(x, y) := -\sigma_{d = 1}^{D}\sigma_{k=1}^{K}ydi * log(xdi)
        
        # Advantageを求める
        advantages = values - tf.stop_gradient(estimateds) # 後で -tf.stop_grad*** をとって数値実験して挙動を検証する
        #https://medium.com/programming-soda/advantage%E3%81%A7actor-critic%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E9%9A%9B%E3%81%AE%E6%B3%A8%E6%84%8F%E7%82%B9-a1b3925bc3e6

        policy_loss = tf.reduce_mean(neg_logs * advantages) # 期待値を計算する，neg_logs は各行動に対して出ているのでベクトルになっているっぽい
        value_loss = tf.keras.losses.MeanSquaredError()(values, estimateds)
        action_entoropy = tf.reduce_mean(self.categorical_entropy(action_evals))

        # loss = policy_loss + value_loss_weight * value_loss - entoropy_weight * action_entoropy
        loss = policy_loss + value_loss_weight * value_loss
        loss -= entoropy_weight * action_entoropy
        
        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
    
        self._updater = K.backend.function(input=[self.model.input, actions, values], 
            outputs = [loss, policy_loss, value_loss, tf.reduce_mean(neg_logs), tf.reduce_mean(advantages), action_entoropy],
            updates = updates)
            # loss は無名関数

    def categorical_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims =True)
        # a0 のsoftmax を計算
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis = -1, keepdims = True)
        p0 = ea0 / z0
        # log(\sum ea_0)-a0 の softmax出力の確率を使った期待値
        return tf.reduce_sum(p0*(tf.math.log(z0)-a0),axis = -1)

    def policy(self, s):
        if not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            action, action_evals, values = self.model.predict(np.array([s]))
            # model.predict は Keras に入っているメソッド
            return action[0]

    def estimate(self, s):
        action, action_evals, values = self.model.predict(np.array([s]))
        return values[0][0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])
    
        
class SampleLayer(K.layers.Layer):
    """
    サンプリング用途のレイヤー
    全結合層で計算したQから、SampleLayer()により行動をサンプリングする
    """
    def __init__(self, **kwargs):
        self.output_dim = 1 # 出力の次元は1
        super(SampleLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)

    def call(self, x):  # action_evals (行動価値) が x に来る
        # ノイズのっけて最大のxをとる
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models?hl=ja
        noise = tf.ramdom.uniform(tf.shape(x))
        return tf.argmax(x - tf.math.log(-tf.math.log(noise)), axis = 1)
        # ノイズを乗っける理由の参考記事
        # https://qiita.com/motorcontrolman/items/587b532f7a493dfb591f
        # Gumbel-Max Trick
        # https://data-analytics.fun/2021/04/06/understanding-gumbel-max-trick/
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    

class ActorCriticAgentTest(ActorCriticAgent):
    """
    ActorCriticAgentテスト用クラス
    """
    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        # Actor、Critic共用
        #   本物はCons2D(2次元畳み込み層のモジュール)を追加するけど
        #   テスト用はDense(全結合層のレイヤモジュール)を追加する        
        model = K.Sequential()
        model.add(K.layers.Dense(10, input_shape = feature_shape,
                                kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(10, kernel_initializer=normal, 
                                activation="relu"))

        # Actor固有                        
        #   価値関数を出力する元となるニューラルネットワーク(上記共用部)から
        #   action_evalsを出力するニューラルネット(actor_layer)
        actor_layer = K.layers.Dense(len(self.actions), 
                                        kernel_initializer=normal)
        
        action_evals = actor_layer(model.output)
        actions = SampleLayer()(action_evals)
        
        # Critic固有
        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)

        # モデル作成
        self.model = K.Model(inputs = model.input, 
                            outputs = [actions, action_evals, values])
        
        # returnがないので，インスタンスの中身にmodelが出来あがる感じ

