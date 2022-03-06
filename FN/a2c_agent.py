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

        _, action_evals, estimateds = self.model.output
        
        # negative logの略。- log πΘを求める
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = action_evals, labels=actions)  
        
        # Advantabgeを求める
        advantages = values - tf.stop_gradient(estimateds)

        policy_loss = tf.reduce_mean(neg_logs * advantages)
        value_loss = tf.keras.losses.MeanSquaredError()(values, estimateds)
        action_entoropy = tf.reduce_mean(self.categorical_entropy(action_evals))

        # loss = policy_loss + value_loss_weight * value_loss - entoropy_weight * action_entoropy
        loss = policy_loss + value_loss_weight * value_loss
        loss -= entoropy_weight * action_entoropy
        
        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
    
        self._updater = K.backend.function(input=[self.model.input, actions, values], 
            outputs = [loss, policy_loss, value_loss, tf.reduce_mean(neg_logs), tf.reduce_mean(advantages), action_entoropy],
            updates = updates)

    def categorical_entropy(self, logits):
        pass
        
class SampleLayer(K.layers.Layer):
    """
    サンプリング用途のレイヤー
    全結合層で計算したQから、SampleLayer()により行動をサンプリングする
    """
    def __init__(self, **kwargs):
        pass