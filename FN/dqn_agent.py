from locale import normalize
import random
import argparse
from collections import deque
import numpy as np
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple #https://pygame-learning-environment.readthedocs.io/en/latest/
from fn_framework import FNAgent, Trainer, Observer

class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None             
        self._teacher_model = None
        # _を付けるとfrom func import* で呼ばれることが無くなる。ただclassz._func()だと呼べる

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape #状態stateの行列のサイズを取っている
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse") #KerasのAPI:学習のためのモデルを設定する
        self.initialized = True
        print("Done initialization. From now, begin training!") 

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        # 初期化
        # https://keras.io/ja/initializers/
        # glorot_normal() の箇所を変えると別の方法で初期化される， e.g., initializers.RandomUniformとか
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu")) #フィルタ32枚、フィルタサイズ8×8、フィルタをずらす4
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        # kernel_size と strides が深い層に行くに従って小さくなっている
        
        model.add(K.layers.Flatten()) # 画像のセットをベクトル化
        model.add(K.layers.Dense(256, kernel_initializer=normal,activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal))

        self.model = model
        self._teacher_model = K.models.clone_model(self.model)  # 本体のコピー、updateの度には更新しない。更新はupdate_teacherを読んだときにする。

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):
        """
        モデルの更新
        
        ・exparience には state, action, reward, next_state, done が行動した分だけ入っている
        ・gammma 学習率
        """
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)  # n_stateの価値は一定期間更新しないteacher_modelで推定

        for i, e in enumerate(experiences): # experiences は実際に観測した値
            # iはexperienceのからとったインデックス＝時系列のインデックス，i.e., i = 0,\dots, e.d = trueとなった時刻
            reward = e.r #観測値から報酬(reward)を取得, i.e. r(i) ?? 
            if not e.d: #e.d = trueになるのはゲームが終了したとき
                reward += gamma * np.max(future[i]) #報酬(reward)を更新：次の状態(state)から価値関数を使って推測した報酬(reward)のうち最大となる行動(action)の報酬で更新
                # gamma が1回しかかかっていないのは，次のステップのみ考えているから : one-step ahead prediction 
                # a(i+1) = argmax (future[i] = reward(x(i+1|a(i))) としている
                # fn_framework の policy ではepsilon-greedy で実装されているので整合とれている
            estimateds[i][e.a] = reward # e.a = action[i], estimated(i, action(i))
            # estimateds を教師データとして学習するが，estimateds 自体もモデルによって推定されている
        
        loss = self.model.train_on_batch(states, estimateds) #https://keras.io/ja/models/model/
        # stateは画像、lossはNNにstate入れた結果とestimatedsのノルム?
        
        return loss 

    def update_teacher(self):
        self.__teacher_model.set_weights(self.model.get_weights())
        
class DeepQNetworkAgentTest(DeepQNetworkAgent):
    """
    モデル以外のバグを見つけるためのテストエージェント
    make_modelをオーバーライドして、簡素なモデルに置き換えている
    """
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        
    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal,
                                 activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)
        

class CatcherObserver(Observer):
    
    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width      # 画像の幅
        self.height = height    # 画像の高さ
        self.frame_count = frame_count  # 入力となるフレーム数
        self._frames = deque(maxlen=frame_count) # グレースケールの画像データを格納する長さ frame_count のキュー
    
    def transform(self, state):
        """
        前処理:
        1. 画像をキューに貯める(初回は同じ画像を4枚コピー)
        2. 画像キューをarray[height, widths, frames]にして返す
        """
        # RGBの配列(state)をグレースケールの画像に変換(grayed⇒resized)
        grayed = Image.fromarray(state).convert("L") # L: 8bit グレイスケール
        resized = grayed.resize((self.width, self.height)) # グレイスケール画像のリサイズ
        # グレースケール画像のRBG値を0～1にスケーリング(resized⇒normalized)
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0 #グレイスケールだから 0 ~ 1 の値になっているためfloatにキャストする
        
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames) # dequeのframesをnp.arrayに変換
        feature = np.transpose(feature, (1, 2, 0)) #(frames, widths, height) -> (height, widths, frames)
        return feature