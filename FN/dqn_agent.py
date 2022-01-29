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
        """
        状態の数に基づいてNNのモデルを設定
        """
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
        self._teacher_model.set_weights(self.model.get_weights())
        
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

class DeepQNetworkTrainer(Trainer):

    def __init__(self, buffer_size=5000, batch_size=32,
                gamma=0.09, initial_epsilon=0.5, final_epsilon=1e-3,
                learning_rate=1e-3, teacher_update_freq=3, report_interval=10,
                log_dir="", file_name=""):
        super().__init__(buffer_size,  batch_size, gamma, report_interval, log_dir)
        # 以下引数の順番で並び替えた
        self.initial_epsilon = initial_epsilon
        self.final_epsilon   = final_epsilon
        self.learning_rate   = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.file_name = file_name if file_name else "dqn_agent.h5"

        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10

    def train(self, env, episode_count=1200, initial_count=200,
            test_mode=False, render = False, observe_interval=100):
        
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = DeepQNetworkAgent(1.0, actions) # DeepQNetworkAgent は本来のNN, 1.0はepsilonの値
        else:
            agent = DeepQNetworkAgentTest(1.0, actions) # DeepQNetworkAgetTestはただのパーセプトロン
            observe_interval = 0

        self.training_episode = episode_count
        
        self.train_loop(env, agent, episode_count, initial_count, render, observe_interval)
        # initial_count > i のときはNNの更新をしない

        return agent
        
    def episode_begin(self, episode, agent):
        """
        エピソード開始時処理
        ロスをゼロにする
        """
        self.loss = 0
    
    def begin_train(self, episode, agent):
        """
        トレイン開始時処理
        NNのモデル生成、optimazer設定
        """
        # https://keras.io/ja/optimizers/ ここから選べる
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)

        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode # trainしないでデータ貯める間は引いてない

    def step(self, episode, step_count, agent, experience):
        """
        step毎の処理
        train中だったら、モデル更新
        """
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    
    def episode_end(self, episode, step_count, agent):
        """
        エピソードが終わったときの処理
        ・train中は直近step_count数のエピソードの報酬がより良い場合はモデルを保存する
        ・update_teacher()
        ・epsilon更新する
        ・コンソールに結果出力
        """
        reward = sum([e.r for e in self.get_recent(step_count)])    # 直近step_count数分のrewardを合計
        self.loss /= step_count
        self.reward_log.append(reward)

        if self.training:
            self.logger.write(self.training_count, "loss", self.loss)
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "epsilon", agent.epsilon)

            if reward > self._max_reward:
                agent.save(self.logger.path(self.file_name)) # 報酬がよりよいモデルの時は保存する
                self._max_reward = reward

            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()
        
            # epsilonの更新(学習が進むにつれ"ランダムに動く確率"を減らす)
            # 反比例の図の第四象限のように減る
            # agent.epsilon - diff/(self.training_episode)
            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)
        
def main(play, is_test):
    """
    play：play  with trained model
    is_test：train by test mode
    """
    file_name = "dqn_agent.h5" if not is_test else "dqn_agent_test.h5"
    trainer = DeepQNetworkTrainer(file_name = file_name)
    path = trainer.logger.path_of(trainer.file_name)

    if is_test:
        print("Train on test mode")
        obs = gym.make("Cart-Pole-v0")
        agent_class = DeepQNetworkAgentTest
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)
        agent_class = DeepQNetworkAgent #インスタンス生成じゃなくてクラスのポインタ渡し的な書き方
    
    if play:
        agent = agent_class.load(obs, path) # load()は@classmethodなのでインスタンス生成してなくても呼べる
        agent.play(obs, render=True)
    else:
        trainer.train(obs, test_mode=is_test)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true", help="play  with trained model")
    parser.add_argument("--test", action="store_true", help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
