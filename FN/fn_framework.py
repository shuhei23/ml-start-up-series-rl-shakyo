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

'''
    状態(s)
    行動(a)
    報酬(r)
    遷移先の状態(n_s)
    エピソード終了フラグ(d)
'''
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
        """
        policy や q関数の更新とかはせず，ただ動かすだけ
        """
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

    @property
    def trainer_name(self):
        # propertyをつけるとこのメソッドを変数みたいに扱える trainer.trainer_name <-○ / trainer.trainer_name() <-× / trainer.trainer_name = ”hoge” <- × 
        class_name = self.__class__.__name__ # 予約語
        snaked = re.sub("(.)([A-Z][a-z]+)",r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])",r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked
    
    def train_loop(self, env, agent, episode = 200, initial_count = -1, 
                    render = False, observe_interval = 0):
        """
        episode回数分学習する
        """ 
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
                 (self.training_count == 1 or self.training_count % observe_interval == 0):
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

                s = n_state
                step_count += 1
                
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
        """
        countで指定されたデータ数だけ直近のexperiencesを取ってくる
        """
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]

class Observer():
    """
    ラッパー(Wrapper): 本質的な処理以外を隠して、煩雑さを回避するために、本質的な処理＋重複する処理をまとめた関数やクラス(できることは本質的な処理と変わらない)
    具体例:
    ImaiPrint(int dst_buf, int dst_size, int src_buf, int src_size){
        // dst_buf; sizeof() # 本質的な作業ではないけど必要
        // src_buf; sizeof() # 煩雑なので，ここの2行を見えないようにする -> つつんでいる -> ラッパー
        if(dst_size >  src_size)
        {
            sprintf()
        }
    }
    
    継承について:
    class ImaiObserver(Observer): 
    # Observerクラスを継承した ImaiObserver，という，継承もとを親/super(Observerクラス)，継承した先を子(ImaiObserverクラス)
    # Imai1 = ImaiObserver(...);
    # Imai1.transform(...) -> ImaiObserverクラスのtransformが呼ばれる，なかったら，継承元/親のObserverクラスから呼ばれる -> エラー吐かれる
    def transform(self, state): 
        # class ImaiObserver(Observer)を後で作ったときに，def transform(...)をImaiObserverの中につくる
        # ImaiObserver の中に def transform(...) がないと　，ここの transform が呼ばれてエラーとなる
        return あんこ
    """
    def __init__(self, env):
        self._env = env

    @property #propertyをつけるとこのメソッドを変数みたいに扱える trainer.trainer_name <-○ / trainer.trainer_name() <-× / trainer.trainer_name = ”hoge” <- × 
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        # Envクラスのreset()関数の返り値はagentの初期値が返ってくるため変換処理であるtransform()関数をかませている
        # <https://github.com/openai/gym/blob/590f2504a76fa98f3a734a4d8d45d536e86eb5d5/gym/core.py#L61>
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")
    
    def step(self, action):
        """
        env.step()で得られる状態stateはtransform()してからreturnする。
        """
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info # 毎回のagentの状態に対して前処理(transform())を実施したい
        
    def transform(self, state): 
        """
        前処理メソッド:要オーバーライド
        """
        # class ImaiObserver(Observer)を後で作ったときに，def transform(...)をImaiObserverの中につくる
        # ImaiObserver の中に def transform(...) がないと　，ここの transform が呼ばれてエラーとなる
        # オーバーライド：親クラスにあるメソッドを子クラスで再定義することによって、子クラス上で親クラスのメソッドを上書きすること
        raise NotImplementedError("You have to implement transform method")

class Logger():

    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")#C:user/toda/samplelogs
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if dir_name: # dir_name が空じゃないとき
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        self._callback = tf.compat.v1.keras.callbacks.TensorBoard(
                            self.log_dir)

    @property
    def writer(self):
        return self._callback.writer
    
    def set_model(self, model):
        self._callback.set_model(model)
        
    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)
    
    def describe(self, name, values, episode=-1, step=-1):
        """
        valuesの平均値と分散を出力
        """
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))
            
    def plot(self, name, values, interval=10):
        """
        intervalごとの平均と分散を求めて、図を出力
        """
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()

    def write(self, index, name, value):
        """
        TenserBoardへ値出力
        """
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = value
        self.writer.add_summary(summary, index)
        self.writer.flush()
        
    def write_image(self, index, frames):
        """
        エージェントのstateを描画
        Deal with a 'frames' as a list of sequential gray scaled image.
        """
        # Deal with a 'frames' as a list of sequential gray scaled image.
        last_frames = [f[:, :, -1] for f in frames]
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0
        channel = 1  # gray scale
        tag = "frames_at_training_{}".format(index)
        values = []

        for f in last_frames:
            height, width = f.shape
            array = np.asarray(f * scale + offset, dtype=np.uint8)
            image = Image.fromarray(array)
            output = io.BytesIO()
            image.save(output, format="PNG")
            image_string = output.getvalue()
            output.close()
            image = tf.compat.v1.Summary.Image(
                        height=height, width=width, colorspace=channel,
                        encoded_image_string=image_string)
            value = tf.compat.v1.Summary.Value(tag=tag, image=image)
            values.append(value)

        summary = tf.compat.v1.Summary(value=values)
        self.writer.add_summary(summary, index)
        self.writer.flush()