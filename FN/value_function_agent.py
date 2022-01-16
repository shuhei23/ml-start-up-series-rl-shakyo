import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import gym
from fn_framework import FNAgent, Trainer, Observer

class ValueFunctionAgent(FNAgent): # FNAgent(親/super)を継承したValueFunctionAgentというクラス

    def save(self, model_path):
        """
        sklearnで学習した学習機をnpzデータ?で保存
        """
        joblib.dump(self.model, model_path) 
    
    @classmethod #インスタンス化しないでも呼び出せる関数
    def load(cls, env, model_path, epsilon = 0.0001):
        actions = list(range(env.action_space.n)) # 左/右
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path) # joblibをつかってscikit-learnによって学習したモデルを読み込み
        agent.initialized = True
        return agent
    # aaa = ValueFunctionAgent.load(clc1, env1, model_path1); 

    def initialize(self, experiences):
        scaler = StandardScaler() # 正規化してくれる fit() を呼ぶ
        estimator = MLPRegressor(hidden_layer_sizes=(10,10), max_iter =1) # ノード数10の隠れ層を2つもつNN
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)]) # pipelineを作ると、そこに設定したものを後からpipelineごしに呼べる<https://dev.classmethod.jp/articles/create_pipeline_scikit-learn_pipeline/>

        states = np.vstack([e.s for e in experiences])
        # Experience = namedtuple("Experince",["s", "a", "r", "n_s", "d"])のsなので，状態変数の値
        self.model.named_steps["scaler"].fit(states)

        self.update([experiences[0]], gamma = 0)    # 学習する前に価値観数の予測を行うと例外が発生するsklearnの仕様回避のため、1件の経験だけで学習しておく。
        self.initialized = True
        print("Done initialization. From now, begin training !")
        # return は，なし
        # TodaInstance = ValueFunctionAgent(...) -> __init__ はやってある
        # ImaiInstance = ValueFunctionAgent(...)
        # ImaiInstance.initialize(experiences1) -> ImaiInstanceが自己研鑽した
        # TodaInstance -> 自己研鑽していない(準備できてない) 
        # 各エピソードごとにインスタンスさくせい... とするのが面倒なので，
        # 各エピソードで代わりに for e in NumberOfEpicodes  TodaInstance.initialize みたいな
    
    def estimate(self, s):
        """
        _predict()と別で準備されている理由は？
        以降それほど使われていない???
        """
        estimated = self.model.predict(s)[0]
        return estimated

    def _predict(self, states):
        """
        価値観数の予測
        """
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            # 学習する前に予測を行うと例外が発生するschikitlearnの仕様に
            # 対応するため            
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size = size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds
    # 上の model.predictとは関係ない
    # from value_function_agent import *
    # in Main.py 
    # TodaInstance = ValueFunctionAgent(...) 
    # aaa = TodaInstance._predict(state1) -> エラー
    # 
    # from value_function_agent import ValueFunctionAgent as vp
    # vp.ValueFuncitonAgent._predict(state1) だと，一応呼べる
    # 
    # cf. 予約語があるときは，hogehoge_ と，後ろにアンダーバーをつける
    # https://qiita.com/kiii142/items/6879cb065ad4c5f0b901#_functionx-%E9%96%A2%E6%95%B0%E5%89%8D%E3%81%AB1%E3%81%A4

    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])
        estimateds = self._predict(states) # Q[s][a] の Q[states][1:end]
        future = self._predict(n_states)

        for i, e in enumerate(experiences): # 要素の順にインデックス番号と要素を取得
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward
        
        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds) # パラメータを更新
    
class CartPoleObsever(Observer):
    
    def transform(self, state):
        """
        stateを1×4のベクトルに成形
        # state = [カートの位置, 加速度, ポールの角度, ポールの倒れる速度（角速度）]
        # 上のは誤植っぽい，state = [カートの位置, 速度, ポールの角度, ポールの倒れる速度（角速度）]
        # cf. https://github.com/openai/gym/wiki/CartPole-v0の Observation 
        """
        return np.array(state).reshape((1, -1)) # ベクトルに変換（reshape((a,-1))でaの要素を持つベクトルに自動計算する）

class ValueFunctionTrainer(Trainer):
    
    def train(self, env, episode_count=220, epsilon=0.1, initial_cout=-1,
              render=False):
        actions = list(range(env.action_space.n)) # env.action_space.n は とれるactionの数
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env, agent, episode_count, initial_cout, render)
        return agent
    
    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)
        
    def step(self, episode, step_cout, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size) # self.experiences から self.batch_size だけランダムに出力(出力はリスト)
            agent.update(batch, self.gamma)
            
    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)] # e = Experience(s, a, reward, n_state, done)
        self.reward_log.append(sum(rewards))
        
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval]
            self.logger.describe("reward", recent_rewards, episode=episode)
                
def main(play):
    env = CartPoleObsever(gym.make("CartPole-v0"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl")
    
    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with train model")
    args = parser.parse_args()
    # コマンドラインで実行時のオプションを自作してるっぽい
    # https://docs.python.org/ja/3/library/argparse.html
    main(args.play)