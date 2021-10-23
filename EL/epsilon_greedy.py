import random
import numpy as np

class CoinToss():
    def __init__(self, head_probs, max_episode_steps=30):  # __init__ の箇所はコンストラクタという
        self.head_probs = head_probs    # head：表, tail:裏, コイル数分の配列？ -> 各コインの表がでる確率
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0
    
    def __len__(self): # 要素数
        return len(self.head_probs)
    
    def reset(self):
        self.toss_count = 0
    
    def step(self, action):
        """
        コイン投げるメソッド
        戻り値：報酬reward, 試行終了done
        """
        final = self.max_episode_steps -1
        done = False
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                        Please reset env.")
            
        elif self.toss_count == final:
            done = True

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin does not exist".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob: #random.random():0.0以上1.0未満の浮動小数点数を生成
                reward = 1.0 # 表(head)が出た
            else:
                reward = 0.0 # 裏(tail)が出た
            self.toss_count += 1
            return reward, done

class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = [] #経験で得た各コインの期待値(表の出る確率になってる)の配列

    def policy(self):
        """
        epsilonの確率でランダムにコインを選ぶ(探索)
        (1-epsilon)の確率で期待値が最大のコインを選ぶ(活用)
        """
        coins = range(len(self.V)) # e.g., range(5) =  [0, 1, 2, 3, 4]
        if random.random() < self.epsilon:
            return random.choice(coins) # coinsの要素を同じ確率で選択, e.g., Pr(random.choice(['a', 'b', 'c', 'd', 'e']) = 'a') = 20 percent
        else:
            return np.argmax(self.V) # 配列からVが最大になるインデックスを返す
            # argmax(array([4, 4, 5, 0, 5, 4])) = 2
            # 
    
    def play(self, env):
        """
        CointTossクラス（コイントスゲームの環境）:env
        """
        # Initialize estimation
        N = [0] * len(env) # N は各コインを振った回数(表の回数じゃない)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin] # selected_coinが今までで表が出た経験確率
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1 # 試行ごとにコインを振った回数を足す
            self.V[selected_coin] = new_average 

        return rewards
        
    
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def main():

        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.4, 0.5, 0.8] # epsilonsの要素数だけ試行
        game_steps = list(range(10, 310, 10))# 0 10 20 ... 300 310
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards)) #コイントスゲームの回数を10~310で増やしながら比較するため各ゲームの報酬の平均値で比べる
                # meansの要素 = \sum_{k=1}^N reward(k) / N のこと， means == 1だと，すべて表がでた，ということ
            result["epilon={}".format(e)] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()
    main()









































































































































































































































            