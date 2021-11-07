import numpy as np
import matplotlib.pyplot as plt

class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsillon = epsilon
        self.reward_log = []
    
    def policy(self, s, actions):
        """
        epsilonの確率でランダムに行動を選ぶ(探索)
        (1-epsilon)の確率で価値Qが最大の行動をする(活用)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions)) # 0からlen(actions)の整数をランダムに選ぶ，
            # 要するに action を actions からランダムに選ぶ
        else:
            if  s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else: # sum(self.Q[s]) ==0 のときはこっち，情報がないとき???
                # cf. 前のコイン問題では，V(s) = zeros(**) のとき，actionは0(最初の要素)でうまくいってなかった．
                return np.random.randint(len(actions))
    
    def init_log(self):
        self.reward_log = []
    
    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:] # 最後からinterval番目から最後までの要素を出力
            mean = np.round(np.mean(rewards), 3) # np.round(a,n) 小数第n番目で丸める
            std = np.round(np.std(rewards), 3)

            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))

        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means) # means +/- stds を実行するためにnumpy行列形式に変換
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g") # 1σ分の信頼区間を表示
            plt.plot(indices, means, "o-", color="g", label = "Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show
