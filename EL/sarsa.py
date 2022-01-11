from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value

class SARSAAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        # super().のとつけると，ELAgent の中の__init__が呼ばれる

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        '''
        LEFT = 0
        DOWN = 1
        RIGHT = 2
        UP = 3
        '''

        for e in range(episode_count):
            s = env.reset()
            done = False
            a = self.policy(s, actions)
            while not done:
                if render:
                    env.render()
                n_state, reward, done, info = env.step(a) # 上のaによって1ステップ進む
                # -> なくなった a = self.policy(s, actions)
                n_action = self.policy(n_state, actions) # n_ はNextの意。
                # -> gain = reward + gamma * max(self.Q[n_state])
                gain = reward + gamma * self.Q[n_state][n_action] # 次のステップでの移動を考慮している，
                # 上の更新式では，policyのランダム性がgainの値に反映されている，
                # policy がランダム探索を選んだ場合，Q-learning の gain とは異なる値になる
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

def train():
    agent = SARSAAgent() # monte_carlo.pyとここが違うだけ
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count = 500) # われわれが1500に設定した
    show_q_value(agent.Q)
    agent.show_reward_log()

if __name__ == "__main__":
    train()