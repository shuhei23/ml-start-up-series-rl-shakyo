import math
from collections import defaultdict
import gym
from el_agent import ELAgent
#from frozen_lake_util import show_q_value
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


class MonteCarloAgent(ELAgent):
    
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        
    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))#https://hackmd.io/yISs3ytnQgKRj940QNjvQw?view#Spaces
        #defaultdictの例：https://docs.python.org/ja/3.6/library/collections.html#defaultdict-examples
        self.Q = defaultdict(lambda: [0] * len(actions)) #actions=[up, down, left, right] Q = { lambda:[0, 0, 0, 0] }
        N = defaultdict(lambda: [0] * len(actions))
        
        for e in range(episode_count):
            s = env.reset() #OpenAI の使用上，reset()しないといけない
            done = False
            experience = []
            while not done:
                if render:
                    env.render() # 描画
                a = self.policy(s, actions) # epsilon-greedy, episode走りおわるまでpolicyはかわらない
                n_state, reward, done, info = env.step(a) # いま走っているエピソードでステップが1進む
                #info {"prob", P}
                #P[s][a] == [(probability, nextstate, reward, done), ...]
                experience.append({"state": s, "action": a, "reward":reward})
                s = n_state
            else:
                self.log(reward)
                
            # ゴールまでいって，各エピソードおわったあとにQ関数更新
            for i, x in enumerate(experience): # リストのインデックスと要素を取得
                s, a = x["state"], x["action"]

                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1 #そのマスでその行動をした回数
                alpha = 1 / N[s][a] 
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=1500)
    #show_q_value(agent.Q)
    agent.show_reward_log()
    
if __name__ == "__main__":
    train()
