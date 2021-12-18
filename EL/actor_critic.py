import numpy as np
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value

class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1) # -1はepsilon-greedyを使わないの意
        nrow = env.observation_space.n # Q-tableの縦幅 = stateの数っぽい???
        ncol = env.action_space.n # Q-tableの横幅 = actionの数っぽい???
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol)) # uniformは0から1までの一様乱数でQ-table初期化
        
            
    def softmax(self, x):
        """
        多変数のシグモイド関数みたいなもの
        Q値->確率に変換する
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s, actions):
        """
        いまのQ[state]で各actionの起こる確率を定めている
        (確率はsoftmax関数の出力として定めた)
        """
        a = np.random.choice(self.action, 1, p = self.softmax(self.Q[s])) 
        # e.g. Q[2] = [10, 100, 1000] だと， Pr(a=1, s=2) = exp(10) / (exp(10)+exp(100)+exp(1000)) > 0
        # a = argmax(Q[s]) じゃないのがポイント -> 探索ができる?
        return a[0]
    
class Critic():
    
    def __init__(self, env):
        """
        Vをゼロで初期化するだけ
        """
        states = env.observation_space.n
        self.V = np.zeros(states)

class ActorCritic():
    
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
    
    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        
        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)
                
                gain = reward + gamma * critic.V[n_state]
                # cf. SARSA: gain = reward + gamma * self.Q[n_state][n_action] # 次のステップでの移動を考慮している，
                #     Q-learning: gain = reward + gamma * max(self.Q[n_state])
                estimated = critic.V[s]
                td = gain - estimated # TD誤差
                actor.Q[s][a] += learning_rate * td # Q[s][a]はこのstateでこのactionをする良さ
                critic.V[s] += learning_rate * td   # V[s]はこのstateの良さ
                s = n_state
            
            else:
                actor.log(reward)
            
            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)
                
        return actor, critic
    
def train():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")
    actor, critic = trainer.train(env, episode_count=3000)
    show_q_value(actor.Q)
    actor.show_reward_log()

if __name__ == "__main__":
    train()
