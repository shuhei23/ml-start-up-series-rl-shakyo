

from jinja2 import pass_eval_context
import numpy as np

class Planner():
    def __init__(self, env, reward_func=None):
        self.env = env
        self.reward_func = reward_func
        if self.reward_func is None:
            self.reward_func = self.env.reward_func
    
    def iniitialize(self):
        self.env.reset()
    
    def transitions_at(self, state, action):
        reward = self.reward_func(state)
        done = self.env.has_done(state) # 現時刻で done かどうか
        transition = []

        if not done: # 現時刻で done していないならば 
            transition_probs = self.env.transit_func(state, action) # e.g., {上: 0.5, 下: 0.2, 右: 0, 左: 0.3}
            for next_state in transition_probs:
                prob = transition_probs[next_state]
                reward = self.reward_func(next_state)
                done = self.env.has_done(next_state) # *****next_stateじゃない？*****
                transition.append((prob, next_state, reward, done))
        else:
            transition.append((1.0, None, reward, done))
        
        for p, n_s, r, d in transition:
            yield p, n_s, r, d
            # 上の d が transition.append((prob, next_state, reward, done)) の done になっているので注目!!!
    
    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method")
    


class PolicyIterationPlanner(Planner):
    
    def __init__(self, env):
        super().__init__(env)
        self.policy = None
        self._limit_count = 1000
        
    def iniitialize(self):
        super().iniitialize() # Planner の init　がよばれるだけ
        self.policy = np.ones((self.env.observation_space.n,
                               self.env.action_space.n))
        #First, take each action uniformly.
        self.policy = self.policy / self.env.action_space.n 
        # e.g., self.env.observation_space.n = 3, self.env.action_space.n = 5 なら 
        # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]] 
    
    def policy_to_q(self, V, gamma):
        Q = np.zeros((self.env.observation_space.n,
                      self.env.action_space.n))
        
        for s in self.env.states:
            for a in self.env.actions: # E[r(s(k+1),a(k))] = \sum_{a_p\in Actions} Pr(a=a_p) E[r(s(k+1),a)|a=a_p]
                action_prob = self.policy[s][a]
                for p, n_s, r, done in self.transitions_at(s, a): # Pr(a=a_p) * E[r(s(k+1),a)|a=a_p]
                    # reward += action_prob * p * (r + gamma * V[n_s] * (not done))
                    if done:
                        Q[s][a] += p * action_prob * r
                    else:
                        Q[s][a] += p * action_prob * (r + gamma * V[n_s])
        return Q
    
    def estimation_by_policy(self, gamma, threshold):
        V = np.zeros(self.env.observation_space.n)
        
        count = 0
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    action_prob = self.policy[s][a]
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += action_prob * p * (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            
            if delta < threshold or count > self._limit_count:
                break
            count += 1
        return V
    
    def act(self, s):
        return np.argmax(self.policy[s])
    
    def plan(self, gamma=0.9, threshold = 0.0001, keep_policy=False):
        """
        計算した報酬のもとで戦略を最適化する
        """
        if not keep_policy:
            self.iniitialize()
        
        count = 0
        while True:
            update_stable = True
            # Estimate expected reward under current policy
            V = self.estimation_by_policy(gamma, threshold)
            
            for s in self.env.states:
                # Get action following to the policy (choose max prob's action)
                policy_action = self.act(s)
                
                # Compare with other actions.
                action_rewards = np.zeros(len(self.env.actions))
                for a in self.env.actions:
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    action_rewards[a] = reward
                best_action = np.argmax(action_rewards)
                if policy_action != best_action:
                    update_stable = False
                
                # Update policy (set best_action prob=1, otherwise=0 (greedy))
                self.policy[s] = np.zeros(len(self.env.actions))
                self.policy[s][best_action] = 1.0
            
            if update_stable or count > self._limit_count:
                # If policy isn't updated, stop iterations.
                break
            count += 1
        return V
