import numpy as np
import scipy.stats
from scipy.special import logsumexp
from planner import PolicyIterationPlanner
from tqdm import tqdm

class BayesianIRL():
    def __init__(self, env, eta=0.9, prior_mean=0.0, prior_scale=0.5):
        self.env = env
        self.planner = PolicyIterationPlanner(env)
        self.eta = eta
        self._mean = prior_mean
        self._scale = prior_scale
        self.prior_dist = scipy.stats.norm(loc=prior_mean,
                                           scale=prior_scale) # 正規分布 (normal distribution) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        
    def estimate(self, trajectories, epoch=50, gamma=0.3,
                 learning_rate=0.1, sigma=0.05, sample_size=20):
        num_states = len(self.env.states)
        reward = np.random.normal(size=num_states,
                                  loc=self._mean, scale=self._scale)
        
        def get_q(r, g):
            self.planner.reward_func = lambda s:r[s]
            V = self.planner.plan(g)
            Q = self.planner.policy_to_q(V, gamma) #バグ?? gammaとgは同じ値
            return Q
        
        for i in range(epoch):
            noises = np.random.randn(sample_size, num_states)
            # = [n(0); n(1); ... n(sample_size)];
            scores = []
            for n in tqdm(noises):
                _reward = reward + sigma * n # n = n(i) = [n(i,1), ..., n(i,num_states)]
                Q = get_q(_reward, gamma)
                
                # Calculate prior (sum of log prob)
                reward_prior = np.sum(self.prior_dist.logpdf(_r) for _r in _reward)
                
                # Calculate likelihood
                likelihood = self.calculate_likelihood(trajectories, Q)
                
                # Calculate posterior
                posterior = likelihood + reward_prior # log P(R+sigma*n(i)|\zeta) 
                scores.append(posterior)
                
            rate = learning_rate / (sample_size * sigma)
            scores = np.array(scores)
            normalized_scores = (scores - scores.mean()) / scores.std() # 標準正規分布にスケーリング
            noise = np.mean(noises * normalized_scores.reshape((-1, 1)),
                            axis=0)
            reward = reward + rate * noise
            print("At itereation {} posterior={}.".format(i, scores.mean()))
        reward = reward.reshape(self.env.shape)
        return reward
    
    def calculate_likelihood(self, trajectories, Q):
        mean_log_prob = 0.0
        for t in trajectories:
            t_log_prob = 0.0
            for s, a in t:
                expert_value = self.eta * Q[s][a] # スカラーですからー
                total = [self.eta * Q[s][_a] for _a in self.env.actions] # アクションの次元を持つベクトル
                t_log_prob += (expert_value - logsumexp(total)) # expert_value は指数を取ってから対数を取っている expert_value = log(exp(expert_value))
                # P(\zeta|\theta) = exp(R(\zeta)) / \sum(exp(R(\zeta)))
            mean_log_prob += t_log_prob
        mean_log_prob /= len(trajectories)
        return mean_log_prob

if __name__ == "__main__":
    def test_estimate():
        from environment import GridWorldEnv
        env = GridWorldEnv(grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
        ])
        # Train Teacher
        teacher = PolicyIterationPlanner(env)
        teacher.plan()
        trajectories = []
        print("Gather demonstrations of teacher.")
        for i in range(20):
            s = env.reset()
            done = False
            steps = []
            while not done:
                a = teacher.act(s)
                steps.append((s, a))
                n_s, r, done, _ = env.step(a)
                s = n_s
            trajectories.append(steps)
            
        print("Estimate reward.")
        irl = BayesianIRL(env)
        rewards = irl.estimate(trajectories, epoch=100)
        print(rewards)
        env.plot_on_grid(rewards)
    test_estimate()
                