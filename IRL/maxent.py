import numpy as np
from planner import PolicyIterationPlanner # われわれがあとで作るファイル
from tqdm import tqdm # プログレスバーを出すライブラリ

class MaxEntIRL():

    def __init__(self, env):
        self.env = env
        self.planner = PolicyIterationPlanner(env)
        
    def estimate(self, trajectories, epoch = 20, learning_rate = 0.01, gamma = 0.9):
        """
        
        """
        state_features = np.vstack([self.env.state_to_feature(s) for s in self.env.states])
        print(state_features)
        theta = np.random.uniform(size=state_features.shape[1])
        teacher_features = self.calculate_expected_feature(trajectories) # $f_expert$
        # $f_\zeta$

        for e in tqdm(range(epoch)):
            # Estimate reward $R(\zeta) = \theta^\top f_\zeta$ を計算
            rewards = state_features.dot(theta.T) # $R(\zeta) = \theta^\top f_\zeta$, 手順1
            # Optimize policy under estimated rewawrds. 
            self.planner.reward_func = lambda s: rewards[s] # 関数ポインタ g = @(x)(rewards[x]) % <- MATLAB
            self.planner.plan(gamma=gamma) # ここで optimize , 手順2
            # Estimate feature under policy 
            features = self.expected_features_under_policy(self.planner.policy, trajectories) # 手順3
            # Update to be closed to teacher, 手順4
            update = teacher_features - features.dot(state_features)
            theta += learning_rate * update

        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        return estimated

    def calculate_expected_feature(self, trajectories): 
        """
        エキスパートの行動(trajectories)から状態遷移の特徴f_expert を計算する.
        特徴f は遷移回数をもとにした確率
        """
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories: 
            for s in t:
                features[s] += 1
        
        features /= len(trajectories)
        return features # 各状態にいた確率をまとめたベクトル, e.g. [0.2, 0.5, 0, 0.3]

    def expected_features_under_policy(self, policy, trajectories):
        """
        パラメーターθ（報酬 R(ζ)）のもとで学習した戦略の状態遷移特徴 f_\zeta の計算
        """
        t_size = len(trajectories)
        states = self.env.states
        transition_probs = np.zeros((t_size, len(states)))
        # 縦方向長さがトラジェクトリーの個数，横方向長さが状態の個数，の行列
        # trajectories 
        # [
        #    [1, 2, 4, 5, 3, 3, 10], 
        #    [1, 2, 2, 4], 
        #    [3, 1, 3 , 3, 4, 2]
        #  ]
        initial_state_probs = np.zeros(len(states))
        for t_state in trajectories:
            initial_state_probs[t_state[0]] += 1
        # initial_state_probs = [2, 0, 1]
        initial_state_probs /= t_size
        # initial_state_probs = [0.66, 0, 0.33]

        for t_idx in range(1, t_size): 
            for prev_s in states: 
                prev_prob = transition_probs[t_idx - 1][prev_s]
                a = self.planner.act(prev_s) # 戦略
                probs = self.env.transit_func(prev_s, a) # 今いるsでaしたときの各状態への遷移確率
                for s in probs: 
                    transition_probs[t_idx][s] += prev_prob * probs[s] # ある時刻での確率分布を求めてるっぽい
        
        total = np.mean(transition_probs, axis =0)
        return total
                
if __name__ == "__main__":
    pass






