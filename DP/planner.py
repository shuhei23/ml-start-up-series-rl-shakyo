class Planner():
    
    def __init__(self, env):
        self.env = env
        self.log = []
    
    def initialize(self):
        self.env.reset()
        self.log = []
    
    def  plan(self, gamma = 0.9, threshold = 0.0001):
        raise Exception("Planner have to implements plan method.")
        # 継承された先でplanが実装されている

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state) # doneは取らない
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        """
        stateがキー、rewardがバリューのdictionaryをリスト(grid)に変換する
        """
        # state_reward_dict にはVの値を入れてる
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            # 配列を作る
            grid.append(row)

        for s in state_reward_dict:
            # 配列にrewardを格納
            grid[s.row][s.column] = state_reward_dict[s]

        return grid

class ValueIterationPlanner(Planner):
    """
    Planner が原型になっていて，書き加えていく感じ
    Planner クラスは設計図... ではない ... (これはクラスとインスタンスでした...)
    Planner は親， Value***Planner は子. 
    """

    def __init__(setf, env):
        super().__init__(env)
    
    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions # 上下右左が入っている
        V = {}
        for s in self.env.states: # env.states は行けるマスがすべて入っている
            V[s] = 0

        while True:
            delta = 0 
            self.log.append(self.dict_to_grid(V))
            for s in V: # for文で dictionary が入ると，keyがでてくる
            # 全部の state について s(stateが入る)のfor文を回している
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                    # prob = T(s^\prime|s,a), next_state = s^\prime, reward = R(s^\prime)
                    # \sum_{s^\prime} T(s^\prime|s,a)(R(s)+\gamma V_i(s^\prime)) を計算
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards) # ここで a が決定
                delta = max(delta, abs(max_reward - V[s])) # 最大の max_reward - V[s] を求めるため
                V[s] = max_reward
                
            if delta < threshold:
                break
        V_grid = self.dict_to_grid(V)
        return V_grid
        
        
