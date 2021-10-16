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
        
        
class PolicyIterationPlanner(Planner):
    
    def __init__(self, env):
        super().__init__(env)
        self.policy = {} # 中括弧は dictionary
    
    def initialize(self):
        """
            policyをactionの等分の確率で初期化する
        """
        super().initialize()
        self.policy = {}
        acticons = self.env.actions # [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in acticons:
                #up/down/right/leftの1/4
                self.policy[s][a] = 1 / len(acticons) #.policy[s][a]はsにおいてアクションaをとる確率
 

    def estimate_by_policy(self, gamma, threshold):
        """
            policy $\pi$に依存する価値関数$V$を返す
        """
        V = {}
        for s in self.env.states:
            # 各状態の報酬の初期値の期待値は0
            V[s] = 0

        while True:
            # Vの更新が終わるまで反復（どこに移動してもVが変わらなくなったら終わり）
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_probs = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        # prob = Pr(x'|x,a) x,aのもとでのs'への遷移確率
                        # next_state = x'
                        r += action_probs * prob * \
                            (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V
    
    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            """
            actionがキー、価値がバリューのdisctionaryから、価値が最大となるactionを返す
            """
            return max(action_value_dict, key=action_value_dict.get)
        
        while True:
        # policyの更新が終わるまで反復
            update_stable = True
            # 1. 戦略に基づいた価値を求める
            V = self.estimate_by_policy(gamma,threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                policy_action = take_max_action(self.policy[s]) # 戦略ベースの評価で最も価値の高いaction
            
                action_rewards = {} # dictionary
                for a in actions: # actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
                # 各アクションで得られるのrewardから，次のpolicyの更新を考える
                    r = 0 
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += 1.0 * prob * (reward + gamma * V[next_state])
                        # action_prob = Pr(a|s) = 1として計算している（価値ベースの評価をしている）
                        # V は estimate_by_policy で計算ずみ, V は $\pi$(policy)に依存する
                    action_rewards[a] = r 
                best_action = take_max_action(action_rewards) # 価値ベースの評価で最も価値の高いaction
                
                if policy_action != best_action:
                    update_stable = False # updateが起こったときのフラグ
                    # sに関するforの中に入っているので，
                    # すべての s で policy_action == best_action となったときのみ break
                
                for a in self.policy[s]:
                    prob = 1.0 if a == best_action else 0
                    self.policy[s][a] = prob
                    # 一番リワードを大きくしそうなactionを100%選択，他のactionは0%

            if update_stable:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid