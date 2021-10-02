from enum import Enum # いなむと読む
import numpy as np

class State():
    # __(ナントカ)__ は，もともと定義されている特別なメソッド(特殊メソッド)である
    # メソッドは変数や値を付けて呼び出す
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        # repr(object):オブジェクトの印字可能な表現を含む文字列を返します。
        return "<State: [{}, {}]>".format(self.row, self.column)
    
    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        # 一意性を担保するため?(今後Pythonの言語仕様に詳しくなったときに見返す)
        return hash((self.row, self.column))

    def __eq__(self, other):
        # これを入れないと，同じ(row, column)の値をもつインスタンスが
        # 別のものと認識される, i.e., (now_state == now_state2) が False
        return self.row == other.row and self.column == other.column

class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2

class Environment():
    # move_probは遷移確率，指定しないと80％
    def __init__(self, grid, move_prob = 0.8):
        # 盤面の値の定義：
        # 0:普通のセル
        # -1: 赤のセル（マイナスの報酬 = 落とし穴、ふんだらゲームオーバー）
        # 1：緑のセル（プラスの報酬 = ゴール）
        # 9: 黒のセル(ブロックセル，エージェントは入れない，)
        # gridは2次元配列
        self.grid = grid
        # 今いる位置
        self.agent_state = State()
        # 移動するときの報酬。なるべく少ない歩数でゴールするために負の値にする。
        self.default_reward = -0.04

        self.move_prob = move_prob
        self.reset() # reset はagent_stateをリセットしている（後で定義）
    
    # property は関数みたいに変数を呼び出せる
    # 直接書き換えはできない
    # gridを入れ直すと，それに応じてrow_length, column_lengthも更新される
    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, 
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # ブロックセルの部分は除いてくっつける
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states
    
    # actionは動きたい方向
    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # ゴールに到達
            return transition_probs

        opposite_direction = Action(action.value * -1 ) # -1 かけると逆方向に動くことになる

        for a in self.actions:
            prob = 0
            if a == action:
                # 行きたい方向と同じ方向はデフォルトの遷移確率
                prob = self.move_prob
            elif a == opposite_direction:
                # 行きたい方向と反対方向は遷移無し
                pass
            else: 
                prob = (1.0 - self.move_prob) / 2.0

            # 候補となる移動先ごとに遷移確率を求める
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                #2方向移動できないセル囲まれている場合に呼び出される
                transition_probs[next_state] += prob
        
        return transition_probs
    
    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False
    
    def _move(self, state, action):
        if not self.can_action_at(state):
            # 下のエクセプションは _move でおこらないようになっているので，
            # バクがあるとかじゃないと，呼び出されない
            raise Exception("Can't move from here!")

        # 今の状態をクローン
        next_state = state.clone()

        # 次の状態へ移動
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        
        # 移動先がgridの範囲外なら、元の位置に戻す
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        
        # 移動先がブロックセルなら、元の位置に戻す
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state
        
        return next_state

    def reward_func(self, state): 
        reward = self.default_reward
        done = False # ゴールに到着したらTrueになる
        # 今のstate がどのブロックにいるのか調べる (0, 1, 9, ...)
        attribute = self.grid[state.row][state.column]
        if attribute == 1: # 緑セル，ゴールのこと
            reward = 1
            done = True # ゴールでおしまい
        elif attribute == -1: # ダメージセル，落とし穴，ふんだらゲームオーバー
            reward = -1
            done = True # ゲームオーバーでおしまい
        
        return reward, done
    def reset(self):
        # pass # プロトタイプ宣言
        self.agent_state = State(self.row_length -1, 0) # 左下スタート, e.g., 3-1 = 2
        return self.agent_state

    def step(self,action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None: # いつ None になんの?
            self.agent_state = next_state
        return next_state, reward, done
    
    # 戻り値3つ next_state, reward, done
    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0: # つぎの行き先がないとき, i.e., can_action_at が Falseのとき
            return None, None, True # doneの意味はゴールと落とし穴のとき???
        
        next_state_candidates = [] # 次のstate の候補のセット，注: テキストではnext_statesとなっている
        probs = [] # 対応する遷移確率のデータのセット
        # transition_probsは，stateがインデックス?キー?，遷移確率の値がペアで入っていたが，
        # dictionary 的なやつ? インデックスがstateに置き換わっている感じ
        # それを2つのセットに分離する感じ(下のnp.random.choiceを使うがため)
        for s in transition_probs:
            next_state_candidates.append(s)
            probs.append(transition_probs[s])

        # 
        next_state = np.random.choice(next_state_candidates, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

if __name__ == "__main__":
    grid = [
            [0, 0, 0, 1], 
            [0, 9, 0, -1], 
            [0, 0, 0, 0]
    ]
    # 座標はゼロはじまり
    env = Environment(grid)
    state = State(2,0)
    print(env.transit_func(state,Action.DOWN))
    # {<State: [1, 0]>: 0.09999999999999998, <State: [2, 0]>: 0.09999999999999998, <State: [2, 1]>: 0.8}

    #state = State(1,0)
    #print(env.transit_func(state,Action.RIGHT))
    # {<State: [0, 0]>: 0.09999999999999998, <State: [2, 0]>: 0.09999999999999998, <State: [1, 0]>: 0.8}