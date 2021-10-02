def V(s, gamma=0.99):
    V = R(s) + gamma * max_V_on_next_state(s)
    return V

def R(s):
    # 即時報酬 (instant reward)
    # 瞬時コスト e.g., x(k)'*Rx(x) とか
    # ここでは状態にのみ依存，actionには非依存
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    """
    関数の説明はここに書く
    """
    # ゲームが終わったら，0を返す
    # 再帰呼び出し、ゴールしたら次は移動しないのでゼロ
    # print(f"max_V_on_next_state is called. ")
    if s in ["happy_end", "bad_end"]:
        return 0
    
    actions = ["up", "down"] # 迷路じゃない，上下するっぽい
    values = [] 
    for a in actions:
        transition_probs = transit_func(s, a)
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state) # ここでVを呼んでる，
        values.append(v)
    indent = len(s.split("_")[1:])*"    "
    print(f"{indent} state:{s}   ,values:{values}")
    return max(values)

def transit_func(s, a):
    """
    Make next state by adding action atr to state.
    ex: (s = 'state', a = 'up') => 'state_up'
        (s = 'state_up', a = 'down') => 'state_up_down'
    """
    # stateから行動を抜き出す
    actions = s.split("_")[1:] # split: 文字列を_でちょん切る，最初のstateを無視するために[1:]，(0を無視した)
    LIMIT_GAME_COUNT = 3
    HAPPY_END_BORDER = 5
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])
    
    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([1 if a == "up" else 0 for a in actions]) # e.g., actions は ['up', 'down', 'up', 'up']　だと，[1 ... actions] は [1, 0, 1, 1] (リスト内包表記)
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }

if __name__ == "__main__":
    print(V("state"))
    #print(V("state_up_up"))
    #print(V("state_down_down"))

