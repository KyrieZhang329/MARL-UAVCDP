import numpy as np

'''
这份代码是以赵世钰老师《强化学习的数学原理》第四章值迭代算法中的grid world游戏为背景设计的一个值迭代算法求解程序
'''
GRID_SHAPE = (2, 2)
ACTIONS = np.array([
    [-1, 0],  # a0: up
    [1, 0],   # a1: down
    [0, -1],  # a2: left
    [0, 1],   # a3: right
])

TARGET_STATE = (0, 1)        # 目标区域
FORBIDDEN_STATE = (1, 0)     # 禁止区域
R_BOUNDARY = -1.0            # 撞到边界的奖励
R_FORBIDDEN = -1.0           # 进入禁止区的奖励
R_TARGET = 1.0               # 进入目标区的奖励
STEP_REWARD = 0.0            # 普通转移奖励
GAMMA = 0.9                  # 折扣因子
THETA = 1e-6                 # 收敛阈值

def is_terminal(state):
    return state == TARGET_STATE or state == FORBIDDEN_STATE

def step(state, action):
    next_pos = np.array(state) + action

    if not (0 <= next_pos[0] < GRID_SHAPE[0] and 0 <= next_pos[1] < GRID_SHAPE[1]):
        return state, R_BOUNDARY, False

    next_state = tuple(next_pos)

    if next_state == FORBIDDEN_STATE:
        return next_state, R_FORBIDDEN, True
    if next_state == TARGET_STATE:
        return next_state, R_TARGET, True

    return next_state, STEP_REWARD, False

V = np.zeros(GRID_SHAPE, dtype=float)       # V(s)
Q = np.zeros(GRID_SHAPE + (len(ACTIONS),))  # Q(s,a)

iteration = 0
while True:
    delta = 0.0
    iteration += 1

    for r in range(GRID_SHAPE[0]):
        for c in range(GRID_SHAPE[1]):
            state = (r, c)

            if is_terminal(state):
                V[state] = R_TARGET if state == TARGET_STATE else R_FORBIDDEN
                Q[state] = R_TARGET if state == TARGET_STATE else R_FORBIDDEN
                continue

            # 计算所有动作的 Q(s,a)
            q_values = []
            for idx, action in enumerate(ACTIONS):
                next_state, reward, terminal = step(state, action)
                if terminal:
                    q_sa = reward
                else:
                    q_sa = reward + GAMMA * V[next_state]
                Q[state + (idx,)] = q_sa
                q_values.append(q_sa)

            new_v = np.max(q_values)
            delta = max(delta, abs(new_v - V[state]))
            V[state] = new_v

    if delta < THETA:
        break

policy = np.argmax(Q, axis=-1)  # 每个状态最优动作的索引

action_symbols = np.array(['up', 'down', 'left', 'right'])
policy_display = action_symbols[policy]

print(f"迭代次数: {iteration}")
print("\n值函数 V(s):")
print(np.round(V, 4))

print("\n最优策略 π*(s) :")
for row in policy_display:
    print('  '.join(row))

