#prisoners dilemma but with shared q table (lookign for higher cooperation)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Payoff matrix
payoffs = {
    (0, 0): (3, 3),  # C,C
    (0, 1): (0, 5),  # C,D
    (1, 0): (5, 0),  # D,C
    (1, 1): (1, 1),  # D,D
}

ACTIONS = [0, 1]  # 0=C, 1=D
JOINT_ACTIONS = [(0,0),(0,1),(1,0),(1,1)]
STATE_SPACE = [(-1,-1)] + [(a1,a2) for a1 in ACTIONS for a2 in ACTIONS]

# Shared Q-table
Q = {s: np.zeros(len(JOINT_ACTIONS)) for s in STATE_SPACE}

# Hyperparameters
alpha = 0.2
gamma = 0.99
epsilon = 0.2  # joint exploration
tremble = 0.01  # random flip of joint action
n_episodes = 2000
episode_length = 100

# Store frequencies of joint outcomes
outcome_history = []

def select_joint_action(state):
    if random.random() < epsilon:
        return random.choice(range(4))
    return np.argmax(Q[state])

for ep in range(n_episodes):
    state = (-1,-1)
    counts = {"CC":0, "CD":0, "DC":0, "DD":0}

    for t in range(episode_length):
        idx = select_joint_action(state)
        a1, a2 = JOINT_ACTIONS[idx]

        if random.random() < tremble: a1 = 1-a1
        if random.random() < tremble: a2 = 1-a2

        r1, r2 = payoffs[(a1,a2)]
        next_state = (a1, a2)
        reward = r1 + r2
        best_next = np.max(Q[next_state])
        Q[state][idx] += alpha * (reward + gamma * best_next - Q[state][idx])

        if (a1,a2)==(0,0): counts["CC"]+=1
        elif (a1,a2)==(0,1): counts["CD"]+=1
        elif (a1,a2)==(1,0): counts["DC"]+=1
        elif (a1,a2)==(1,1): counts["DD"]+=1

        state = next_state

    # Normalize frequencies
    total = sum(counts.values())
    for k in counts:
        counts[k] /= total
    outcome_history.append(counts)

# show choices over time
fig, ax = plt.subplots()
bars = ax.bar(["CC","CD","DC","DD"], [0,0,0,0], color=["green","blue","red","gray"])
ax.set_ylim(0,1)
ax.set_ylabel("Frequency")
ax.set_title("Shared Q-learning Joint Action Frequencies")

def update(frame):
    counts = outcome_history[frame]
    for bar, k in zip(bars, ["CC","CD","DC","DD"]):
        bar.set_height(counts[k])
    ax.set_xlabel(f"Episode {frame}")
    return bars

ani = animation.FuncAnimation(fig, update, frames=len(outcome_history), interval=50, blit=False)
plt.show()
