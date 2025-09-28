# Implementation of Prisoners Dilema and q_leanring, but with separate q tables for each agent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Payoffs
payoffs = {
    (0, 0): (3, 3),   # C, C
    (0, 1): (0, 5),   # C, D
    (1, 0): (5, 0),   # D, C
    (1, 1): (1, 1),   # D, D
}
ACTIONS = [0, 1]  # 0 = Cooperate, 1 = Defect
STATE_SPACE = [(a1, a2) for a1 in ACTIONS for a2 in ACTIONS] + [(-1, -1)]

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.Q = {s: np.zeros(len(ACTIONS)) for s in STATE_SPACE}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

# Training parameters
n_episodes = 1000
episode_length = 100
eps_decay = 0.995
min_eps = 0.05

agent1, agent2 = QLearningAgent(), QLearningAgent()

# Store frequencies of outcomes per episode
outcome_history = []

for ep in range(n_episodes):
    state = (-1, -1)
    counts = {"CC":0, "CD":0, "DC":0, "DD":0}
    
    for t in range(episode_length):
        a1 = agent1.select_action(state)
        a2 = agent2.select_action(state)
        
        r1, r2 = payoffs[(a1, a2)]
        next_state = (a1, a2)
        
        agent1.update(state, a1, r1, next_state)
        agent2.update(state, a2, r2, next_state)
        
        # Count joint outcome
        if (a1, a2) == (0,0): counts["CC"] += 1
        elif (a1, a2) == (0,1): counts["CD"] += 1
        elif (a1, a2) == (1,0): counts["DC"] += 1
        elif (a1, a2) == (1,1): counts["DD"] += 1
        
        state = next_state
    
    # Normalize frequencies
    total = sum(counts.values())
    for k in counts:
        counts[k] /= total
    outcome_history.append(counts)
    
    # Decay epsilon
    agent1.epsilon = max(min_eps, agent1.epsilon * eps_decay)
    agent2.epsilon = max(min_eps, agent2.epsilon * eps_decay)

# Show choices over time 
fig, ax = plt.subplots()
bars = ax.bar(["CC", "CD", "DC", "DD"], [0,0,0,0], color=["green","blue","red","gray"])
ax.set_ylim(0,1)
ax.set_ylabel("Frequency")
ax.set_title("Joint Action Frequencies Over Training")

def update(frame):
    counts = outcome_history[frame]
    for bar, k in zip(bars, ["CC","CD","DC","DD"]):
        bar.set_height(counts[k])
    ax.set_xlabel(f"Episode {frame}")
    return bars

ani = animation.FuncAnimation(fig, update, frames=len(outcome_history), interval=50, blit=False)
plt.show()

