#Agents with q learning 
#typically just get stuck in safe space somewhere, not approaching apple
#Still looking into this. 

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict


GRID_SIZE = 10
N_EPISODES = 1000
EPISODE_LENGTH = 50


APPLE_REWARD = 10 #trying to increase reward to force agents towards apple 
COLLISION_PENALTY = -2

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

# Posisble Actions: up, down, left, right, stay
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]
ACTION_NAMES = ['up', 'down', 'right', 'left', 'stay']

def step(pos, action):
    """Move agent within grid boundaries"""
    new_x = max(0, min(GRID_SIZE-1, pos[0] + action[0]))
    new_y = max(0, min(GRID_SIZE-1, pos[1] + action[1]))
    return (new_x, new_y)

class QLearningAgent:
    def __init__(self, agent_id, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = EPSILON_START
        
        # Q-table: state -> action -> value
        # State = (my_pos, other_agent_pos, apple_pos)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience tracking
        self.last_state = None
        self.last_action = None
        
    def get_state(self, my_pos, other_pos, apple_pos):
        """Convert positions to state representation"""
        return (my_pos, other_pos, apple_pos)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Exploration: random action
            action_idx = random.randint(0, len(ACTIONS) - 1)
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state][i] for i in range(len(ACTIONS))]
            action_idx = np.argmax(q_values)
        
        return action_idx, ACTIONS[action_idx]
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value using Q-learning formula"""
        # Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action_idx]
        
        if next_state is not None:
            max_next_q = max([self.q_table[next_state][i] for i in range(len(ACTIONS))])
        else:
            max_next_q = 0  # Terminal state
            
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
    
    def learn(self, reward, next_state):
        """Update Q-values based on experience"""
        if self.last_state is not None and self.last_action is not None:
            self.update_q_value(self.last_state, self.last_action, reward, next_state)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def run_episode(agent1, agent2, record=False):
    while True:
        pos1 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        pos2 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        apple = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        
        if pos1 != pos2 and pos1 != apple and pos2 != apple:
            break
    
    rewards1, rewards2 = 0, 0
    collisions = 0
    trajectory = []
    
    agent1.last_state = None
    agent1.last_action = None
    agent2.last_state = None  
    agent2.last_action = None

    for t in range(EPISODE_LENGTH):
        # Get current states for both agents
        state1 = agent1.get_state(pos1, pos2, apple)
        state2 = agent2.get_state(pos2, pos1, apple)
        
        # Choose actions
        action1_idx, action1 = agent1.choose_action(state1)
        action2_idx, action2 = agent2.choose_action(state2)
        
        if record and t < 10:  # Only print first 10 turns for readability
            print(f"Turn {t}: Agent1 at {pos1} chose {ACTION_NAMES[action1_idx]}, "
                  f"Agent2 at {pos2} chose {ACTION_NAMES[action2_idx]}")

        # Storing old positions
        old_pos1, old_pos2 = pos1, pos2

        new_pos1 = step(pos1, action1)
        new_pos2 = step(pos2, action2)
        
        # Update positions
        pos1, pos2 = new_pos1, new_pos2
        
        if record:
            trajectory.append({
                'old_pos': (old_pos1, old_pos2),
                'new_pos': (pos1, pos2),
                'apple': apple,
                'turn': t
            })
        
        # Calculate rewards 
        step_reward1 = step_reward2 = 0
        
        # Check for apple collection
        agent1_got_apple = (pos1 == apple)
        agent2_got_apple = (pos2 == apple)
        
        # Check for collision
        collision_occurred = (pos1 == pos2)
        
        episode_ended = False
        
        if agent1_got_apple and agent2_got_apple:
       
            step_reward1 += APPLE_REWARD
            step_reward2 += APPLE_REWARD
            if record and t < 10:
                print("Both agents reached apple simultaneously!")
            
   
            step_reward1 += COLLISION_PENALTY
            step_reward2 += COLLISION_PENALTY
            if record and t < 10:
                print("...but they collided!")
            episode_ended = True
            
        elif agent1_got_apple:
            step_reward1 += APPLE_REWARD
            if record and t < 10:
                print("Agent 1 got the apple!")
            if collision_occurred:
                step_reward1 += COLLISION_PENALTY
                step_reward2 += COLLISION_PENALTY
            episode_ended = True
            
        elif agent2_got_apple:
            step_reward2 += APPLE_REWARD
            if record and t < 10:
                print("Agent 2 got the apple!")
            if collision_occurred:
                step_reward1 += COLLISION_PENALTY
                step_reward2 += COLLISION_PENALTY
            episode_ended = True
            
        elif collision_occurred:

            step_reward1 += COLLISION_PENALTY
            step_reward2 += COLLISION_PENALTY
            collisions += 1
            if record and t < 10:
                print(f"Collision at {pos1}!")
        
        # Add small negative reward for each step to encourage efficiency (again trying to avoid them getting stuck..)
        step_reward1 -= 0.01
        step_reward2 -= 0.01
        
        rewards1 += step_reward1
        rewards2 += step_reward2
        
        # Get nexs state
        next_state1 = None if episode_ended else agent1.get_state(pos1, pos2, apple)
        next_state2 = None if episode_ended else agent2.get_state(pos2, pos1, apple)
        
        # Agents learn
        agent1.learn(step_reward1, next_state1)
        agent2.learn(step_reward2, next_state2)
        
        # Store current experience for next update
        agent1.last_state = state1
        agent1.last_action = action1_idx
        agent2.last_state = state2
        agent2.last_action = action2_idx
        
        if episode_ended:
            break

    return rewards1, rewards2, collisions, trajectory

#Simulation
print("Initializing Q-learning agents...")
agent1 = QLearningAgent(agent_id=1)
agent2 = QLearningAgent(agent_id=2)

all_r1, all_r2, all_collisions = [], [], []
all_epsilons = []

print("Starting training...")
for ep in range(N_EPISODES):
    r1, r2, c, _ = run_episode(agent1, agent2, record=False)
    all_r1.append(r1)
    all_r2.append(r2)
    all_collisions.append(c)
    
    # Decay exploration for both agents
    agent1.decay_epsilon()
    agent2.decay_epsilon()
    all_epsilons.append(agent1.epsilon)
    
    if ep % 100 == 0:
        recent_r1 = np.mean(all_r1[-100:]) if len(all_r1) >= 100 else np.mean(all_r1)
        recent_r2 = np.mean(all_r2[-100:]) if len(all_r2) >= 100 else np.mean(all_r2)
        recent_c = np.mean(all_collisions[-100:]) if len(all_collisions) >= 100 else np.mean(all_collisions)
        
        print(f"Episode {ep}: Agent1 avg reward: {recent_r1:.3f}, "
              f"Agent2 avg reward: {recent_r2:.3f}, "
              f"Avg collisions: {recent_c:.2f}, "
              f"Epsilon: {agent1.epsilon:.3f}")

# --- Analysis ---
print(f"\nFinal Results after {N_EPISODES} episodes:")
print(f"Agent 1 total reward: {sum(all_r1):.1f}")
print(f"Agent 2 total reward: {sum(all_r2):.1f}")
print(f"Total collisions: {sum(all_collisions)}")
print(f"Agent 1 positive episodes: {sum(1 for r in all_r1 if r > 0) / N_EPISODES:.2%}")
print(f"Agent 2 positive episodes: {sum(1 for r in all_r2 if r > 0) / N_EPISODES:.2%}")
print(f"Final exploration rate: {agent1.epsilon:.3f}")


plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.plot(np.cumsum(all_r1), label="Agent 1 (cumulative)", alpha=0.8)
plt.plot(np.cumsum(all_r2), label="Agent 2 (cumulative)", alpha=0.8)
plt.legend()
plt.title("Cumulative Rewards")
plt.xlabel("Episode")
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(all_collisions, alpha=0.6)
plt.title("Collisions per Episode")
plt.xlabel("Episode")
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 3)
window = 50
if len(all_r1) >= window:
    moving_avg_r1 = np.convolve(all_r1, np.ones(window)/window, mode='valid')
    moving_avg_r2 = np.convolve(all_r2, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg_r1, label="Agent 1", alpha=0.8)
    plt.plot(moving_avg_r2, label="Agent 2", alpha=0.8)
    plt.legend()
    plt.title(f"Moving Average Rewards (window={window})")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(all_epsilons)
plt.title("Exploration Rate (Epsilon)")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
# Plot recent performance
recent_episodes = 200
if len(all_r1) >= recent_episodes:
    recent_r1 = all_r1[-recent_episodes:]
    recent_r2 = all_r2[-recent_episodes:]
    plt.hist([recent_r1, recent_r2], bins=20, alpha=0.7, label=['Agent 1', 'Agent 2'])
    plt.legend()
    plt.title(f"Reward Distribution (Last {recent_episodes} Episodes)")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

plt.subplot(2, 3, 6)
# Learning progress
batch_size = 100
batch_means_1 = [np.mean(all_r1[i:i+batch_size]) for i in range(0, len(all_r1), batch_size)]
batch_means_2 = [np.mean(all_r2[i:i+batch_size]) for i in range(0, len(all_r2), batch_size)]
batch_episodes = [i + batch_size//2 for i in range(0, len(all_r1), batch_size)]

plt.plot(batch_episodes, batch_means_1, 'o-', label='Agent 1', alpha=0.8)
plt.plot(batch_episodes, batch_means_2, 's-', label='Agent 2', alpha=0.8)
plt.legend()
plt.title('Learning Progress (Batch Averages)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Animation of One Episode ---
print("\nRunning one episode with trained agents for animation...")
agent1.epsilon = 1.0
agent2.epsilon = 1.0
_, _, _, traj = run_episode(agent1, agent2, record=True)

if traj:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True)
    ax.set_aspect('equal')

    agent1_dot, = ax.plot([], [], 'bo', markersize=15, label="Agent 1 (Q-learner)")
    agent2_dot, = ax.plot([], [], 'ro', markersize=15, label="Agent 2 (Q-learner)")
    apple_dot, = ax.plot([], [], 'go', markersize=12, label="Apple")
    ax.legend(loc="upper right")
    ax.set_title("Q-Learning Agents in Gridworld")

    def init():
        agent1_dot.set_data([], [])
        agent2_dot.set_data([], [])
        apple_dot.set_data([], [])
        return agent1_dot, agent2_dot, apple_dot

    def update(frame):
        if len(traj) == 0:
            return agent1_dot, agent2_dot, apple_dot
            
        frames_per_turn = 10
        turn_index = frame // frames_per_turn
        sub_frame = frame % frames_per_turn
        
        if turn_index >= len(traj):
            if len(traj) > 0:
                final_move = traj[-1]
                a1_pos = final_move['new_pos'][0]
                a2_pos = final_move['new_pos'][1]
                apple = final_move['apple']
                agent1_dot.set_data([a1_pos[0]], [a1_pos[1]])
                agent2_dot.set_data([a2_pos[0]], [a2_pos[1]])
                apple_dot.set_data([apple[0]], [apple[1]])
                ax.set_title("Game Complete - Q-Learning Agents")
            return agent1_dot, agent2_dot, apple_dot
        
        move = traj[turn_index]
        old_a1, old_a2 = move['old_pos']
        new_a1, new_a2 = move['new_pos']
        apple = move['apple']
        turn = move['turn']
        
        progress = sub_frame / (frames_per_turn - 1) if frames_per_turn > 1 else 1
        
        a1_x = old_a1[0] + (new_a1[0] - old_a1[0]) * progress
        a1_y = old_a1[1] + (new_a1[1] - old_a1[1]) * progress
        
        a2_x = old_a2[0] + (new_a2[0] - old_a2[0]) * progress
        a2_y = old_a2[1] + (new_a2[1] - old_a2[1]) * progress
        
        agent1_dot.set_data([a1_x], [a1_y])
        agent2_dot.set_data([a2_x], [a2_y])
        apple_dot.set_data([apple[0]], [apple[1]])
        
        ax.set_title(f"Turn {turn} - Q-Learning in Action")
        
        return agent1_dot, agent2_dot, apple_dot

    frames_per_turn = 10
    total_frames = len(traj) * frames_per_turn if traj else 1

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  init_func=init, blit=False, interval=100, repeat=True)
    plt.show()

print(f"\nQ-table sizes: Agent1: {len(agent1.q_table)}, Agent2: {len(agent2.q_table)}")
print("Training complete!")