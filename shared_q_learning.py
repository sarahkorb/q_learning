import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict

GRID_SIZE = 10
N_EPISODES = 1000
EPISODE_LENGTH = 50

APPLE_REWARD = 10
COLLISION_PENALTY = -2
COOPERATION_BONUS = 2  # Bonus for coordinated behavior

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

# Actions: up, down, left, right, stay
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]
ACTION_NAMES = ['up', 'down', 'right', 'left', 'stay']

def step(pos, action):
    """Move agent within grid boundaries"""
    new_x = max(0, min(GRID_SIZE-1, pos[0] + action[0]))
    new_y = max(0, min(GRID_SIZE-1, pos[1] + action[1]))
    return (new_x, new_y)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class SharedQTable:
    """Shared Q-table that both agents can read from and write to"""
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))  # Track how often state-action pairs are visited
        
    def get_q_value(self, state, action_idx):
        return self.q_table[state][action_idx]
    
    def update_q_value(self, state, action_idx, new_value):
        self.q_table[state][action_idx] = new_value
        self.visit_counts[state][action_idx] += 1
    
    def get_best_action(self, state):
        """Get the action with highest Q-value for given state"""
        q_values = [self.q_table[state][i] for i in range(len(ACTIONS))]
        return np.argmax(q_values)

class CooperativeQLearningAgent:
    def __init__(self, agent_id, shared_q_table, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = EPSILON_START
        
        # Reference to shared Q-table
        self.shared_q = shared_q_table
        
        # Experience tracking
        self.last_state = None
        self.last_action = None
        
    def get_state(self, my_pos, other_pos, apple_pos):
        """Convert positions to state representation - normalized so agents see equivalent situations similarly"""
        # Sort positions to make state representation symmetric
        pos1, pos2 = sorted([my_pos, other_pos])
        return (pos1, pos2, apple_pos)
    
    def choose_action(self, state, other_agent_action=None):
        """Choose action with cooperation consideration"""
        if random.random() < self.epsilon:
            # Exploration: random action
            action_idx = random.randint(0, len(ACTIONS) - 1)
        else:
            # Exploitation: best known action with cooperation bias
            q_values = [self.shared_q.get_q_value(state, i) for i in range(len(ACTIONS))]
            
            # Add cooperation bonus - prefer actions that don't lead to collision
            if other_agent_action is not None:
                my_pos = state[0] if self.agent_id == 1 else state[1]
                other_pos = state[1] if self.agent_id == 1 else state[0]
                
                for i, action in enumerate(ACTIONS):
                    new_my_pos = step(my_pos, action)
                    new_other_pos = step(other_pos, other_agent_action)
                    
                    # Bonus for avoiding collision
                    if new_my_pos != new_other_pos:
                        q_values[i] += COOPERATION_BONUS
            
            action_idx = np.argmax(q_values)
        
        return action_idx, ACTIONS[action_idx]
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value in shared table"""
        current_q = self.shared_q.get_q_value(state, action_idx)
        
        if next_state is not None:
            max_next_q = max([self.shared_q.get_q_value(next_state, i) for i in range(len(ACTIONS))])
        else:
            max_next_q = 0
            
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.shared_q.update_q_value(state, action_idx, new_q)
    
    def learn(self, reward, next_state):
        """Update Q-values based on experience"""
        if self.last_state is not None and self.last_action is not None:
            self.update_q_value(self.last_state, self.last_action, reward, next_state)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

def calculate_cooperative_reward(pos1, pos2, apple_pos, collision_occurred, apple_collected):
    """Calculate additional cooperative rewards"""
    cooperative_reward = 0
    
    # Reward for coordination - being roughly equidistant from apple
    dist1_to_apple = manhattan_distance(pos1, apple_pos)
    dist2_to_apple = manhattan_distance(pos2, apple_pos)
    
    # Small bonus if agents are approaching from different sides
    if abs(dist1_to_apple - dist2_to_apple) <= 2:
        cooperative_reward += 0.1
    
    # Penalty for being too close to each other (crowding)
    agent_distance = manhattan_distance(pos1, pos2)
    if agent_distance == 1:  # Adjacent
        cooperative_reward -= 0.05
    
    return cooperative_reward

def run_cooperative_episode(agent1, agent2, record=False):
    """Run episode with cooperative agents sharing Q-table"""
    # Initialize positions
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
        
        # Agents choose actions (can consider each other's likely moves)
        action1_idx, action1 = agent1.choose_action(state1)
        action2_idx, action2 = agent2.choose_action(state2, action1)  # Agent2 knows Agent1's action
        
        if record and t < 10:
            print(f"Turn {t}: Agent1 at {pos1} chose {ACTION_NAMES[action1_idx]}, "
                  f"Agent2 at {pos2} chose {ACTION_NAMES[action2_idx]}")

        # Store old positions
        old_pos1, old_pos2 = pos1, pos2

        # Execute actions
        new_pos1 = step(pos1, action1)
        new_pos2 = step(pos2, action2)
        
        pos1, pos2 = new_pos1, new_pos2
        
        if record:
            trajectory.append({
                'old_pos': (old_pos1, old_pos2),
                'new_pos': (pos1, pos2),
                'apple': apple,
                'turn': t
            })
        
        # Calculate base rewards
        step_reward1 = step_reward2 = 0
        
        # Check for apple collection
        agent1_got_apple = (pos1 == apple)
        agent2_got_apple = (pos2 == apple)
        collision_occurred = (pos1 == pos2)
        
        episode_ended = False
        
        if agent1_got_apple and agent2_got_apple:
            # Both got apple (simultaneous)
            step_reward1 += APPLE_REWARD * 0.7  # Shared reward
            step_reward2 += APPLE_REWARD * 0.7
            if record and t < 10:
                print("Both agents reached apple simultaneously!")
            
            if collision_occurred:
                step_reward1 += COLLISION_PENALTY
                step_reward2 += COLLISION_PENALTY
            episode_ended = True
            
        elif agent1_got_apple:
            step_reward1 += APPLE_REWARD
            # Give agent2 small reward for cooperation
            step_reward2 += APPLE_REWARD * 0.2
            if record and t < 10:
                print("Agent 1 got the apple! Agent 2 gets cooperation bonus.")
            episode_ended = True
            
        elif agent2_got_apple:
            step_reward2 += APPLE_REWARD
            # Give agent1 small reward for cooperation  
            step_reward1 += APPLE_REWARD * 0.2
            if record and t < 10:
                print("Agent 2 got the apple! Agent 1 gets cooperation bonus.")
            episode_ended = True
            
        elif collision_occurred:
            step_reward1 += COLLISION_PENALTY
            step_reward2 += COLLISION_PENALTY
            collisions += 1
            if record and t < 10:
                print(f"Collision at {pos1}!")
        
        # Add cooperative rewards
        coop_reward = calculate_cooperative_reward(pos1, pos2, apple, collision_occurred, 
                                                  agent1_got_apple or agent2_got_apple)
        step_reward1 += coop_reward
        step_reward2 += coop_reward
        
        # Small negative reward for each step
        step_reward1 -= 0.01
        step_reward2 -= 0.01
        
        rewards1 += step_reward1
        rewards2 += step_reward2
        
        # Get next states
        next_state1 = None if episode_ended else agent1.get_state(pos1, pos2, apple)
        next_state2 = None if episode_ended else agent2.get_state(pos2, pos1, apple)
        
        # Both agents learn from shared experience
        agent1.learn(step_reward1, next_state1)
        agent2.learn(step_reward2, next_state2)
        
        # Store current experience
        agent1.last_state = state1
        agent1.last_action = action1_idx
        agent2.last_state = state2
        agent2.last_action = action2_idx
        
        if episode_ended:
            break

    return rewards1, rewards2, collisions, trajectory

# Initialize cooperative system
print("Initializing cooperative Q-learning agents with shared knowledge...")
shared_qtable = SharedQTable()
agent1 = CooperativeQLearningAgent(agent_id=1, shared_q_table=shared_qtable)
agent2 = CooperativeQLearningAgent(agent_id=2, shared_q_table=shared_qtable)

all_r1, all_r2, all_collisions = [], [], []
all_epsilons = []

print("Starting cooperative training...")
for ep in range(N_EPISODES):
    r1, r2, c, _ = run_cooperative_episode(agent1, agent2, record=False)
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

# Analysis
print(f"\nCooperative Results after {N_EPISODES} episodes:")
print(f"Agent 1 total reward: {sum(all_r1):.1f}")
print(f"Agent 2 total reward: {sum(all_r2):.1f}")
print(f"Combined total reward: {sum(all_r1) + sum(all_r2):.1f}")
print(f"Total collisions: {sum(all_collisions)}")
print(f"Agent 1 positive episodes: {sum(1 for r in all_r1 if r > 0) / N_EPISODES:.2%}")
print(f"Agent 2 positive episodes: {sum(1 for r in all_r2 if r > 0) / N_EPISODES:.2%}")
print(f"Shared Q-table size: {len(shared_qtable.q_table)}")

# Plotting results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(np.cumsum(all_r1), label="Agent 1 (cumulative)", alpha=0.8)
plt.plot(np.cumsum(all_r2), label="Agent 2 (cumulative)", alpha=0.8)
plt.plot(np.cumsum(np.array(all_r1) + np.array(all_r2)), label="Combined", alpha=0.8, linestyle='--')
plt.legend()
plt.title("Cumulative Rewards (Cooperative)")
plt.xlabel("Episode")
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(all_collisions, alpha=0.6, color='red')
plt.title("Collisions per Episode")
plt.xlabel("Episode")
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
window = 50
if len(all_r1) >= window:
    moving_avg_r1 = np.convolve(all_r1, np.ones(window)/window, mode='valid')
    moving_avg_r2 = np.convolve(all_r2, np.ones(window)/window, mode='valid')
    combined_avg = np.convolve(np.array(all_r1) + np.array(all_r2), np.ones(window)/window, mode='valid')
    plt.plot(moving_avg_r1, label="Agent 1", alpha=0.8)
    plt.plot(moving_avg_r2, label="Agent 2", alpha=0.8)
    plt.plot(combined_avg, label="Combined", alpha=0.8, linestyle='--')
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
# Cooperation metric: how often both agents get positive rewards
both_positive = [(r1 > 0 and r2 > 0) for r1, r2 in zip(all_r1, all_r2)]
if len(both_positive) >= 50:
    coop_window = 50
    cooperation_rate = np.convolve(both_positive, np.ones(coop_window)/coop_window, mode='valid')
    plt.plot(cooperation_rate)
    plt.title("Cooperation Rate (Both Agents Positive)")
    plt.xlabel("Episode")
    plt.ylabel("Cooperation Rate")
    plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
# Show reward balance between agents
reward_diff = np.array(all_r1) - np.array(all_r2)
if len(reward_diff) >= 50:
    diff_window = 50
    balanced_rewards = np.convolve(reward_diff, np.ones(diff_window)/diff_window, mode='valid')
    plt.plot(balanced_rewards)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title("Reward Balance (Agent1 - Agent2)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Animation of trained cooperative agents
print("\nRunning one episode with trained cooperative agents...")
agent1.epsilon = 0.0  # No exploration for demo
agent2.epsilon = 0.0
_, _, _, traj = run_cooperative_episode(agent1, agent2, record=True)

if traj:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True)
    ax.set_aspect('equal')

    agent1_dot, = ax.plot([], [], 'bo', markersize=15, label="Agent 1 (Cooperative)")
    agent2_dot, = ax.plot([], [], 'ro', markersize=15, label="Agent 2 (Cooperative)")
    apple_dot, = ax.plot([], [], 'go', markersize=12, label="Apple")
    ax.legend(loc="upper right")
    ax.set_title("Cooperative Q-Learning Agents")

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
                ax.set_title("Cooperative Mission Complete!")
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
        
        ax.set_title(f"Turn {turn} - Cooperative Strategy")
        
        return agent1_dot, agent2_dot, apple_dot

    frames_per_turn = 10
    total_frames = len(traj) * frames_per_turn if traj else 1

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  init_func=init, blit=False, interval=100, repeat=True)
    plt.show()

print("Cooperative training complete!")