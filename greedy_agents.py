# Greedy agent implementation
# 2 agents trying to get an apple. Lose points for collision
# Greedy implementaiton means always take closest steps to apple, dont consider collisions

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


GRID_SIZE = 10
N_EPISODES = 200
EPISODE_LENGTH = 50

APPLE_REWARD = 1
COLLISION_PENALTY = -2

# Possible Actions: up, down, left, right, stay
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]

def step(pos, action):
    """Move agent within grid boundaries"""
    new_x = max(0, min(GRID_SIZE-1, pos[0] + action[0]))
    new_y = max(0, min(GRID_SIZE-1, pos[1] + action[1]))
    return (new_x, new_y)


def run_episode(policy="random", record=False):
    #Random placement
    while True:
        agent1 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        agent2 = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        apple = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
        
        if agent1 != agent2 and agent1 != apple and agent2 != apple:
            break
    
    rewards1, rewards2 = 0, 0
    collisions = 0
    trajectory = []

    for t in range(EPISODE_LENGTH):
        # Random or greedy policy
        def choose_action(agent, apple):
            if policy == "greedy":
                dx = apple[0] - agent[0]
                dy = apple[1] - agent[1]
                
                # If already at apple, stay put
                if dx == 0 and dy == 0:
                    return (0, 0)
                
                # Choose direction based on larger distance, with randomization for ties
                if abs(dx) > abs(dy):
                    return (np.sign(dx), 0)
                elif abs(dy) > abs(dx):
                    return (0, np.sign(dy))
                else:
                    # Equal distance --> just randomize
                    return random.choice([(np.sign(dx), 0), (0, np.sign(dy))])
            else:
                return random.choice(ACTIONS)

        a1 = choose_action(agent1, apple) #agents pick aciton/move simultaneously 
        a2 = choose_action(agent2, apple)
        
        print(f"Turn {t}: Agent1 action: {a1}, Agent2 action: {a2}")

        # Store old positions for display
        old_agent1 = agent1
        old_agent2 = agent2

        new_agent1 = step(agent1, a1)
        new_agent2 = step(agent2, a2)
        
        # update pos
        agent1 = new_agent1
        agent2 = new_agent2
        
        print(f"New positions: Agent1: {agent1}, Agent2: {agent2}")
        
        if record:
            # Record the movement for smooth animation from old to new
            trajectory.append({
                'old_pos': (old_agent1, old_agent2),
                'new_pos': (agent1, agent2),
                'apple': apple,
                'turn': t
            })
        
        #Check if any got the apple
        agent1_got_apple = (agent1 == apple)
        agent2_got_apple = (agent2 == apple)
        
        # Checking for collision
        collision_occurred = (agent1 == agent2)
        if collision_occurred:
            collisions += 1
            print(f"Collision at {agent1}!")
        
        if agent1_got_apple and agent2_got_apple:
            rewards1 += APPLE_REWARD
            rewards2 += APPLE_REWARD
            print("Both agents reached apple simultaneously!")
            
    
            rewards1 += COLLISION_PENALTY
            rewards2 += COLLISION_PENALTY
            print("...but they collided, so both lose collision penalty!")
            break
        elif agent1_got_apple:
            rewards1 += APPLE_REWARD
            print("Agent 1 got the apple!")
            if collision_occurred:
                rewards1 += COLLISION_PENALTY
                rewards2 += COLLISION_PENALTY
            break
        elif agent2_got_apple:
            rewards2 += APPLE_REWARD
            print("Agent 2 got the apple!")
            if collision_occurred:
                rewards1 += COLLISION_PENALTY
                rewards2 += COLLISION_PENALTY
            break
        elif collision_occurred:
            rewards1 += COLLISION_PENALTY
            rewards2 += COLLISION_PENALTY

    return rewards1, rewards2, collisions, trajectory

# --- Run Simulation ---
#record all for summary 
all_r1, all_r2, all_collisions = [], [], []
total_episodes = 0

for ep in range(N_EPISODES):
    r1, r2, c, _ = run_episode(policy="greedy", record=False)
    all_r1.append(r1)
    all_r2.append(r2)
    all_collisions.append(c)
    total_episodes += 1
    
    if ep % 50 == 0:
        print(f"Episode {ep}: Agent1 avg reward: {np.mean(all_r1[-50:]):.2f}, "
              f"Agent2 avg reward: {np.mean(all_r2[-50:]):.2f}, "
              f"Avg collisions: {np.mean(all_collisions[-50:]):.2f}")

# --- Analysis ---
print(f"\nFinal Results after {N_EPISODES} episodes:")
print(f"Agent 1 total reward: {sum(all_r1)}")
print(f"Agent 2 total reward: {sum(all_r2)}")
print(f"Total collisions: {sum(all_collisions)}")
print(f"Agent 1 win rate: {sum(1 for r in all_r1 if r > 0) / N_EPISODES:.2%}")
print(f"Agent 2 win rate: {sum(1 for r in all_r2 if r > 0) / N_EPISODES:.2%}")

# --- Plot Rewards ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(np.cumsum(all_r1), label="Agent 1 (cumulative)")
plt.plot(np.cumsum(all_r2), label="Agent 2 (cumulative)")
plt.legend()
plt.title("Cumulative Rewards")
plt.xlabel("Episode")

plt.subplot(1, 3, 2)
plt.plot(all_collisions)
plt.title("Collisions per Episode")
plt.xlabel("Episode")

plt.subplot(1, 3, 3)
window = 20
if len(all_r1) >= window:
    moving_avg_r1 = np.convolve(all_r1, np.ones(window)/window, mode='valid')
    moving_avg_r2 = np.convolve(all_r2, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg_r1, label="Agent 1 (moving avg)")
    plt.plot(moving_avg_r2, label="Agent 2 (moving avg)")
    plt.legend()
    plt.title(f"Moving Average Rewards (window={window})")
    plt.xlabel("Episode")

plt.tight_layout()
plt.show()

# --- Animation of One Episode ---
print("\nRunning one episode for animation...")
_, _, _, traj = run_episode(policy="greedy", record=True)

if traj:  # Only animate if we have trajectory data
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True)
    ax.set_aspect('equal')

    agent1_dot, = ax.plot([], [], 'bo', markersize=15, label="Agent 1")
    agent2_dot, = ax.plot([], [], 'ro', markersize=15, label="Agent 2")
    apple_dot, = ax.plot([], [], 'go', markersize=12, label="Apple")
    ax.legend(loc="upper right")
    ax.set_title("Gridworld Competition")

    def init():
        agent1_dot.set_data([], [])
        agent2_dot.set_data([], [])
        apple_dot.set_data([], [])
        return agent1_dot, agent2_dot, apple_dot

    def update(frame):
        if len(traj) == 0:
            return agent1_dot, agent2_dot, apple_dot
            
        # Each turn gets multiple animation frames for smooth movement
        frames_per_turn = 10
        turn_index = frame // frames_per_turn
        sub_frame = frame % frames_per_turn
        
        if turn_index >= len(traj):
            # Animation finished, show final positions
            if len(traj) > 0:
                final_move = traj[-1]
                a1_pos = final_move['new_pos'][0]
                a2_pos = final_move['new_pos'][1]
                apple = final_move['apple']
                agent1_dot.set_data([a1_pos[0]], [a1_pos[1]])
                agent2_dot.set_data([a2_pos[0]], [a2_pos[1]])
                apple_dot.set_data([apple[0]], [apple[1]])
                ax.set_title("Game Complete")
            return agent1_dot, agent2_dot, apple_dot
        
        move = traj[turn_index]
        old_a1, old_a2 = move['old_pos']
        new_a1, new_a2 = move['new_pos']
        apple = move['apple']
        turn = move['turn']
        
        # Calculate interpolated positions (smooth movement)
        progress = sub_frame / (frames_per_turn - 1) if frames_per_turn > 1 else 1
        
        # Interpolate agent 1 position
        a1_x = old_a1[0] + (new_a1[0] - old_a1[0]) * progress
        a1_y = old_a1[1] + (new_a1[1] - old_a1[1]) * progress
        
        # Interpolate agent 2 position  
        a2_x = old_a2[0] + (new_a2[0] - old_a2[0]) * progress
        a2_y = old_a2[1] + (new_a2[1] - old_a2[1]) * progress
        
        agent1_dot.set_data([a1_x], [a1_y])
        agent2_dot.set_data([a2_x], [a2_y])
        apple_dot.set_data([apple[0]], [apple[1]])
        
        if sub_frame == 0:  # Start of turn
            ax.set_title(f"Turn {turn} - Moving...")
        elif sub_frame == frames_per_turn - 1:  # End of turn
            ax.set_title(f"Turn {turn} - Complete")
        
        return agent1_dot, agent2_dot, apple_dot

    # Calculate total frames needed
    frames_per_turn = 10
    total_frames = len(traj) * frames_per_turn if traj else 1

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  init_func=init, blit=False, interval=100, repeat=True)
    plt.show()
else:
    print("No trajectory data to animate.")