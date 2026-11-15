import gymnasium as gym

# 1. Initialize Lunar Lander
env = gym.make("LunarLander-v3")
state, info = env.reset()

print("Initial Observapython test_env.pytion (State Vector):")
print(state)
print("\nAction Space:", env.action_space)

# 2. Take one random action (0, 1, 2, or 3)
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)

print(f"\nAction taken: {action}")
print(f"Reward received: {reward:.2f}")
print("Next State Vector:")
print(next_state)

env.close()