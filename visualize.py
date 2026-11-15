import gymnasium as gym
import time
import os
from stable_baselines3 import PPO

# Dictionary of all your trained checkpoint files
MODELS = {
    "1": ("Baseline (150k steps)", "ppo_lunar_lander"),
    "2": ("Mid-Tuned (400k steps)", "ppo_lunar_lander_solved"),
    "3": ("Naive Shaped (300k steps)", "ppo_lunar_lander_precision"),
    "4": ("Master Solved (700k steps)", "ppo_lunar_lander_master"),
}

print("\n--- Select Model to Visualize ---")
for key, (label, path) in MODELS.items():
    exists = "✓ Ready" if os.path.exists(f"{path}.zip") else "✗ Missing"
    print(f"[{key}] {label} ({exists})")

choice = input("\nEnter choice (1-4) [Default is 4]: ").strip()
if choice not in MODELS:
    choice = "4"

label, model_path = MODELS[choice]
print(f"\nLoading: {label}...")

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load(model_path)

episodes = 5
for ep in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    total_reward = 0

    print(f"\n>>> Running Episode {ep} of 5...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(1 / 60)

    status = "Landed Safely!" if total_reward >= 100 else "Rough Touchdown / Crash"
    print(f"Episode {ep} Outcome: Score = {total_reward:.2f} ({status})")

env.close()