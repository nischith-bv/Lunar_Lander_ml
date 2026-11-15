import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    print("Initializing environment with Monitor...")
    # Wrap with Monitor to track episodic returns cleanly
    raw_env = gym.make("LunarLander-v3", render_mode=None)
    env = Monitor(raw_env)

    print("Building tuned PPO Agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,     # Slightly lower learning rate for stable fine-tuning
        n_steps=2048,           # Larger rollout buffer collects smoother trajectory data
        batch_size=64,
        n_epochs=10,            # More update passes per rollout
        gamma=0.99,             # Emphasizes long-term safe touchdown
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,         # Mild entropy bonus to preserve control stability
        verbose=1,
    )

    total_timesteps = 400_000
    print(f"Beginning training for {total_timesteps} steps (~4-7 mins)...")
    model.learn(total_timesteps=total_timesteps)

    # Save to a dedicated solved checkpoint
    save_path = "ppo_lunar_lander_solved"
    model.save(save_path)
    print(f"\nModel saved successfully as '{save_path}.zip'")

    # Evaluate over 20 random initial seeds
    print("\nRunning post-training benchmark evaluation (20 episodes)...")
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode=None))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"\n==========================================")
    print(f"Final Benchmark Score: {mean_reward:.2f} +/- {std_reward:.2f}")
    if mean_reward >= 200.0:
        print("Status: ENVIRONMENT SOLVED (>= 200 pts)!")
    else:
        print("Status: Strong convergence, close to solved threshold.")
    print(f"==========================================")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()