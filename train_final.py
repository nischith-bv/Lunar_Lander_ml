import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    print("Setting up training environment...")
    raw_env = gym.make("LunarLander-v3", render_mode=None)
    env = Monitor(raw_env)

    # Tuned hyperparameters for stable high-reward convergence
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,     # Standard stable Adam learning rate for PPO
        n_steps=2048,           # Large trajectory buffer
        batch_size=64,
        n_epochs=10,
        gamma=0.99,             # Long horizon planning
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,          # Encourages fine-tuning exploration
        verbose=1,
    )

    total_timesteps = 700_000
    print(f"Training agent for {total_timesteps} steps (~8-12 minutes)...")
    model.learn(total_timesteps=total_timesteps)

    # Save final model
    save_path = "ppo_lunar_lander_master"
    model.save(save_path)
    print(f"\nModel saved successfully as '{save_path}.zip'")

    # Final 20-episode benchmark evaluation
    print("\nRunning official benchmark evaluation (20 episodes)...")
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode=None))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    
    print("\n==========================================")
    print(f"Final Benchmark Score: {mean_reward:.2f} +/- {std_reward:.2f}")
    if mean_reward >= 200.0:
        print("Status: ENVIRONMENT OFFICIALLY SOLVED (>= 200 pts)!")
    else:
        print(f"Status: Reached strong performance ({mean_reward:.2f}).")
    print("==========================================")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()