import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Custom Reward Shaping Wrapper
class PrecisionLandingWrapper(RewardWrapper):
    def reward(self, reward):
        # The lander's 8D state vector
        # s[0] = x position (-1.0 to 1.0; 0 is center)
        # s[1] = y position (altitude)
        # s[2] = horizontal velocity (vx)
        # s[4] = tilt angle (theta)
        s = self.env.unwrapped.lander.position
        vel = self.env.unwrapped.lander.linearVelocity
        angle = self.env.unwrapped.lander.angle

        # 1. Distance penalty: strongly discourage drifting away from x = 0
        x_dist = abs(s.x)
        reward -= 0.5 * x_dist

        # 2. Angle penalty: penalize tilting away from upright
        reward -= 0.3 * abs(angle)

        # 3. Horizontal sway penalty when low to the ground
        if s.y < 0.5:
            reward -= 0.4 * abs(vel.x)

        return reward

def main():
    print("Initializing environment with Precision Reward Shaping...")
    base_env = gym.make("LunarLander-v3", render_mode=None)
    shaped_env = PrecisionLandingWrapper(base_env)
    env = Monitor(shaped_env)

    print("Training PPO Agent on Shaped Dynamics...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
    )

    # Train for 300,000 steps
    total_timesteps = 300_000
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # Save shaped weights
    save_path = "ppo_lunar_lander_precision"
    model.save(save_path)
    print(f"\nModel saved successfully as '{save_path}.zip'")

    # Evaluate on raw, standard environment rules (true benchmark)
    print("\nEvaluating precision model against standard rules (20 episodes)...")
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode=None))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"\n==========================================")
    print(f"Final Standard Benchmark Score: {mean_reward:.2f} +/- {std_reward:.2f}")
    if mean_reward >= 200.0:
        print("Status: ENVIRONMENT SOLVED (>= 200 pts)!")
    else:
        print(f"Status: Benchmark finished with score {mean_reward:.2f}")
    print(f"==========================================")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()