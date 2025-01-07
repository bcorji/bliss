import gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("CartPole-v1")

# Train the model using PPO algorithm
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
env.close()