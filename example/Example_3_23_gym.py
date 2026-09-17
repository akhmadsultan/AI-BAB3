# Example 3.23 OpenAI Gym CartPole 
#https://gym.openai.com/docs/
#https://gym.openai.com/envs/CartPole-v0/
# Example 3.23 OpenAI Gym CartPole

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

for i_episode in range(20):
    observation, info = env.reset()

    for t in range(100):
        env.render()

        # Memilih aksi secara acak
        action = env.action_space.sample()

        # Menjalankan aksi
        observation, reward, terminated, truncated, info = env.step(action)

        print("Observation:", observation)

        if terminated or truncated:
            print(
                "Episode finished after {} timesteps".format(t + 1)
            )
            break

env.close()