import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('model_v_1_2/Outputs/Drone_RL_PPO_V1.2_log_01.csv')


time_steps = file['Timestep']
rewards = file['Reward']


plt.scatter(time_steps, rewards)
plt.title('Rewards vs Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Rewards')

plt.show()