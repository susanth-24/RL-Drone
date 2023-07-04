import re
import matplotlib.pyplot as plt
import csv
import pandas as pd

# file_path = 'model_v_1_3/output/Drone_RL_PPO_V1-3_verbose.txt'  # Replace with your file path

# with open(file_path, 'r') as file:
#     text_data = file.read()

# ep_rew_mean_values = re.findall(r"\| +ep_rew_mean +\| +(\d+) +\|", text_data)
# #print(ep_rew_mean_values)
# timestamp_values=re.findall(r"\| +total_timesteps +\| +(\d+) +\|", text_data)
# #print(timestamp_values)



# file_path = 'Drone_RL_PPO_V1_3.csv'  
# data = list(zip(timestamp_values, ep_rew_mean_values))

# with open(file_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Timestamp', 'Reward'])  
#     writer.writerows(data)  

# print("Data saved to", file_path)

file = pd.read_csv('model_v_1_3/output/Drone_RL_PPO_V1_3.csv')


time_steps = file['Timestamp']
rewards = file['Reward']


plt.scatter(time_steps, rewards)
plt.title('Rewards vs Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Rewards')

plt.show()
