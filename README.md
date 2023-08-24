#Reinforcement Learninng Based Drone

In this project I have used a 2d environment for drone, with following reward function,
```python
def reward_function(self,obs):
        velocity_x = obs[0]
        velocity_y =obs[1]
        omega = obs[2]
        alpha =obs[3]
        distance_x = obs[4]
        distance_y = obs[5]
        pos_x =obs[6]
        pos_y = obs[7]

        target_pos_x = 0.0
        target_pos_y = 0.0

        angle_weight = 0.2
        distance_weight = 0.5
        rotation_weight = 0.1

        #velocity_reward = velocity_weight * (velocity_x ** 2 + velocity_y ** 2)
        angle_reward = angle_weight*abs(alpha)
        distance_reward = distance_weight *np.sqrt( (1.0/(np.abs(obs[4])+0.1)) + (1.0/(np.abs(obs[5])+0.1)))
        rotation_reward = rotation_weight * abs(omega)

        reward = distance_reward - rotation_reward - angle_reward

        return float(reward)
```
