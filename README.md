# Reinforcement Learning Based Drone

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
## A2C PPO
This project revolved around actor critic two player method, here we have two neural networks that correct themselves as we train mainly actor and critic. The actor predicts action and critic evaluates how good the predicted action is based on previous rewards and states.
Proximal policy optimisation proposes TRPO(Trust Region) to prevent the new policy to not deviate much from old policy.
These are the results from training the agent with the above mentioned reward function
![Graph](/model_v_1_2/Outputs/rewardv_1_2.png)