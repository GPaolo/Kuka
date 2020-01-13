# Created by giuseppe
# Date: 22/11/19

from gym.envs.registration import register

register(
    id='KukaPush-v0',
    entry_point='gym_kuka.envs:KukaPush',
)