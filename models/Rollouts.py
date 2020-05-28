import numpy
from models.enviroment import  fastMRIEnviroment


max_trails=1000

ROLLOUT_DIR='./ROLLOUT'
training_datadir = '/media/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/'


if __name__ == '__main__':

    env=fastMRIEnviroment(datadir=training_datadir)
    actions = []
    rewards = []
    state = []
    done = []

    for rollout in range(max_trails):



        training_labels, training_images, action, reward, done = env.step()

        actions.append(action)
        rewards.append(reward)
        if done:
            print(actions)
            print(rewards)
            actions = []
            rewards = []
            state = []
            done = []