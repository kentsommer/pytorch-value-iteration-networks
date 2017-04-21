import sys
import argparse

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable

from dataset.dataset import *
from utility.utils import *
from model import *

from domains.gridworld import *
from generators.obstacle_gen import *


def main(config, n_domains=100, max_obs=30, 
            max_obs_size=None, n_traj=1, n_actions=8):
    # Correct vs total:
    correct, total = 0.0, 0.0
    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()
    vin = torch.load(config.weights)
    if use_GPU:
         vin = vin.cuda()

    for dom in range(n_domains):
        # Randomly select goal position
        goal = [np.random.randint(config.imsize),
                np.random.randint(config.imsize)]
        # Generate obstacle map
        obs = obstacles([config.imsize, config.imsize], goal, max_obs_size)
        # Add obstacles to map
        n_obs = obs.add_n_rand_obs(max_obs)
        # Add border to map
        border_res = obs.add_border() 
        # Ensure we have valid map
        if n_obs == 0 or not border_res:
            continue
        # Get final map
        im = obs.get_final()

        # Generate gridworld from obstacle map
        G = gridworld(im, goal[0], goal[1])
        # Get value prior
        value_prior = G.get_reward_prior()
        # Sample random trajectories to our goal
        states_xy, states_one_hot = sample_trajectory(G, n_traj)

        for i in range(n_traj):
            if len(states_xy[i]) > 1:

                # Get number of steps to goal
                L = len(states_xy[i]) * 2
                # Allocate space for predicted steps
                pred_traj = np.zeros((L, 2))
                # Set starting position
                pred_traj[0,:] = states_xy[i][0,:]

                for j in range(1, L):
                    # Transform current state data 
                    state_data = pred_traj[j-1, :]
                    state_data = state_data.astype(np.int)
                    # Transform domain to Networks expected input shape
                    im_data = G.image.astype(np.int)
                    im_data = 1 - im_data
                    im_data = im_data.reshape(1, 1, config.imsize, config.imsize)
                    # Transfrom value prior to Networks expected input shape
                    value_data = value_prior.astype(np.int)
                    value_data = value_data.reshape(1, 1, config.imsize, config.imsize)
                    # Get inputs as expected by network
                    X_in = torch.from_numpy(np.append(im_data, value_data, axis=1)).float()
                    S1_in = torch.from_numpy(state_data[0].reshape([1, 1])).float()
                    S2_in = torch.from_numpy(state_data[1].reshape([1, 1])).float()
                    # Send Tensors to GPU if available
                    if use_GPU:
                        X_in = X_in.cuda()
                        S1_in = S1_in.cuda()
                        S2_in = S2_in.cuda()
                    # Wrap to autograd.Variable
                    X_in, S1_in, S2_in = Variable(X_in), Variable(S1_in), Variable(S2_in)
                    # Forward pass in our neural net
                    _, predictions = vin(X_in, S1_in, S2_in, config)
                    _, indices = torch.max(predictions.cpu(),1)
                    a = indices.data.numpy()[0][0]
                    # Transform prediction to indices
                    s = G.map_ind_to_state(pred_traj[j-1, 0], pred_traj[j-1, 1])
                    ns = G.sample_next_state(s, a)
                    nr, nc = G.get_coords(ns)
                    pred_traj[j, 0] = nr
                    pred_traj[j, 1] = nc
                    if nr == goal[0] and nc == goal[1]:
                        # We hit goal so fill remaining steps
                        pred_traj[j+1:,0] = nr
                        pred_traj[j+1:,1] = nc
                        break
                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
                    correct += 1
                total += 1
                if config.plot == True:
                    visualize(G.image.T, states_xy[i], pred_traj)
        sys.stdout.write("\r" + str(int((float(dom)/n_domains) * 100.0)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Accuracy: {:.2f}%'.format(100*(correct/total)))


def visualize(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:,0], states_xy[:,1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:,0], pred_traj[:,1], '-X', c='r', label='Predicted Path')
    ax.plot(states_xy[0,0], states_xy[0,1], '-o', label='Start')
    ax.plot(states_xy[-1,0], states_xy[-1,1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small') # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='trained/vin_8x8.pth', 
                        help='Path to trained weights')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--imsize', 
                        type=int, 
                        default=8, 
                        help='Size of image')
    parser.add_argument('--k', 
                        type=int, 
                        default=10, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=2, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=10, 
                        help='Number of channels in q layer (~actions) in VI-module')
    config = parser.parse_args()
    # Compute Paths generated by network and plot
    main(config)