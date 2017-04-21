import sys

import numpy as np
from dataset import *

sys.path.append('.')
from domains.gridworld import *
from generators.obstacle_gen import *
sys.path.remove('.')

def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of 
    #  actions corresponding to the trajectory. 
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.],[1.,0.],[0.,1.],[0.,-1.],[-1.,1.],
                              [-1.,-1.],[1.,1.],[1.,-1.]])
    action_vecs[4:] = 1/np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(1/np.sqrt(np.sum(np.square(
                                   state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff -1)<0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


def make_data(dom_size, n_domains, max_obs, 
                max_obs_size, n_traj, state_batch_size):

    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []

    dom = 0.0
    while dom <= n_domains:
        goal = [np.random.randint(dom_size[0]),
                np.random.randint(dom_size[1])]
        # Generate obstacle map
        obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
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
        value_prior = G.t_get_reward_prior()
        # Sample random trajectories to our goal
        states_xy, states_one_hot = sample_trajectory(G, n_traj)
        for i in range(n_traj):
            if len(states_xy[i]) > 1:
                # Get optimal actions for each state
                actions = extract_action(states_xy[i])
                ns = states_xy[i].shape[0] - 1
                # Invert domain image => 0 = free, 1 = obstacle
                image = 1 - im
                # Resize domain and goal images and concate
                image_data = np.resize(image, (1,1,dom_size[0],dom_size[1]))
                value_data = np.resize(value_prior, (1,1,dom_size[0],
                                                         dom_size[1]))
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                X_current = np.tile(iv_mixed, (ns, 1, 1, 1))
                # Resize states
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)
                # Resize labels
                Labels_current = np.expand_dims(actions, axis=1)
                # Append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                Labels_l.append(Labels_current)
        dom += 1
        sys.stdout.write("\r" + str(int((dom/n_domains) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Concat all outputs
    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    Labels_f = np.concatenate(Labels_l)
    return X_f, S1_f, S2_f, Labels_f


def main(dom_size=[8,8], n_domains=15000, max_obs=30, max_obs_size=None, 
            n_traj=7, state_batch_size=1):
    # Get path to save dataset
    save_path = "dataset/gridworld_{0}x{1}".format(dom_size[0], dom_size[1])
    # Get training data
    print("Now making training data...")    
    X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr = make_data(
        dom_size, n_domains, max_obs, max_obs_size, n_traj, state_batch_size)
    # Get testing data
    print("\nNow making  testing data...")
    X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts = make_data(
        dom_size, n_domains/6, max_obs, max_obs_size, n_traj, state_batch_size)
    # Save dataset
    np.savez_compressed(save_path, X_out_tr, S1_out_tr, S2_out_tr, 
        Labels_out_tr, X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts)
    

if __name__ == '__main__':
    main()