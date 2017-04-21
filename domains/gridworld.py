import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class gridworld:
    """A class for making gridworlds"""
    def __init__(self, image, targetx, targety):
        self.image = image
        self.n_row = image.shape[0]
        self.n_col = image.shape[1]
        self.obstacles = []
        self.freespace = []
        self.targetx = targetx
        self.targety = targety
        self.G = []
        self.W = []
        self.R = []
        self.P = []
        self.A = []
        self.n_states = 0
        self.n_actions = 0
        self.state_map_col = []
        self.state_map_row = []
        self.set_vals()


    def set_vals(self):
        # Setup function to initialize all necessary
        #  data
        row_obs, col_obs = np.where(self.image == 0)
        row_free, col_free = np.where(self.image != 0)
        self.obstacles = [row_obs, col_obs]
        self.freespace = [row_free, col_free]

        n_states = self.n_row * self.n_col
        n_actions = 8
        self.n_states = n_states
        self.n_actions = n_actions

        p_n = np.zeros((self.n_states, self.n_states))
        p_s = np.zeros((self.n_states, self.n_states))
        p_e = np.zeros((self.n_states, self.n_states))
        p_w = np.zeros((self.n_states, self.n_states))
        p_ne = np.zeros((self.n_states, self.n_states))
        p_nw = np.zeros((self.n_states, self.n_states))
        p_se = np.zeros((self.n_states, self.n_states))
        p_sw = np.zeros((self.n_states, self.n_states))

        R = -1 * np.ones((self.n_states, self.n_actions))
        R[:,4:self.n_actions] = R[:,4:self.n_actions] * np.sqrt(2)
        target = np.ravel_multi_index([self.targetx,self.targety], 
            (self.n_row,self.n_col), order='F')
        R[target,:] = 0
        
        for row in range(0, self.n_row):
            for col in range(0, self.n_col):

                curpos = np.ravel_multi_index([row,col], 
                    (self.n_row,self.n_col), order='F')

                rows, cols = self.neighbors(row, col)

                neighbor_inds = np.ravel_multi_index([rows,cols], 
                    (self.n_row,self.n_col), order='F')

                p_n[curpos, neighbor_inds[0]] = p_n[curpos, 
                                                    neighbor_inds[0]] + 1
                p_s[curpos, neighbor_inds[1]] = p_s[curpos, 
                                                    neighbor_inds[1]] + 1
                p_e[curpos, neighbor_inds[2]] = p_e[curpos, 
                                                    neighbor_inds[2]] + 1
                p_w[curpos, neighbor_inds[3]] = p_w[curpos, 
                                                    neighbor_inds[3]] + 1
                p_ne[curpos, neighbor_inds[4]] = p_ne[curpos, 
                                                    neighbor_inds[4]] + 1
                p_nw[curpos, neighbor_inds[5]] = p_nw[curpos, 
                                                    neighbor_inds[5]] + 1
                p_se[curpos, neighbor_inds[6]] = p_se[curpos, 
                                                    neighbor_inds[6]] + 1
                p_sw[curpos, neighbor_inds[7]] = p_sw[curpos, 
                                                    neighbor_inds[7]] + 1

        G = np.logical_or.reduce((p_n, p_s, p_e, p_w, 
            p_ne, p_nw, p_se, p_sw))

        W = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(
            np.maximum(np.maximum(p_n, p_s), p_e), p_w), np.sqrt(2) * p_ne), 
            np.sqrt(2) * p_nw), np.sqrt(2) * p_se), np.sqrt(2) * p_sw)

        non_obstacles = np.ravel_multi_index([self.freespace[0], 
            self.freespace[1]], (self.n_row,self.n_col), order='F')

        non_obstacles = np.sort(non_obstacles)
        p_n = p_n[non_obstacles,:]
        p_n = np.expand_dims(p_n[:, non_obstacles], axis=2)
        p_s = p_s[non_obstacles,:]
        p_s = np.expand_dims(p_s[:, non_obstacles], axis=2)
        p_e = p_e[non_obstacles,:]
        p_e = np.expand_dims(p_e[:, non_obstacles], axis=2)
        p_w = p_w[non_obstacles,:]
        p_w = np.expand_dims(p_w[:, non_obstacles], axis=2)
        p_ne = p_ne[non_obstacles,:]
        p_ne = np.expand_dims(p_ne[:, non_obstacles], axis=2)
        p_nw = p_nw[non_obstacles,:]
        p_nw = np.expand_dims(p_nw[:, non_obstacles], axis=2)
        p_se = p_se[non_obstacles,:]
        p_se = np.expand_dims(p_se[:, non_obstacles], axis=2)
        p_sw = p_sw[non_obstacles,:]
        p_sw = np.expand_dims(p_sw[:, non_obstacles], axis=2)
        G = G[non_obstacles, :]
        G = G[:, non_obstacles]
        W = W[non_obstacles, :]
        W = W[:, non_obstacles]
        R = R[non_obstacles, :]

        P = np.concatenate((p_n, p_s, p_e, p_w, 
            p_ne, p_nw, p_se, p_sw), axis=2)

        self.G = G
        self.W = W
        self.P = P
        self.R = R
        state_map_col, state_map_row = np.meshgrid(np.arange(0,self.n_col), 
            np.arange(0, self.n_row))
        self.state_map_col = state_map_col.flatten('F')[non_obstacles]
        self.state_map_row = state_map_row.flatten('F')[non_obstacles]


    def get_graph(self):
        # Returns graph
        G = self.G
        W = self.W[self.W != 0]
        return G, W


    def get_graph_inv(self):
        # Returns transpose of graph
        G = self.G.T
        W = self.W.T
        return G, W


    def val_2_image(self, val):
        # Zeros for obstacles, val for free space
        im = np.zeros((self.n_row, self.n_col))
        im[self.freespace[0], self.freespace[1]] = val
        return im


    def get_value_prior(self):
        # Returns value prior for gridworld
        s_map_col, s_map_row = np.meshgrid(np.arange(0,self.n_col), 
            np.arange(0, self.n_row))
        im = np.sqrt(np.square(s_map_col - self.targety) 
            + np.square(s_map_row - self.targetx))
        return im


    def get_reward_prior(self):
        # Returns reward prior for gridworld
        im = -1 * np.ones((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
        return im


    def t_get_reward_prior(self):
        # Returns reward prior as needed for
        #  dataset generation
        im = np.zeros((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
        return im


    def get_state_image(self, row, col):
        # Zeros everywhere except [row,col]
        im = np.zeros((self.n_row, self.n_col))
        im[row, col] = 1
        return im


    def map_ind_to_state(self, row, col):
        # Takes [row, col] and maps to a state
        rw = np.where(self.state_map_row == row)
        cl = np.where(self.state_map_col == col)
        return np.intersect1d(rw, cl)[0]


    def get_coords(self, states):
        # Given a state or states, returns
        #  [row,col] pairs for the state(s)
        non_obstacles = np.ravel_multi_index(
            [self.freespace[0], self.freespace[1]], 
            (self.n_row,self.n_col), order='F')
        non_obstacles = np.sort(non_obstacles)
        states = states.astype(int)
        r, c = np.unravel_index(non_obstacles[states], 
            (self.n_col, self.n_row), order='F')
        return r, c


    def rand_choose(self, in_vec):
        # Samples 
        if len(in_vec.shape) > 1:
            if in_vec.shape[1] == 1:
                in_vec = in_vec.T
        temp = np.hstack((np.zeros((1)), np.cumsum(in_vec))).astype('int')
        q = np.random.rand()
        x = np.where(q > temp[0:-1])
        y = np.where(q < temp[1:])
        return np.intersect1d(x, y)[0]


    def next_state_prob(self, s, a):
        # Gets next state probability for
        #  a given action (a)
        if hasattr(a, "__iter__"):
            p = np.squeeze(self.P[s, :, a])
        else:
            p = np.squeeze(self.P[s, :, a]).T
        return p


    def sample_next_state(self, s, a):
        # Gets the next state given the
        #  current state (s) and an 
        #  action (a)
        vec = self.next_state_prob(s, a)
        result = self.rand_choose(vec)
        return result


    def get_size(self):
        # Returns domain size
        return self.n_row, self.n_col


    def north(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row-1, 0])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def northeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def northwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def south(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def southeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def southwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def east(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def west(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def neighbors(self, row, col):
        # Get valid neighbors in all valid directions
        rows, cols = self.north(row, col)
        new_row, new_col = self.south(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.east(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.west(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        return rows, cols


def trace_path(pred, source, target):
    # traces back shortest path from
    #  source to target given pred
    #  (a predicessor list)
    max_len = 1000
    path = np.zeros((max_len, 1))
    i = max_len - 1
    path[i] = target
    while path[i] != source and i > 0:
        try:
            path[i-1] = pred[int(path[i])]
            i -= 1
        except Exception as e:
            return []
    if i >= 0:
        path = path[i:]
    else:
        path = None
    return path


def sample_trajectory(M, n_states):
    # Samples trajectories from random nodes
    #  in our domain (M)
    G, W = M.get_graph_inv()
    N = G.shape[0]
    if N >= n_states:
        rand_ind = np.random.permutation(N)
    else:
        rand_ind = np.tile(np.random.permutation(N), (1,10))
    init_states = rand_ind[0:n_states].flatten()
    goal_s = M.map_ind_to_state(M.targetx, M.targety)
    states = []
    states_xy = []
    states_one_hot = []
    # Get optimal path from graph
    g_dense = W
    g_masked = np.ma.masked_values(g_dense,0)
    g_sparse = csr_matrix(g_dense)
    d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)
    for i in range(n_states):
        path = trace_path(pred, goal_s, init_states[i])
        path = np.flip(path, 0)
        states.append(path)
    for state in states:
        L = len(state)
        r, c = M.get_coords(state)
        row_m = np.zeros((L, M.n_row))
        col_m = np.zeros((L, M.n_col))
        for i in range(L):
            row_m[i, r[i]] = 1
            col_m[i, c[i]] = 1
        states_one_hot.append(np.hstack((row_m, col_m)))
        states_xy.append(np.hstack((r, c)))
    return states_xy, states_one_hot