import pickle5 as pickle
import os
import pandas as pd
import numpy as np


PATH_city = 'Austin'
PATH_demand_model = '/home/DiTo/odysseus-escooter-dqn/odysseus/' \
                    'demand_modelling/demand_models/' + PATH_city


def get_neighbors_dicts(grid, grid_matrix):
    neighbors_dict = {}

    for i in grid_matrix.index:
        for j in grid_matrix.columns:
            zone = grid_matrix.iloc[i, j]

            if zone not in grid.index:
                continue

            i_low = i-1 if i-1 >= 0 else 0
            i_up = i+1 if i+1 < len(grid_matrix.index) \
                else len(grid_matrix.index) - 1

            j_low = j-1 if j-1 >= 0 else 0
            j_up = j+1 if j+1 < len(grid_matrix.columns) \
                else len(grid_matrix.columns) - 1

            neighbors_dict[int(zone)] = {}
            iii = 1

            for ii in range(i-1, i+2):
                for jj in range(j-1, j+2):
                    if ii == i and jj == j:
                        continue
                    
                    if ii < i_low or ii > i_up            \
                            or jj < j_low or jj > j_up    \
                            or grid_matrix.iloc[ii, jj] not in grid.index:
                        neighbors_dict[int(zone)] \
                            .update({iii: None})
                    else:
                        neighbors_dict[int(zone)] \
                            .update({iii: grid_matrix.iloc[ii, jj]})
                    
                    iii += 1

    return neighbors_dict


if __name__ == '__main__':
    grid = pickle.Unpickler(open(os.path.join(
        PATH_demand_model, "grid.pickle"), "rb")).load()

    grid_matrix = pickle.Unpickler(open(os.path.join(
        PATH_demand_model, "grid_matrix.pickle"), "rb")).load()

    # Count NaN values in neighbors DF
    with open(os.path.join(PATH_demand_model,
                           "neighbors_dict.pickle"),
              "rb") as f:
        N = pickle.load(f)

        print('NaN zones:',
              N.isna().sum().sum(), N.shape)

        h, w = grid_matrix.shape

        # N_tot = h * w * 8
        N_nan = 4 * 5 + 2*(h + w - 4)*3

        print('NaN zones #2:', N_nan)
    
    neighbors_dict = pd.DataFrame(get_neighbors_dicts(
        grid, grid_matrix)).fillna(np.nan)
    
    neighbors_dict.to_csv(PATH_city
        + '_neighbors_dict.csv')
    
    neighbors_dict.to_pickle(PATH_city
        + '_neighbors_dict.pickle')
