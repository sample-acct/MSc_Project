import numpy as np 
import cv2 
import scipy.io   
#from scipy.misc import imread 
import matplotlib.pyplot as plt
import imageio
import open3d
from mpl_toolkits.mplot3d import Axes3D
import time
from spacecarve import SpaceCarve as SC
from collections import defaultdict
from SCAgent import SCAgent, DinoAgent

def main():
    dino = DinoAgent()
    nviews = 6
    ntrials = 5
    dino.run_experiment('horizon', var_params=[0, 1, 2], setting_names=('Horizon 0', 'Horizon 1', 'Horizon 2'), method='mpc', plot_title="Dino Random vs Greedy vs Evenly Spaced Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1)
    
    #dino.visualize_ptcloud('greedy', 100, zprob=.9, xprob=.9, horizon=1, update=1, traj_viz=False, num_views=6)
    # dino.uncertainty_test(50, method='same', update=2, z_prob_occ=.7)
    # dino.uncertainty_test(50, method='same', update=3, z_prob_occ=.7)
    # dino.uncertainty_test(50, method='same', update=1)
    #dino.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, update=1, traj_viz=True, num_views=6, horizon=0)
    #dino.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, update=1, traj_viz=False, num_views=6, horizon=1)
    #dino.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, update=1, traj_viz=True, num_views=8, horizon=2)
    #dino.run_experiment('horizon', var_params=[1], setting_names=['Horizon 0'], method='mpc', plot_title="Dino MPC Average Uncertainty over Views for Different Horizons", num_views=6, num_trials=1, resolution=100, viz=True, update=1, version=(1, 1))
    #dino.run_experiment('horizon', var_params=[0, 1, 2], setting_names=['Horizon 0', 'Horizon 1', 'Horizon 2'], plot_title="Dino MPC Average Uncertainty over Views for Different Horizons", num_views=nviews, num_trials=ntrials, resolution=80, viz=False, update=1, version=(1, 1))

    #dino.run_experiment('method', var_params=('random', 'greedy', 'mpc'), setting_names=('random', 'greedy', 'mpc'), plot_title="Dino Random vs Greedy vs Evenly Spaced Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=50, viz=False, update=3, horizon=2)
    #dino.run_experiment('update', var_params=(1, 2, 3), setting_names=( 'Version 1', 'Version 2', 'Version 3'), plot_title="Dino Average Uncertainty with Different Update Rules", method='same', num_views=nviews, num_trials=ntrials, viz=False, ylabel="Overall Uncertainty", yaxis="uncertainty")
    #dino.run_experiment('horizon', var_params=[0, 1, 2], setting_names=['Horizon 0', 'Horizon 1', 'Horizon 2'], plot_title="Dino MPC Average Uncertainty over Views for Different Horizons", num_views=nviews, num_trials=ntrials, resolution=80, viz=False, update=1, version=(1, 1))
    #dino.run_experiment('method', var_params=['same'], setting_names=['same'], plot_title="Dino Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1, version=(1, 1))

    #dino.uncertainty_test(1, version=(1,1), update=2)
    # for res in [50, 80, 100, 150, 200]:
    #     dino.visualize_ptcloud(method='even', resolution=res, zprob=.9, xprob=.9, version=(1, 1), update=1, num_views=36, thresholds=[.5])

if __name__ == "__main__":
    start_time = time.process_time()
    main()
    print(time.process_time() - start_time, "seconds")
