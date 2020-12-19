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
from cam_funcs import *
from SCAgent import SCAgent, BirdAgent, DinoAgent, BeethovenAgent

def main():

    bird = BirdAgent()
    dino = DinoAgent()
    beethoven = BeethovenAgent()
    #bird.find_transmat(save=True)
    nviews = 6
    ntrials = 5
    #bird.run_experiment('horizon', var_params=[1], setting_names=['Horizon 1'], method='mpc',  plot_title="Bird MPC Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1, version=(1, 1))
    #bird.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, horizon=2, update=1, traj_viz=True, num_views=6)
    #beethoven.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, horizon=1, update=1, traj_viz=True, num_views=6)
    #bird.run_experiment('horizon', var_params=[0, 1, 2], setting_names=['Horizon 0', 'Horizon 1', 'Horizon 2'], method='mpc',  plot_title="Bird MPC Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1, version=(1, 1))
    #bird.run_experiment('resolution', var_params=(50, 80, 100, 150, 200), setting_names=('50', '80', '100', '150', '200'), plot_title="Bird Reconstruction Times at Different Resolutions", num_views=nviews, num_trials=ntrials, viz=False, update=1, xaxis='params', yaxis='times', ylabel='Time in Seconds')
    #bird.run_experiment('method', var_params=('mpc'), setting_names=('mpc'), plot_title="Bird Random vs Greedy vs MPC Spaced Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=80, viz=True, update=2, horizon=2)

##### RUN THESE
    bird.run_experiment('method', var_params=('random', 'even', 'greedy'), setting_names=('random', 'even', 'greedy'), plot_title="Bird Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1)
    # bird.run_experiment('update', method='greedy', var_params=(1, 2, 3), setting_names=('Version 1', 'Version 2', 'Version 3'), plot_title="Bird Average Uncertainty over Views for Different Update Rules", num_views=nviews, num_trials=ntrials, resolution=80, viz=False)
    # bird.run_experiment('update', method='greedy', var_params=(1, 2, 3), setting_names=('Version 1', 'Version 2', 'Version 3'), plot_title="Bird Average Uncertainty over Views for Different Update Rules", num_views=nviews, num_trials=ntrials, resolution=80, viz=False, yaxis="times")
####

if __name__ == "__main__":
    start_time = time.process_time()
    main()
    print(time.process_time() - start_time, "seconds")

    
