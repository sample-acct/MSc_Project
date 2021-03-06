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
from SCAgent import SCAgent, BeethovenAgent




def main():

    beethoven = BeethovenAgent()
    nviews = 6
    ntrials = 10

    #beethoven.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, horizon=0, update=1, traj_viz=True, num_views=6)
    #beethoven.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, horizon=1, update=1, traj_viz=True, num_views=6)

    #bird.run_experiment('method', var_params=['even'], setting_names=['even'], plot_title="Bird Data Random vs Greedy vs Evenly Spaced Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=80, viz=True, update=1)
    #beethoven.run_experiment('horizon', var_params=[0, 1, 2], method='mpc',  setting_names=('Horizon 2'), plot_title="Beethoven MPC Horizon 0 vs Horizon 1 vs Horizon 2 Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=3)

    #beethoven.run_experiment('horizon', method='mpc', var_params=[0, 1, 2], setting_names=('Horizon 0', 'Horizon 1', 'Horizon 2'), plot_title="Beethoven MPC Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1)

    beethoven.run_experiment('method', var_params=('random', 'even','greedy'), setting_names=('random', 'even','greedy'), plot_title="Beethoven Average Uncertainty over Views", num_views=nviews, num_trials=ntrials, resolution=100, viz=False, update=1, horizon=1)
    #beethoven.run_experiment('update', method='greedy', var_params=(1, 2, 3), setting_names=('Version 1', 'Version 2', 'Version 3'), plot_title="Bird Data Average Uncertainty over Views for Different Update Rules", num_views=nviews, num_trials=ntrials, resolution=80, viz=False)


if __name__ == "__main__":
    start_time = time.process_time()
    main()
    print(time.process_time() - start_time, "seconds")

    
