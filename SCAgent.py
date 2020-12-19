import numpy as np 
import cv2 
import scipy.io   
import matplotlib.pyplot as plt
import imageio
import open3d
from mpl_toolkits.mplot3d import Axes3D
import time
from spacecarve import SpaceCarve as SC
from collections import defaultdict
from cam_funcs import *
import trajectory as T


class SCAgent(object):
    def __init__(self, name, zprob=.9, xprob=1, num_imgs=20, dist_thresh=1):
        self.name = name 
        self.num_imgs = num_imgs
        self.cams = self.init_cams()
        self.imgs = self.get_imgs()
        self.masks = self.get_masks()
        (self.im_height, self.im_width) = np.shape(self.masks[0])
        self.center, self.size = self.init_voxbox()
        self.cam_positions = self.get_cam_positions()
        self.transition_matrix = self.find_transmat()
        self.current_view_index = None
        self.dist_thresh = dist_thresh
        self.visited_views = []
        
        #sc = SC(resolution, zprob, xprob, None, rgb_lower=[180, 180, 0], rgb_upper=[255, 255, 20], frame_width=self.im_width, frame_height=self.im_height, voxel_center=center, voxbox_size=size, mode='dino', version=version, update=3)

        print("trans_mat: {}".format(self.transition_matrix))

    def init_cams(self):
        raise NotImplementedError("Please Implement this method")

    def get_imgs(self):
        raise NotImplementedError("Please Implement this method")

    def get_masks(self):
        raise NotImplementedError("Please Implement this method")

    def time_experiment(self, variable, var_params, setting_names="", plot_title="", num_views=10, num_trials=10, viz=False, update=3, method='greedy', resolution=80, version=(1,1)):
        results = []
        stderrs = []
        times = []
        for param in var_params:

            if variable == 'method':
                result, stderr, time = self.run_trials(method=param, num_views=num_views, num_trials=num_trials, viz=viz, version=version, update=update, resolution=resolution)
                
            elif variable == 'version':
                result, stderr, time = self.run_trials(version=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, update=update, resolution=resolution)

            elif variable == 'update':
                result, stderr, time = self.run_trials(update=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, version=version, resolution=resolution)

            elif variable == 'resolution':
                result, stderr, time = self.run_trials(resolution=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, version=version, update=update)


            results.append(result)
            stderrs.append(stderr)
            times.append(time)

        print(times)
        print(var_params)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plot_title = "Resolution"

        ax1.errorbar(var_params, times, yerr=stderrs, mfc='b', ls='-', alpha=.5)

        plt.legend(loc='upper right')
        plt.xlabel("Resolution")
        plt.ylabel("Time")
        plt.title(plot_title)

        plt.savefig('{}_imgs/{}_{}_{}_{}'.format(self.name, variable, var_params, num_views, num_trials))

    def run_experiment(self, variable, var_params, setting_names, plot_title='Results', ylabel="Model Uncertainty", num_views=10, num_trials=10, viz=False, version=(1,1), update=3, method='greedy', resolution=80, xaxis='views', yaxis='uncertainty', horizon=1):

        results = []
        stderrs = []
        times = []
        pcds = []
        pcd_stderrs = []
        for param in var_params:
            print("param: " + str(param))
            if variable == 'method':
                result, stderr, time, pcd, pcd_stderr = self.run_trials(method=param, num_views=num_views, num_trials=num_trials, viz=viz, version=version, update=update, resolution=resolution, horizon=horizon)
                
            elif variable == 'version':
                result, stderr, time, pcd, pcd_stderr = self.run_trials(version=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, update=update, resolution=resolution, horizon=horizon)

            elif variable == 'update':
                result, stderr, time, pcd, pcd_stderr = self.run_trials(update=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, version=version, resolution=resolution, horizon=horizon)

            elif variable == 'resolution':
                result, stderr, time, pcd, pcd_stderr = self.run_trials(resolution=param, num_views=num_views, num_trials=num_trials, viz=viz, method=method, version=version, update=update, horizon=horizon)
            
            elif variable == 'horizon':
                result, stderr, time, pcd, pcd_stderr = self.run_trials(horizon=param, method='mpc', resolution=resolution, num_views=num_views, num_trials=num_trials, viz=viz, version=version, update=update)

            np.savetxt("{}_imgs/results_{}_{}_{}.csv".format(self.name, variable, param, self.dist_thresh), result, delimiter=",")
            np.savetxt("{}_imgs/stderr_{}_{}_{}.csv".format(self.name, variable, param, self.dist_thresh), stderr, delimiter=",")
            results.append(result)
            stderrs.append(stderr)
            times.append(time)
            pcds.append(pcd)
            pcd_stderrs.append(pcd_stderr)

    

        view_list = range(1, num_views + 1)

        axis_dict = {
            "times":times, 
            "params":var_params,
            "uncertainty":results, 
            "views":view_list,
            "pcds":pcds
        }

        if yaxis == "pcd":
            stderrs= pcd_stderrs
        print("times: {}".format(times))
        colors = ['r', 'g', 'b', 'y', 'c', 'p']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(len(var_params)):
            ax1.errorbar(view_list, axis_dict.get(yaxis)[i], yerr=stderrs[i], mfc=colors[i], ls='-', label=setting_names[i], alpha=.5)

        plt.legend(loc='upper right', title="Horizon")
        plt.xlabel("Number of Views")
        plt.ylabel(ylabel)
        plt.title(plot_title)

        plt.savefig('{}_imgs/{}_{}_{}_{}_v{}_u{}'.format(self.name, variable, var_params, num_views, num_trials, version, update))

    def viewpoint_selection(self, sc, i, method='random', num_views=6, horizon=1, first_index=None, fpath=None):
        num_imgs = self.num_imgs 
        print("viewpoint horizon: " + str(horizon))
        if method == 'random':
            print("random view")
            index = np.random.randint(0, np.shape(self.cams)[0])  

            # if i == 1:
            #     index = np.random.randint(0, np.shape(self.cams)[0])   

            # else:
            #     cand_index_list = [k for k in range(0, self.num_imgs) if self.transition_matrix[self.current_view_index, k] < self.dist_thresh] # and k not in self.visited_views] 
            #     index = cand_index_list[np.random.randint(0, np.size(cand_index_list))]
        elif method == 'greedy':
            # cand_index_list = [k for k in range(0, self.num_imgs) if self.transition_matrix[self.current_view_index, k] < self.dist_thresh] # and k not in self.visited_views] 

            # view_certainty_list = np.array([sc.view_certainty(self.cams[k]) for k in cand_index_list])
            #print(view_certainty_list)

            view_certainty_list = np.array([sc.view_certainty(self.cams[i]) for i in range(self.num_imgs)])
            index = np.argmin(view_certainty_list)
        elif method == 'even':
            if i == 1:
                index = np.random.randint(0, num_imgs)
            else:
                index = (first_index + int((i - 1) * num_imgs/num_views)) % num_imgs # make this start at the same view as others?
        elif method == 'mpc':
            if horizon == 0:
                cand_index_list = [k for k in range(0, self.num_imgs) if self.transition_matrix[self.current_view_index, k] < self.dist_thresh] # and k not in self.visited_views] 
                index = cand_index_list[np.random.randint(0, np.size(cand_index_list))]
            elif horizon==1:
                cand_index_list = [k for k in range(0, self.num_imgs) if self.transition_matrix[self.current_view_index, k] < self.dist_thresh] # and k not in self.visited_views] 
                view_certainty_list = np.array([sc.view_certainty(self.cams[k]) for k in cand_index_list])
                index = cand_index_list[np.argmin(view_certainty_list)]
                print("Dist: {}".format(self.transition_matrix[self.current_view_index, index]))
                
            # elif horizon == 2:
            #     print("horizon 2")
            #     index = self.choose_mpc_action_v2(sc)

            else:
                _, index = self.choose_mpc_action(self.current_view_index, sc, horizon=horizon)
            #view_certainty_list = np.array([sc.view_certainty(cam) for cam in self.cams])
            #greedy_index = np.argmin(view_certainty_list)
            #print('mpc index: {}, greedy index: {}'.format(index, greedy_index))

        elif method == 'same':
            index = 0

        else:
            index = i - 1

        self.current_view_index = index
        self.visited_views.append(index)
    
        return index

    def choose_mpc_action_v2(self, sc):
            cand_index_list = [i for i in range(0, self.num_imgs) if self.transition_matrix[self.current_view_index, i] < self.dist_thresh] 
            index_array = np.array([[j for j in range(0, self.num_imgs) if self.transition_matrix[j, i] < self.dist_thresh] for i in cand_index_list])
            print("cand shape: {}, index array shape {}".format(len(cand_index_list), np.shape(index_array)))
            step2_certainties = np.array([min([sc.view_certainty(self.cams[i]) for i in index_array[j, :]]) for j in range(len(cand_index_list))])
            step1_certainties = np.array([sc.view_certainty(self.cams[i]) for i in cand_index_list])
            certainties = step1_certainties + step2_certainties
            return cand_index_list[np.argmin(certainties)]

    def choose_mpc_action(self, cur_view_index, SC, horizon=1):
        certainties = []
        cand_index_list = [i for i in range(0, self.num_imgs) if self.transition_matrix[cur_view_index, i] < self.dist_thresh] 
        print("cand list: {}".format(cand_index_list))
        #self.visualize_cams()
        for cam_index in cand_index_list:
            cam = self.cams[cam_index]
            certainty = SC.view_certainty(cam)
            certainties.append(certainty)
        if horizon == 1:
            print("optimal view: {}".format(cand_index_list[np.argmin(np.array(certainties))]))
            print("certainties: {}".format(certainties))
            return min(certainties), cand_index_list[np.argmin(np.array(certainties))]
            
        certainty_list = [certainties[i] + self.choose_mpc_action(cand_index_list[i], SC, horizon - 1)[0] for i in range(0, len(certainties))]

        return min(certainty_list), cand_index_list[np.argmin(np.array(certainty_list))]

    def find_transmat(self, save=False):
        print("cam positions: {}".format(self.cam_positions))
        mat = np.array([[np.linalg.norm(self.cam_positions[i] - self.cam_positions[j]) for i in range(self.num_imgs)] for j in range(self.num_imgs)])
        return mat

    def run_trials(self, method='random', num_views = 10, num_trials=10, viz=False, version=(1, 1), update=3, resolution=80, horizon=1):
        zprob = .8
        xprob = 1
        threshold = .5
        
        f_prefix = method
        first_index = None
        trialResults = []
        all_times = []
        all_pcd_cts = []
        
        sc = SC(resolution, zprob, xprob, None, rgb_lower=[180, 180, 0], rgb_upper=[255, 255, 20], frame_width=self.im_width, frame_height=self.im_height, voxel_center=self.center, voxbox_size=self.size, mode='dino', version=version, update=update)

        for t in range(num_trials):
            start_time = time.process_time()
            i = 0
            overall_uncertainty = []
            trialTimes = []
            pcd_cts = []
            sc.reset()
            self.visited_views = []

            while i < num_views:
                print("trial {}, carve {}, horizon {} ".format(t, i, horizon))

                i += 1
                if i == 1: 
                    index = self.viewpoint_selection(sc, i, 'random', num_views)
                    first_index = index 
                else:
                    print("horizon: " + str(horizon))
                    index = self.viewpoint_selection(sc, i, method, num_views=num_views, horizon=horizon, first_index=first_index)

                if viz == True: 
                    sc.view_certainty(self.cams[index], '{}_imgs/{}_certainty_proj_3'.format(self.name, f_prefix))
                #sc.view_certainty(cams[index])
                sc.carve(self.cams[index], self.masks[index], False, fpath='{}_imgs/{}_mask'.format(self.name, f_prefix)) #fpath='dino_imgs/{}_mask'.format(f_prefix)
                if viz == True:
                    imageio.imwrite('{}_imgs/{}_{}_{}.jpg'.format(self.name, f_prefix, self.name, sc.num_carves), self.imgs[index])
                overall_uncertainty.append(sc.voxel_uncertainty())
                run_time = time.process_time() - start_time
                trialTimes.append(run_time)
                pcd_cts.append(sc.pcd_count())

            trialResults.append(overall_uncertainty)
            all_times.append(trialTimes)
            all_pcd_cts.append(pcd_cts)
            

            if t == num_trials - 1 and viz == True:
                sc.visualize(True, 'pointclouds/{}_{}_viz_{}_v{}_u{}_r{}.ply'.format(method, self.name, threshold, version, update, resolution), threshold, show_frame=False)

        trialResults = np.array(trialResults)
        avg_results = np.mean(trialResults, 0)
        results_stderr = np.std(trialResults, 0)/(num_trials**.5)
        avg_time = np.mean(np.array(all_times), 0)
        pcd_results = np.array(all_pcd_cts)
        avg_pcd = np.mean(pcd_results, 0)
        pcd_stderr = np.std(pcd_results, 0)/(num_trials**.5)
        return avg_results, results_stderr, avg_time, avg_pcd, pcd_stderr

    
    def get_cam_positions(self):
        cam_positions = np.array([-1*np.matmul(np.linalg.inv(cam[:, 0:3]), cam[:, 3]) for cam in self.cams])
        
        return cam_positions

    def visualize_cams(self):
        cam_positions = np.array([-1*np.matmul(np.linalg.inv(cam[:, 0:3]), cam[:, 3]) for cam in self.cams])

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.Vector3dVector(cam_positions)
        pcd.paint_uniform_color([1,0.706,0])
        open3d.visualization.draw_geometries([pcd])


    def init_voxbox(self):
        cam_positions = np.array([-1*np.matmul(np.linalg.inv(cam[:, 0:3]), cam[:, 3]) for cam in self.cams])

        xmin, xmax = min(cam_positions[:, 0]), max(cam_positions[:, 0])
        ymin, ymax = min(cam_positions[:, 1]), max(cam_positions[:, 1])
        zmin, zmax = min(cam_positions[:, 2]), max(cam_positions[:, 2])

        
        size = int(max([xmax - xmin, ymax - ymin, zmax - zmin]))
        res = size 

        (x_c, y_c, z_c) = ((xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2)
        x_s = np.linspace(x_c - size/2, x_c + size/2, res)
        y_s = np.linspace(y_c - size/2, y_c + size/2, res)
        z_s = np.linspace(z_c - size/2, z_c + size/2, res)
        voxels = np.array([[x, y, z, 1] for x in x_s for y in y_s for z in z_s])
        for cam in self.cams:
            print("vox shape: {}, cam shape: {}".format(np.shape(voxels), np.shape(cam)))
            proj = np.matmul(cam, np.transpose(voxels))
            voxels = np.array([voxels[i, 0:4] for i in range(np.shape(voxels)[0]) if int(proj[1][i]/proj[2][i]) < self.im_height and int(proj[1][i]/proj[2][i]) >= 0 and
            int(proj[0][i]/proj[2][i]) < self.im_width and int(proj[0][i]/proj[2][i]) >= 0])

        vxmin, vxmax = min(voxels[:, 0]), max(voxels[:, 0])
        vymin, vymax = min(voxels[:, 1]), max(voxels[:, 1])
        vzmin, vzmax = min(voxels[:, 2]), max(voxels[:, 2])
        bounds = [(vxmin, vxmax), (vymin, vymax), (vzmin, vzmax)]
        size = int(max([vxmax - vxmin, vymax - vymin, vzmax - vzmin])) 
        center = ((vxmin + vxmax)/2, (vymin + vymax)/2, (vzmin + vzmax)/2)

        print("center: {}, size: {}".format(center, size))
        return center, size   

    def visualize_ptcloud(self, method, resolution, zprob=.8, xprob=1, version=(1, 1), update=1, horizon=1, num_views=1, thresholds=[.5], traj_viz=False):
        sc = SC(resolution, zprob, xprob, None, rgb_lower=[180, 180, 0], rgb_upper=[255, 255, 20], frame_width=self.im_width, frame_height=self.im_height, voxel_center=self.center, voxbox_size=self.size, mode='dino', version=version, update=update)
        i = 0

        trajectory_indices = []
        while i < num_views:
            i += 1
            if i == 1: 
                index = self.viewpoint_selection(sc, i, "random", num_views)
                print("first index should be " + str(index))
                first_index = index 
            else:
                index = self.viewpoint_selection(sc, i, method, num_views, first_index=first_index, horizon=horizon)
                print("index: " + str(index))
            sc.carve(self.cams[index], self.masks[index], False, fpath='{}_imgs/{}_mask'.format(self.name, method)) #fpath='dino_imgs/{}_mask'.format(f_prefix)
            imageio.imwrite('{}_imgs/{}_mmm{}_h{}_{}.jpg'.format(self.name, self.name, method, horizon, i), self.imgs[index])
            trajectory_indices.append(index)
        
        for threshold in thresholds:
            sc.visualize(True, 'pointclouds/{}_{}_viz_{}_v{}_u{}_r{}_h{}.ply'.format(method, self.name, threshold, version, update, resolution, horizon), threshold, show_frame=False)

        interpolated_pts = []
        if traj_viz == True:
            trajectory_cams = np.array([self.get_cam_positions()[i, :] for i in trajectory_indices])
            start_view = trajectory_cams[0, :]
            for i in range(1, num_views):
                
                target_view = trajectory_cams[i, :]
                if np.linalg.norm(np.array(start_view) - np.array(target_view)) == 0:
                    continue
                print("start: {}, target: {}".format(start_view, target_view))
                traj_int = T.interpolateTargetPoints(start_view, target_view, 5, self.center)
                print(traj_int)
                interpolated_pts += traj_int 
                start_view = target_view
            
            print("shapes: {}, {}".format(np.shape(trajectory_cams), np.shape(np.array(interpolated_pts))))
            points = np.vstack((trajectory_cams, np.array(interpolated_pts)))

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.Vector3dVector(points)
            pcd.paint_uniform_color([1,0.706,0])
            open3d.visualization.draw_geometries([pcd])

            open3d.write_point_cloud("{}_mpc_trajectory_viz_h{}.ply".format(self.name, horizon), pcd)

        X = np.array([sc.getVoxelCoords()[i, :] for i in range(0, np.shape(sc.getVoxelCoords())[0]) if sc.getVoxelVals()[i] >= threshold])
        return X, sc, self.get_cam_positions()[index] 

    def uncertainty_test(self, num_carves, method='greedy', resolution=100, zprob=.8, z_prob_occ=.55, xprob=.9, version=(1,1), update=3):
        carved_indices = []
        sc = SC(resolution, zprob, xprob, None, rgb_lower=[180, 180, 0], rgb_upper=[255, 255, 20], frame_width=self.im_width, frame_height=self.im_height, voxel_center=self.center, voxbox_size=self.size, mode='dino', version=version, update=update, z_prob_occ=z_prob_occ)

        for i in range(num_carves):
            index = self.viewpoint_selection(sc, i, method=method, num_views=num_carves, first_index=None)
            sc.carve(self.cams[index], self.masks[index], segment=False)
            carved_indices.append(index)
            print("index carved: {}".format(index))

        certainties = [sc.view_certainty(self.cams[i], fpath="figs/uncertainty_proj_{}_v{}_u{}".format(i, version, update)) for i in range(self.num_imgs)]
        min_dists_from_carved = [min([self.transition_matrix[i, j] for j in carved_indices]) for i in range(self.num_imgs)]
        print("min dists: {}".format(min_dists_from_carved))
        print("certainties: {}".format(certainties))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(range(self.num_imgs), certainties)

        #plt.legend(loc='upper right')
        plt.xlabel("Viewpoint")
        plt.ylabel("Viewpoint Certainty")
        plt.title("View Certainties over all Viewpoints after 1 Carve")
        plt.ylim(.13, .19)

        plt.savefig('figs/uncertainty2_{}_{}_{}_v{}_u{}'.format(self.name, num_carves, resolution, version, update))


class DinoAgent(SCAgent):
    def __init__(self):
        super(self.__class__, self).__init__("Dino", zprob=.9, xprob=.9, num_imgs=36, dist_thresh=.7)

    def init_cams(self):
        cams = scipy.io.loadmat('dino_Ps.mat')
        cams = np.array(cams['P'][0])
        return cams

    def get_imgs(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MATLAB\\dino_imgs\\"
        imgs = np.array([imageio.imread(img_file_dir + "viff.{0:03d}.ppm".format(n)) for n in range(0, self.num_imgs)])
        return imgs 

    def get_masks(self):
        mask = [cv2.inRange(img, np.array([50, 60, 100]), np.array([150, 150, 255])) for img in self.imgs]
        mask2 = [cv2.inRange(img, np.array([0,0,0]), np.array([35,35,35])) for img in self.imgs]
        #print(np.shape(mask))
        finalmask = [cv2.medianBlur(mask[i] + mask2[i], 9)[:, 0:694] for i in range(0, self.num_imgs)]
        
        return finalmask 

    def init_voxbox(self):
        return (0,0,-.62), .2


class BirdAgent(SCAgent):
    def __init__(self):
        super(self.__class__, self).__init__("Bird", zprob=.9, xprob=1, dist_thresh=50)

    def init_cams(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\bird_data\\"
        cams = np.array([np.loadtxt(img_file_dir + "calib/{0:04d}.txt".format(n), skiprows=1).reshape((3, 4)) for n in range(20)])
        return cams

    def get_imgs(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\bird_data\\"
        imgs = np.array([imageio.imread(img_file_dir + "images/{0:04d}.ppm".format(n)) for n in range(0, 20)])
        return imgs 

    def get_masks(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\bird_data\\"
        masks = np.array([imageio.imread(img_file_dir + "silhouettes/{0:04d}.pgm".format(n)) for n in range(0, 20)])
        return masks 

    def init_voxbox(self):
        return (0.5691154577392901, 0.08196922130526474, -2.734843896761255), 20

class BeethovenAgent(SCAgent):
    def __init__(self):
        super(self.__class__, self).__init__("Beethoven", zprob=.9, xprob=.9, dist_thresh=50)

    def init_cams(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\beethoven_data\\"
        cams = np.array([np.loadtxt(img_file_dir + "calib/{0:04d}.txt".format(n), skiprows=1).reshape((3, 4)) for n in range(20)])
        return cams

    def get_imgs(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\beethoven_data\\"
        imgs = np.array([imageio.imread(img_file_dir + "images/{0:04d}.ppm".format(n)) for n in range(0, 20)])
        return imgs 

    def get_masks(self):
        img_file_dir = "C:\\Users\\jelly\\Documents\\MSc Project\\MSc-proj\\beethoven_data\\"
        masks = np.array([imageio.imread(img_file_dir + "silhouettes/{0:04d}.pgm".format(n)) for n in range(0, 20)])
        return masks 

#dino = DinoAgent()
# bird = BirdAgent()

#dino.visualize_ptcloud(method='even', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=2, num_views=6, thresholds=[.9])
#dino.visualize_ptcloud(method='even', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=3, num_views=36, thresholds=[.1, .25, .5, .75, .9])
#dino.visualize_ptcloud(method='even', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=1, num_views=36, thresholds=[.1, .25, .5, .75, .9])

# bird.visualize_ptcloud(method='even', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=1, num_views=6, thresholds=[.5])
# bird.visualize_ptcloud(method='greedy', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=1, num_views=6, thresholds=[.5])
# bird.visualize_ptcloud(method='random', resolution=100, zprob=.9, xprob=.9, version=(1, 1), update=1, num_views=6, thresholds=[.5])
#dino.visualize_ptcloud('mpc', 100, zprob=.9, xprob=.9, update=1, traj_viz=False, num_views=6, horizon=1)
