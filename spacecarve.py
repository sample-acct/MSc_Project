import numpy as np 
import cv2 
import scipy.io   
#from scipy.misc import imread 
#import matplotlib.pyplot as plt
import imageio
import open3d
from mpl_toolkits.mplot3d import Axes3D
import time
from skimage.transform import downscale_local_mean
from scipy.stats import entropy 
from collections import defaultdict
from functools import reduce
#from cam_funcs import *

class SpaceCarve(object):
    def __init__(self, resolution, p_sense, p_change, K, rgb_lower=[180, 180, 0], rgb_upper=[255, 255, 20], frame_width=576, frame_height=694, voxel_center=(0,0,-.6), voxbox_size=.4, mode='mujoco', version=(1, 1), update=3, z_prob_occ=.55):
        #self.resolution = resolution
        self.p_sense = p_sense
        self.p_change = p_change 
        self.voxelCoords = self.makeVoxels(resolution, voxel_center, voxbox_size)
        self.voxelVals = np.divide(np.ones(np.shape(self.voxelCoords)[0]), 2)
        self.rgb_lower = rgb_lower
        self.rgb_upper = rgb_upper
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.K = K 
        self.num_carves = 0
        self.resolution = resolution 
        self.mode = mode 
        self.version = version 
        self.update = update
        self.z_prob_occ = z_prob_occ

    def reset(self):
        self.num_carves = 0
        self.voxelVals = np.divide(np.ones(np.shape(self.voxelCoords)[0]), 2)

    def getVoxelCoords(self):
        return self.voxelCoords

    def getVoxelVals(self):
        return self.voxelVals

    def carve(self, cam_ext, img, segment=True, fpath=None):

        self.num_carves += 1

        proj = self.project(cam_ext, np.transpose(self.voxelCoords))

        if segment:
            img_mask = self.applySegmentation(img)
        else:
            img_mask = img

        cv2.imshow('mask', img_mask)
        cv2.waitKey(10)

        #imageio.imwrite('imgs/mask_{}.jpg'.format(self.num_carves), img_mask)
        if fpath is not None:
            imageio.imwrite('{}_mask_{}.jpg'.format(fpath, self.num_carves), img_mask)

        # if self.mode == 'mujoco':
        #     hit_pixels = [(u, v) for u in range(0, self.frame_width) for v in range(0, self.frame_height) if img_mask[v, u] > 0]
        #     voxel_hits = []
        #print(np.shape(img_mask))

        hit_dict = {}
        unhit_dict = {}
        count = 0
    
        for i in range(0, np.shape(proj)[1]):
            p_prev = self.p_change * self.voxelVals[i] + (1 - self.p_change) * (1 - self.voxelVals[i])  
            v = int(proj[1][i]/proj[2][i])
            u = int(proj[0][i]/proj[2][i])
            
            
            if u >=0 and v >= 0 and u < self.frame_width and v < self.frame_height: 
                count += 1
                
                if img_mask[v, u] > 0 and self.mode == 'mujoco': # change back
                    occupied = 1 
                    if self.update == 2:
                        self.voxelVals[i] = (self.z_prob_occ * p_prev)/(self.z_prob_occ * p_prev + (1 - self.z_prob_occ) * (1 - p_prev))
                    
                    if self.update == 3:
                        if (u, v) not in hit_dict.keys():
                            hit_dict[(u, v)] = [i]
                        else:
                            hit_dict.get((u, v)).append(i)

                elif img_mask[v, u] == 0 and self.mode == 'dino':
                    occupied = 1
                    if self.update == 2:
                        self.voxelVals[i] = (self.z_prob_occ * p_prev)/(self.z_prob_occ * p_prev + (1 - self.z_prob_occ) * (1 - p_prev))
                    
                    if self.update == 3:
                        if (u, v) not in hit_dict.keys():
                            hit_dict[(u, v)] = [i]
                        else:
                            hit_dict.get((u, v)).append(i)
                
                else:
                    occupied = 0
                    if self.update != 3:
                        likelihood = (1 - self.p_sense) # p(z = 0 | x = 1)
                        nlikelihood = self.p_sense # p(z = 0 | x = 0)
                        # p(z = 0 | x = 1) * p(x = 1) / (p(z = 0 | x = 1) * p(x = 1) + p(z = 0 | x = 0) * p(x = 0))
                        self.voxelVals[i] = (likelihood * p_prev)/(likelihood * p_prev + nlikelihood * (1 - p_prev))

                    else:
                        if (u, v) not in unhit_dict.keys():
                            unhit_dict[(u, v)] = [i]
                        elif (u, v) in unhit_dict.keys():                         
                            unhit_dict.get((u, v)).append(i)                         


        if self.update == 3:
            voxelValsCopy = self.voxelVals
            for k in unhit_dict.keys():
                v_list = unhit_dict.get(k)
                for i in v_list:
                    if len(v_list) > 1:
                        p_others_empty = reduce(lambda x, y: x * y, map(lambda x: 1 - voxelValsCopy[x], list(set(v_list) - set([i]))))
                        nlikelihood = p_others_empty * self.p_sense + (1 - p_others_empty) * (1 - self.p_sense)
                    else:
                        nlikelihood = self.p_sense
                    likelihood = 1 - self.p_sense
                    p_prev = self.p_change * voxelValsCopy[i] + (1 - self.p_change) * (1 - voxelValsCopy[i])
                    self.voxelVals[i] = likelihood * p_prev/(likelihood * p_prev + nlikelihood * (1 - p_prev))
            for k in hit_dict.keys():
                v_list = hit_dict.get(k)
                for i in v_list:
                    if len(v_list) > 1:
                        p_others_empty = reduce(lambda x, y: x * y, map(lambda x: 1 - voxelValsCopy[x], list(set(v_list) - set([i]))))
                        nlikelihood = (1 - p_others_empty) * self.p_sense + p_others_empty * (1 - self.p_sense)
                    else:
                        nlikelihood = (1 - self.p_sense) 
                    likelihood = self.p_sense 
                    p_prev = self.p_change * voxelValsCopy[i] + (1 - self.p_change) * (1 - voxelValsCopy[i])
                    self.voxelVals[i] = likelihood * p_prev/(likelihood * p_prev + nlikelihood * (1 - p_prev))
            
        #projection = self.projectThresh(cam_ext, threshold=.5)
        #imageio.imwrite('imgs/camprojection_{}.jpg'.format(self.num_carves), projection)
                        
        return 

    def project(self, camE, worldCoords):
        transformedCoords = np.matmul(camE, worldCoords)

        if self.mode == 'mujoco':
            camX = -transformedCoords[0, :] - 0.23571429 +.01
            camY = transformedCoords[1, :] - 0.1744898 
            camZ = -transformedCoords[2, :] - .25 #+ 0.225 - .08 
            
            camCoords = np.vstack((camX, camY, camZ, transformedCoords[3, :]))
            res = np.matmul(self.K, camCoords)

        else:
            res = transformedCoords
        return res

    def projectThresh(self, camE, threshold=.5, fname=None):
        worldCoords = np.transpose(np.array([self.voxelCoords[i, :] for i in range(np.shape(self.voxelCoords)[0]) if self.voxelVals[i] >= threshold]))

        if np.shape(worldCoords)[0] == 0:
            thresh = max(self.voxelVals)
            print("threshold too high, use {}".format(thresh))
            worldCoords = np.transpose(np.array([self.voxelCoords[i, :] for i in range(0, np.shape(self.voxelCoords)[0]) if self.voxelVals[i] >= thresh]))

        proj = self.project(camE, worldCoords)
        
        proj_img = np.zeros((self.frame_height, self.frame_width))

        distToCenter = []

        for i in range(0, np.shape(proj)[1]):
            v = int(proj[1, i]/proj[2, i])
            u = int(proj[0, i]/proj[2, i])

            distToCenter.append(np.linalg.norm(np.array([v, u]) - np.array([self.frame_height/2, self.frame_width/2])))
            
            if u >=0 and v >= 0 and u < self.frame_width and v < self.frame_height: 
                proj_img[v, u] = 1
                #if np.isclose(u, self.frame_width/2, atol=20) and np.isclose(v, self.frame_height/2, atol=20):
                    #print("index: {}".format(i))

        if fname is not None:
            imageio.imwrite('{}_proj_{}.jpg'.format(fname, self.num_carves), proj_img)

        return proj_img


    def projectUncertainty(self, cam, fpath=None):
        proj = self.project(cam, np.transpose(self.voxelCoords))
       
        proj_img = np.ones((self.frame_height, self.frame_width))
        counts = np.zeros((self.frame_height, self.frame_width))
        max_vals = np.zeros((self.frame_height, self.frame_width))

        hit_dict = defaultdict(lambda: 1)
    
        for i in range(0, np.shape(proj)[1]):
            v = int(proj[1][i]/proj[2][i])
            u = int(proj[0][i]/proj[2][i])
            
            if u >=0 and v >= 0 and u < self.frame_width and v < self.frame_height: 
                if self.version[0] == 1:
                    proj_img[v, u] *= 1 - self.voxelVals[i] # fix to deal with unhit pixels
                    hit_dict[(u, v)] = hit_dict[(u, v)] * (1 - self.voxelVals[i])
                elif self.version[0] == 2:
                    counts[v, u] += 1
                # VERSION 2
                # running avg of sqr dist from .5
                    #proj_img[v, u] += ((self.voxelVals[i] - 0.5)**2 - proj_img[v, u])/counts[v, u] # UNCOMMENT TO RUN
                    if self.voxelVals[i] > max_vals[v, u]:
                        max_vals[v, u] = self.voxelVals[i]
                        hit_dict[(u, v)] = max_vals[v, u] 

        # if fpath is not None:
        #     imageio.imwrite('{}_{}.jpg'.format(fpath, self.num_carves), proj_img)
     
        # VERSION 1
        if self.version[0] == 1:
            res = np.ones((self.frame_height, self.frame_width)) - np.array(proj_img)
            for k in hit_dict.keys():
                hit_dict[k] = 1 - hit_dict[k]

        # VERSION 2
        else:
            #res = 0.5 - np.array(proj_img) # UNCOMMENT TO RUN 
            res = np.array(proj_img)

        return res, hit_dict 

    def view_certainty(self, cam_viewpoint, fpath=None):
        proj, value_dict = self.projectUncertainty(cam_viewpoint)
        value_array = np.array([value_dict.get(k) for k in value_dict.keys()])

        if fpath is not None:
            imageio.imwrite('{}_{}.jpg'.format(fpath, self.num_carves), proj)

        # VERSION 1
        if self.version[1] == 1:
            #dist = np.square(proj - np.ones((np.shape(proj)[0], np.shape(proj)[1])) * .5)
            dist = np.square(value_array - np.ones(np.shape(value_array)[0]) * .5)
            res = np.average(dist) # add percentage of visible voxels

        # VERSION 2
        elif self.version[1] == 2:
            # proj_distribution = proj/np.sum(proj)
            # res = entropy(proj_distribution)
            res = -entropy(value_array)

        return res 

    def voxel_uncertainty(self):
        dist = np.square(self.voxelVals - 0.5)
        return (0.5**2 - np.average(dist))/(0.5**2)

    def pcd_count(self, thresh=.5):
        vox_count = [v for v in self.voxelVals if v >= thresh]
        return len(vox_count)


    def applySegmentation(self, img):
        mask = cv2.inRange(img, np.array(self.rgb_lower), np.array(self.rgb_upper))
        return cv2.medianBlur(mask, 25)
    
    def makeVoxels(self, res, center, size):
        (x_c, y_c, z_c) = center
        x_s = np.linspace(x_c - size/2, x_c + size/2, res)
        y_s = np.linspace(y_c - size/2, y_c + size/2, res)
        z_s = np.linspace(z_c - size/2, z_c + size/2, res)
        voxels = np.array([[x, y, z, 1] for x in x_s for y in y_s for z in z_s])
        return voxels 

    def toVoxelRep(self, downscale=True, d_scale=10):
        xmin = min(self.voxelCoords[:, 0])
        ymin = min(self.voxelCoords[:, 1])
        zmin = min(self.voxelCoords[:, 2])
        xmax = max(self.voxelCoords[:, 0])
        vox_range = xmax - xmin
        
        scale = (self.resolution - 1)/vox_range 

        shift_vec = np.array([xmin, ymin, zmin, 0])
        voxels = self.voxelCoords - shift_vec
        voxels *= scale
        voxels = np.rint(voxels)
        voxelCube = np.zeros((self.resolution, self.resolution, self.resolution))

        for i in range(np.size(voxels[:, 0])):
            voxelCube[int(voxels[i, 0]), int(voxels[i, 1]), int(voxels[i, 2])] = self.voxelVals[i]

        if downscale == True:
            factor = int(self.resolution/d_scale)
            res = downscale_local_mean(voxelCube, (factor, factor, factor))
        else:
            res = voxelCube
        return res


    def visualize(self, save=False, fname="./dino.ply", threshold=0.5, show_frame=False):

        pcd = open3d.geometry.PointCloud()
        X = np.array([self.voxelCoords[i, :] for i in range(0, np.shape(self.voxelCoords)[0]) if self.voxelVals[i] >= threshold])
        if np.shape(X)[0] == 0:
            thresh = max(self.voxelVals)
            print("threshold too high, use {}".format(thresh))
            X = np.array([self.voxelCoords[i, :] for i in range(0, np.shape(self.voxelCoords)[0]) if self.voxelVals[i] >= thresh])

        pcd.points = open3d.Vector3dVector(X[:, 0:3])
        open3d.estimate_normals(pcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 100))
        open3d.orient_normals_to_align_with_direction(pcd)
        pcd.paint_uniform_color([1,0.706,0])
        mesh_frame = open3d.geometry.create_mesh_coordinate_frame(size=0.6, origin=[.6, 0, .75])
                    
        if show_frame:           
            open3d.visualization.draw_geometries([pcd, mesh_frame])
        else:
            open3d.visualization.draw_geometries([pcd])

        if save == True:
            open3d.write_point_cloud(fname, pcd)


    def visualize_plt(self, threshold=0.5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        resVox = np.array([self.voxelCoords[i, :] for i in range(0, np.shape(self.voxelCoords)[0]) if self.voxelVals[i] >= threshold])


        x = resVox[:, 0]
        y = resVox[:, 1]
        z = resVox[:, 2]

        ax.scatter(x, y, z, c=voxVals, marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #plt.gray()
        plt.show()
