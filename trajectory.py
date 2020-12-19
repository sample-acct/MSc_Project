import numpy as np
from spacecarve import SpaceCarve
from geom_funcs import *
import queue
import imageio
from SCAgent import SCAgent, DinoAgent
import open3d
#from functools import reduce

def interpolateTarget(start_view, direction, steps, dist, center):
    trajectory = queue.Queue(maxsize=steps)
    step_length = dist/steps
    target = dirToView(direction, start_view, center, dist)
    print("next target: {}".format(target))
    current_view = start_view
    for t in range(1, steps + 1):
        next_view = dirToView(direction, start_view, center, step_length * t)
        #next_view = dirToView(direction, current_view, center, step_length)
        trajectory.put(next_view)
        print("next interpolated view: {}".format(next_view))
        #current_view = next_view
    return trajectory

def interpolateTargetPoints(start_view, target_view, steps, center):
    startVec_unscaled = np.array(start_view) - np.array(center)
    targetVec_unscaled = np.array(target_view) - np.array(center)
    startVec = startVec_unscaled/np.linalg.norm(startVec_unscaled)
    targetVec = targetVec_unscaled/np.linalg.norm(targetVec_unscaled)
    theta = np.arccos(np.dot(targetVec,startVec))
    axis = np.cross(startVec, targetVec)/np.linalg.norm(np.cross(startVec, targetVec))
    start_length = np.linalg.norm(startVec_unscaled)
    target_length = np.linalg.norm(targetVec_unscaled)
    length_ratio = target_length/start_length
    length_diff = target_length - start_length

    int_pts = []
    for i in range(1, steps):
        angle = i * theta/steps 
        quat = [np.sin(angle/2)* axis[0],np.sin(angle/2) * axis[1], np.sin(angle/2) * axis[2], np.cos(angle/2)]
        R = Rot.from_quat(quat).as_dcm()
        new_pt = np.matmul(R, startVec_unscaled) * ((start_length + i * length_diff/steps)/start_length)
        new_pt_shifted = new_pt + np.array(center)
        int_pts.append(new_pt_shifted)

    return int_pts


def getCandidateParams(cam_view, numDirs, centers, dists):
    dirs = [n * 2 * np.pi/numDirs for n in range(numDirs)] 
    params = [(d, c, dist) for d in dirs for c in centers for dist in dists]
    #cand_view_list = [dirToView(d, cam_view, c, dist) for (d, c, dist) in params]
    print("params: {}".format(params))
    return params 


def chooseGreedyAction(cam_view, dists, numDirs, SC, centers, horizon=1):
    certainties = []
    
    params = getCandidateParams(cam_view, numDirs, centers, dists)
    cand_view_list = [dirToView(d, cam_view, c, dist) for (d, c, dist) in params]
    cand_view_list = [c for c in cand_view_list] # if c[2] >= .8
    for v in cand_view_list:
        e_cam_mat = viewToCamMat(v)
        certainty = SC.view_certainty(e_cam_mat)
        certainties.append(certainty)

    print("candidate views: " + str(cand_view_list))
    print("certanties: " + str(certainties))
    min_index = np.argmin(np.array(certainties))
    print(cand_view_list[min_index])
    return params[min_index]


def getMinCertainty(cur_view, centers, env, SC, horizon=1, dists=[.2], numDirs=8):
    certainties = []
    params = getCandidateParams(cur_view, numDirs, centers, dists)
    cand_view_list = [dirToView(d, cur_view, c, dist) for (d, c, dist) in params]
    for v in cand_view_list:
        e_cam_mat = viewToCamMat(v)
        cam = np.matmul(SC.K, e_cam_mat)
        certainty = SC.view_certainty(cam)
        certainties.append(certainty)
    if horizon == 1:
        print("optimal view: {}".format(cand_view_list[np.argmin(np.array(certainties))]))
        return min(certainties), params[np.argmin(np.array(certainties))]
        
    certainty_list = [certainties[i] + getMinCertainty(cand_view_list[i], env, SC, horizon - 1)[0] for i in range(0, len(certainties))]

    return min(certainty_list), params[np.argmin(np.array(certainty_list))]


def generateTrajectoryPoints(start_view, center):

    trajectory = []
    current = start_view

    pos_list = [[1.5, 0, .8], [0, 0, .8],  [1.2, 0, 1.2],[.2, 0, 1], [.9, .4, .9], [1, 0, .8], [.6, 1, .8], [.6, -1, .8], [.1, 0, 1], [1, 0, 1], [.6, 0, 1.2], [.6, .5, 1], [.6, -.4, 1],  [.6, .3, 1], [.3, 0, 1.2]]

    for pos in pos_list:
        rot = orientationTowardsCenter(pos, center)
        next_view = np.concatenate((np.array(pos), np.array(rot)))
        #next_view = getCircleNextTarget(t, current, center, time_steps, radius=.4)
        trajectory.append(next_view)
        current = next_view

    trajectory = [np.concatenate((np.array(pos), orientationTowardsCenter(pos, center))) for pos in pos_list]

    trajectoryQ = queue.Queue(maxsize=len(pos_list))
    for t in trajectory:
        trajectoryQ.put(t)
    
    return trajectoryQ


def generateTrajectory(time_steps, start_view, center):
    radius = np.linalg.norm(np.array(start_view)[0:3] - np.array(center))
    trajectory = queue.Queue(maxsize=time_steps)
    current = start_view

    #pos_list = [[.7, .2, 1],[0.9, 0, 1], [.9, .3, 1], [.6, .3, 1], [.3, .3, 1], [.3, .1, .9], [.3, 0, 1], [.2, -.2, 1], [.6, -.2, 1], [.8, -.2, 1], [.8, 0, 1]]
    
    view_list = [getCircleNextTarget(t, current, center, time_steps, radius=.6) for t in range(1, time_steps)]
    
    params = getCandidateParams(start_view, 8, [center], [.8])
    #view_list = [dirToView(d, start_view, c, dist) for (d, c, dist) in params]
    for t in range(0, len(view_list)):
        next_view = view_list[t]
        trajectory.put(next_view)
        current = next_view

    return trajectory

def getCircleNextTarget(t, cur_view, obj_pos, steps, radius=.5):
    i = t * 2 * np.pi/steps
    cur_pos = cur_view[0:3]
    cur_rot = cur_view[3:6]
    R_mat = Rot.from_euler('XYZ', cur_rot).as_dcm()
    z = radius - radius* 15/20 + obj_pos[2]
    print("radius, z, obj_pos: {}, {}, {}".format(radius, z, obj_pos[2]))
    r = np.sqrt(radius * radius - (z - obj_pos[2]) * (z - obj_pos[2]))
    x, y = r * np.cos(t) + obj_pos[0], r * np.sin(t) + obj_pos[1]

    new_rot = orientationTowardsCenter(np.array([x, y, z]), obj_pos)
    
    return np.concatenate((np.array([x, y, z]), new_rot))


def getSpiralNextTarget(t, cur_view, obj_pos, steps, radius=1):
    cur_pos = cur_view[0:3]
    cur_rot = cur_view[3:6]
    z = radius - radius* t/steps + obj_pos[2]
    r = np.sqrt(radius * radius - (z - obj_pos[2]) * (z - obj_pos[2]))

    x, y = r * np.cos(t) + obj_pos[0], r * np.sin(t) + obj_pos[1]
    new_rot = orientationTowardsCenter(np.array([x, y, z]), obj_pos)

    return np.concatenate((np.array([x, y, z]), new_rot))


def visualize_trajectory(numDirs=8, dists=[.1, .3, .5], steps=10):

    dino = DinoAgent()
    center = dino.center 
    print("dino is located at: " + str(center))
    offset_vals = [-.1, .1]
    offsets = [np.array([i, j, k]) for i in offset_vals for j in offset_vals for k in offset_vals] + [np.array([0, 0, 0])]
    centers = [np.array(center) + offset for offset in offsets]


    ptcloud, sc, start_view = dino.visualize_ptcloud('greedy', 80)
    print("start view is: " + str(start_view))

    params = getCandidateParams(start_view, numDirs, centers, dists)
    print("params: " + str(params))
    cand_view_list = [dirToView(d, start_view, c, dist) for (d, c, dist) in params]

    cand_traj_pts = cand_view_list
    for (d, c, dist) in params:
        print("param: " + str((d, c, dist)))
        int_pts = interpolateTarget(start_view, d, steps, dist, c)
        pts_list = [int_pts.get() for i in range(int_pts.qsize())]
        print("points: " + str(pts_list))
        cand_traj_pts += pts_list

    (greedy_dir, greedy_c, greedy_dist) = chooseGreedyAction(start_view, dists, numDirs, sc, centers)
    print("greedy params: d {}, c {}, dist {}".format(greedy_dir, greedy_c, greedy_dist))
    greedy_traj = interpolateTarget(start_view, greedy_dir, steps, greedy_dist, greedy_c)
    greedy_pts_list = [greedy_traj.get() for i in range(greedy_traj.qsize())]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.Vector3dVector(ptcloud[:, 0:3])
    pcd.paint_uniform_color([1,0.706,0])

    total_ptcloud = np.vstack((ptcloud[:, 0:3], np.array(cand_traj_pts)[:, 0:3]))
    tpcd = open3d.geometry.PointCloud()
    tpcd.points = open3d.Vector3dVector(total_ptcloud)
    open3d.estimate_normals(tpcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 100))
    open3d.orient_normals_to_align_with_direction(tpcd)
    tpcd.paint_uniform_color([1,0.706,0])

    greedy_pcd = open3d.geometry.PointCloud()
    greedy_pcd.points = open3d.Vector3dVector(np.array(greedy_pts_list)[:, 0:3])
    greedy_pcd.paint_uniform_color([0, 1, 1])

    traj_pcd = open3d.geometry.PointCloud()
    
    traj_pcd.points = open3d.Vector3dVector(np.array(cand_traj_pts)[:, 0:3])
    traj_pcd.paint_uniform_color([1,0.706,0])
    open3d.visualization.draw_geometries([pcd, greedy_pcd])

    open3d.write_point_cloud("trajectory_viz.ply", tpcd)



visualize_trajectory(numDirs=8, dists=[.3, .5, .7], steps=5)


