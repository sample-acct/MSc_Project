from scipy.spatial.transform import Rotation as Rot
import numpy as np

def vecsToRotMat(startVec, targetVec):
    """returns rotation mat of angle change between two vectors"""
    startVec = startVec/np.linalg.norm(startVec)
    targetVec = targetVec/np.linalg.norm(targetVec)
    theta = np.arccos(np.dot(targetVec,startVec))

    print("rot difference in angle: {}, between {} and {}".format(theta, targetVec, startVec))
    
    if theta != 0 and not np.isnan(theta):
        axis = np.cross(startVec, targetVec)/np.linalg.norm(np.cross(startVec, targetVec)) ### ??
        quat = [np.sin(theta/2)* axis[0],np.sin(theta/2) * axis[1], np.sin(theta/2) * axis[2], np.cos(theta/2)]
        r = Rot.from_quat(quat)
        res = r.as_dcm()
    else:
        res = np.identity(3)
    
    return res

def vecsToRotControl(current_orientation, target_orientation):
    current_Rmat = Rot.from_euler('XYZ', current_orientation).as_dcm()
    target_Rmat = Rot.from_euler('XYZ', target_orientation).as_dcm()
    dR = np.matmul(target_Rmat, np.linalg.inv(current_Rmat))
    res = Rot.from_dcm(dR).as_euler('XYZ')
    res[1] = -res[1]
    res[2] = -res[2]
    return res 

def vecsToAngleChanges(startVec, targetVec):
    """returns euler angles of angle change between two vectors"""
    startVec = startVec/np.linalg.norm(startVec)
    targetVec = targetVec/np.linalg.norm(targetVec)
    theta = np.arccos(np.dot(startVec, targetVec))
    axis = np.cross(startVec, targetVec)
    print("theta: {}".format(theta))
    print("axis: {}".format(axis))
    if theta != 0:
        quat = [np.cos(theta/2), np.sin(theta/2)* axis[0],np.sin(theta/2) * axis[1], np.sin(theta/2) * axis[2]]
        r = Rot.from_quat(quat)
        res = r.as_euler('XYZ')
    else:
        res = [0, 0, 0]
    
    return res

def arcDistToTarget(self, start_pos, target_pos, center):
    """returns arc distance needed to reach target on sphere"""
    start_vec = start_pos - center 
    tar_vec = target_pos - center 
    radius = np.linalg.norm(start_vec)
    theta = np.arccos(np.dot(start_vec, tar_vec))
    return theta * radius

def viewToCamMat(view):
    """
    input: view as absolute position and rotation
    output: extrinsic camera matrix of resulting view
    """
    position = view[0:3]
    rotation = view[3:6]
    r = Rot.from_euler('XYZ', rotation)
    R = np.transpose(r.as_dcm())
    T = -1  * np.matmul(R, position)
    return np.vstack((np.column_stack((R, T)), np.array([0, 0, 0, 1]))) 


def dirToView(direction, cam_q, center, arc_length):
    ''' Used in: 
    interpolateTarget
    chooseGreedyAction
    '''
    # vector from cam to center in world coords
    cam_norm = np.array(center) - cam_q[0:3] 
    # orientation matrix toward center
    R_prime = Rot.from_euler('XYZ', orientationTowardsCenter(cam_q[0:3], center)).as_dcm()
    radius = np.linalg.norm(cam_norm)
    angle = arc_length/radius
    inner_angle = (np.pi - angle)/2
    chord_length =  radius * np.sin(angle) / np.sin(inner_angle)
    theta_c = np.pi/2 - inner_angle
    r_c = chord_length * np.cos(theta_c)
    
    # new view vector in camera coordinates
    cvec = np.array([r_c * np.cos(direction), r_c * np.sin(direction), -chord_length * np.sin(theta_c)])
    wvec = np.matmul(R_prime, cvec) 
    pos = cam_q[0:3] + wvec 

    pos_norm = np.array(center) - pos 

    new_rot = orientationTowardsCenter(pos, center)
    return np.concatenate((pos, new_rot))

def orientationTowardsCenter(pos, center):
    orientVec = (np.array(center) - np.array(pos))
    orientVec = orientVec/np.linalg.norm(orientVec)
    R = vecsToRotMat(np.array([0, 0, -1]), orientVec)
    res = Rot.from_dcm(R).as_euler('XYZ')
    return res
    

def eulerChangeToWorldRot(current_Rmat, d_rot):
    initial_rot = Rot.from_euler(np.array([0, np.pi, 0]), 'XYZ')
    init_mat = Rot.as_dcm(initial_rot)
    cameraCoordNewRot = Rot.from_euler('XYZ', d_rot).as_dcm()
    # old orientation of cam frame in world coords
    curCamFrameR = np.matmul(current_Rmat, init_mat)
    # new orientation of cam frame in world coords
    worldNewRotCamFrame = np.matmul(cameraCoordNewRot, curCamFrameR)
    # rotate back to pre cam frame orientation
    worldNewRotCamFrame = np.matmul(np.transpose(init_mat), worldNewRotCamFrame)
    return Rot.from_dcm(worldNewRotCamFrame).as_euler('XYZ')



    
