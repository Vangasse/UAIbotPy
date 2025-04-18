from utils import *
import numpy as np
import os
import torch

def _fkm(self, q=None, axis='eef', htm=None, mode='auto'):

    if q is None:
        q = self.q
    if htm is None:
        htm = np.matrix(self.htm)

    n = len(self._links)

    # Error handling
    if mode not in ['python','c++','auto', 'gpu']:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")
    
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not (axis == "eef" or axis == "dh" or axis == "com"):
        raise Exception("The parameter 'axis' should be one of the following strings:\n" \
                        "'eef': End-effector \n" \
                        "'dh': All " + str(n) + " axis of Denavit-Hartenberg\n" \
                                                "'com': All " + str(
            n) + " axis centered at the center of mass of the objects.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")
    
    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == 'gpu':
        return _fkm_gpu(self, q, axis, htm)
    elif mode == 'python' or axis == 'com' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        return _fkm_python(self, q, axis, htm)
    else:
        fk_res = self.cpp_robot.fk(q, htm, False)
        if axis=='eef':
            return np.matrix(fk_res.htm_ee)
        else:
            return [np.matrix(m) for m in fk_res.htm_dh]

def _fkm_python(self, q, axis, htm):

    n = len(self._links)
    htm_dh = [np.matrix(np.zeros((4,4))) for i in range(n)]

    for i in range(n):
        if i == 0:
            htm_dh[i][:, :] = htm * self._htm_base_0
        else:
            htm_dh[i][:, :] = htm_dh[i - 1][:, :]

        if self.links[i].joint_type == 0:
            htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.rotz(q[i])
        else:
            htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.rotz(self._links[i].theta)

        if self.links[i].joint_type == 1:
            htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.trn([0, 0, q[i]])
        else:
            htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.trn([0, 0, self._links[i].d])

        htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.rotx(self._links[i].alpha)
        htm_dh[i][:, :] = htm_dh[i][:, :] * Utils.trn([self._links[i].a, 0, 0])

    if axis == 'com':
        for i in range(n):
            htm_dh[i][0:3, 3] = htm_dh[i][0:3, 3] + htm_dh[i][0:3, 0:3] * self._links[i].com_coordinates

    if axis == 'eef':
        htm_dh = htm_dh[-1][:, :] * self.htm_n_eef

    return htm_dh

def _fkm_gpu(self, q, axis, htm):

    n = len(self._links)

    q_tensor = torch.tensor(q, dtype=torch.float32)
    htm_tensor = torch.tensor(htm, dtype=torch.float32)
    htm_dh = torch.eye(4, dtype=torch.float32).repeat(n, 1, 1)

    alpha_tensor = torch.tensor([link.alpha for link in self._links], dtype=torch.float32)
    theta_tensor = torch.tensor([link.theta for link in self._links], dtype=torch.float32)
    
    for i in range(n):
        if i == 0:
            htm_dh[i] = torch.matmul(htm_tensor, torch.tensor(self._htm_base_0, dtype=torch.float32))
        else:
            htm_dh[i] = htm_dh[i - 1]

        if self.links[i].joint_type == 0:
            htm_dh[i] = torch.matmul(htm_dh[i], Utils.rotz_torch(q_tensor[i]))
        else:
            htm_dh[i] = torch.matmul(htm_dh[i], Utils.rotz_torch(theta_tensor[i]))

        if self.links[i].joint_type == 1:
            htm_dh[i] = torch.matmul(htm_dh[i], Utils.trn_torch([0, 0, q_tensor[i]]))
        else:
            htm_dh[i] = torch.matmul(htm_dh[i], Utils.trn_torch([0, 0, self._links[i].d]))

        htm_dh[i] = torch.matmul(htm_dh[i], Utils.rotx_torch(alpha_tensor[i]))
        htm_dh[i] = torch.matmul(htm_dh[i], Utils.trn_torch([self._links[i].a, 0, 0]))

    if axis == 'com':
        for i in range(n):
            com_coordinates = torch.tensor(self._links[i].com_coordinates, dtype=torch.float32)
            htm_dh[i][:3, 3] += torch.matmul(htm_dh[i][:3, :3], com_coordinates)

    if axis == 'eef':
        htm_dh = torch.matmul(htm_dh[-1], torch.tensor(self.htm_n_eef, dtype=torch.float32))

    return htm_dh