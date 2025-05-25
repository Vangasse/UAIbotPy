from math import *
import numpy as np
from colour import Color
import httplib2
import string as st
from scipy.linalg import null_space
from httplib2 import *
import sys
import quadprog
import torch

import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp
    

class Utils:
    """A library that contains some utilities for UAIbot. All of the functions are static."""

    #######################################
    # Constants
    #######################################

    _PI = 3.1415926
    _SQRTHALFPI = 1.2533141
    _SQRT2 = 1.4142135
    _CONSTJA = 2.7889
    _CONSTI0HAT1 = 0.24273
    _CONSTI0HAT2 = 0.43023

    UAIBOT_NAME_TYPES = ['uaibot.', 'cylinder.', 'box.', 'ball.', 'convexpolytope.', 'robot.', 'simulation.', 'meshmaterial.',
                             'texture.', 'pointlight.', 'frame.', 'model3d.', 'links.', 'pointcloud.', 'vector.', 'rigidobject.',
                             '.group', '.htmldiv', 'CPP_GeometricPrimitives', 'CPP_DistStructRobotObj','CPP_AABB', '.Pedestrian', 
                             '.ObstacleColumn', '.ObstacleThinWall']

    IS_SIMPLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope']
    
    IS_METRIC = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.PointCloud', 'uaibot.ConvexPolytope']

    IS_GROUPABLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Frame',
                    'uaibot.RigidObject', 'uaibot.Group', 'uaibot.Robot', 'uaibot.PointLight']

    IS_OBJ_SIM = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Robot',
                  'uaibot.PointLight', 'uaibot.Frame', 'uaibot.PointCloud', 'uaibot.Vector',
                  'uaibot.RigidObject', 'uaibot.Group', 'uaibot.HTMLDiv', 'uaibot.Pedestrian', 'uaibot.ObstacleColumn', 'uaibot.ObstacleThinWall']

    #######################################
    # Basic functions
    #######################################

    @staticmethod
    def S(v):
        """
      Returns a 3x3 matrix that implements the cross product for a 3D vector  
      as a matricial product, that is, a matrix S(v) such that for any other 
      3D column  vector w, S(v)w = cross(v,w).
      
      Parameters
      ----------
      v : a 3D vector
          The vector for which the S matrix will be created.

      Returns
      -------
      S : 3x3 numpy matrix
          A matrix that implements the cross product with v.
      """
        vv = np.matrix(v).reshape((3,1))
        return np.matrix([[0, -vv[2,0], vv[1,0]],
                         [vv[2,0], 0, -vv[0,0]],
                         [-vv[1,0], vv[0,0], 0]])

    @staticmethod
    def rot(axis, angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in an axis.
      
      Parameters
      ----------
      axis : a 3D vector
          The axis of rotation.
      
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        a = np.reshape(axis, (3,))
        a = a / np.linalg.norm(a)
        K = Utils.S(a)
        Q = np.identity(3) + sin(angle) * K + (1 - cos(angle)) * (K @ K)
        return np.hstack([np.vstack([Q, np.matrix([0, 0, 0])]), np.matrix([[0], [0], [0], [1]])])

    @staticmethod
    def trn(vector):
        """
      Homogeneous transformation matrix that represents the displacement
      of a vector
      
      Parameters
      ----------
      vector : a 3D vector
          The displacement vector.
      
      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        v = np.matrix(vector).reshape((3,1))
        return np.matrix([[1, 0, 0, v[0,0]],
                         [0, 1, 0, v[1,0]],
                         [0, 0, 1, v[2,0]],
                         [0, 0, 0, 1]])

    @staticmethod
    def rotx(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'x' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[1, 0, 0, 0],
                         [0, cos(angle), -sin(angle), 0],
                         [0, sin(angle), cos(angle), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def roty(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'y' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[cos(angle), 0, sin(angle), 0],
                         [0, 1, 0, 0],
                         [-sin(angle), 0, cos(angle), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def rotz(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'z' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[cos(angle), -sin(angle), 0, 0],
                         [sin(angle), cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def trn_torch(vector):
        """
      Homogeneous transformation matrix that represents the displacement
      of a vector using PyTorch.
      
      Parameters
      ----------
      vector : a 3D vector
          The displacement vector.
      
      Returns
      -------
      htm : 4x4 torch.Tensor
          The homogeneous transformation matrix.
      """
        v = torch.tensor(vector, dtype=torch.float32).view(3, 1)
        return torch.tensor([[1, 0, 0, v[0, 0]],
                             [0, 1, 0, v[1, 0]],
                             [0, 0, 1, v[2, 0]],
                             [0, 0, 0, 1]], dtype=torch.float32)

    @staticmethod
    def trn_torch_multi(vectors):
        """
        Batch homogeneous transformation matrix for translation given a set of 3D vectors.

        Parameters
        ----------
        vectors : torch.Tensor of shape (N, 3)
            A batch of 3D displacement vectors.

        Returns
        -------
        htm_batch : torch.Tensor of shape (N, 4, 4)
            A batch of homogeneous transformation matrices (one per vector).
        """
        N = vectors.shape[0]  # Número de vetores
        
        # Cria um tensor (N, 4, 4) com a matriz identidade
        htm_batch = torch.eye(4, dtype=torch.float32, device=vectors.device).unsqueeze(0).repeat(N, 1, 1)
        
        # Insere os componentes do vetor nas posições corretas
        htm_batch[:, :3, 3] = vectors  # Preenche os elementos [tx, ty, tz]
        
        return htm_batch

    @staticmethod
    def rotx_torch(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'x' axis using PyTorch.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 torch.Tensor
          The homogeneous transformation matrix.
      """
        return torch.tensor([[1, 0, 0, 0],
                             [0, torch.cos(angle), -torch.sin(angle), 0],
                             [0, torch.sin(angle), torch.cos(angle), 0],
                             [0, 0, 0, 1]], dtype=torch.float32)

    @staticmethod
    def roty_torch(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'y' axis using PyTorch.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 torch.Tensor
          The homogeneous transformation matrix.
      """
        return torch.tensor([[torch.cos(angle), 0, torch.sin(angle), 0],
                             [0, 1, 0, 0],
                             [-torch.sin(angle), 0, torch.cos(angle), 0],
                             [0, 0, 0, 1]], dtype=torch.float32)

    @staticmethod
    def rotz_torch(angle):
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'z' axis using PyTorch.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 torch.Tensor
          The homogeneous transformation matrix.
      """
        return torch.tensor([[torch.cos(angle), -torch.sin(angle), 0, 0],
                             [torch.sin(angle), torch.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.float32)

    @staticmethod
    def rotx_torch_multi(angles):
        """
        Batch homogeneous transformation matrix for rotation around the 'x' axis.
        
        Parameters
        ----------
        angles : torch.Tensor of shape (N,)
            A 1D tensor containing N angles (in radians).

        Returns
        -------
        htm_batch : torch.Tensor of shape (N, 4, 4)
            A batch of homogeneous transformation matrices (one per angle).
        """
        N = angles.shape[0]  # Número de ângulos
        
        # Inicializa um tensor (N, 4, 4) com a matriz identidade
        htm_batch = torch.eye(4, dtype=torch.float32, device=angles.device).unsqueeze(0).repeat(N, 1, 1)
        
        # Preenche os valores de rotação em X
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        
        htm_batch[:, 1, 1] = cos_theta   # cos(θ) na posição (1,1)
        htm_batch[:, 1, 2] = -sin_theta  # -sin(θ) na posição (1,2)
        htm_batch[:, 2, 1] = sin_theta   # sin(θ) na posição (2,1)
        htm_batch[:, 2, 2] = cos_theta   # cos(θ) na posição (2,2)
        
        return htm_batch

    @staticmethod
    def rotz_torch_multi(angles):
        """
        Batch homogeneous transformation matrix for rotation around the 'z' axis.
        
        Parameters
        ----------
        angles : torch.Tensor of shape (N,)
            A 1D tensor containing N angles (in radians).

        Returns
        -------
        htm_batch : torch.Tensor of shape (N, 4, 4)
            A batch of homogeneous transformation matrices (one per angle).
        """
        N = angles.shape[0]  # Número de ângulos
        
        # Inicializa um tensor (N, 4, 4) preenchido com zeros
        htm_batch = torch.zeros(N, 4, 4, dtype=torch.float32, device=angles.device)
        
        # Preenche os valores vetorizados
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        
        htm_batch[:, 0, 0] = cos_theta   # cos(θ)
        htm_batch[:, 0, 1] = -sin_theta  # -sin(θ)
        htm_batch[:, 1, 0] = sin_theta   # sin(θ)
        htm_batch[:, 1, 1] = cos_theta   # cos(θ)
        htm_batch[:, 2, 2] = 1.0         # 1 (diagonal)
        htm_batch[:, 3, 3] = 1.0         # 1 (diagonal)

        return htm_batch

    @staticmethod
    def htm_rand(trn_min=[0.,0.,0.], trn_max = [1.,1.,1.], rot=np.pi/2):
        """
      Returns a random homogeneous transformation matrix.

      Parameters
      ----------
      trn: float
          Maximum parameter for random translation in x, y and z.
          (default: 1)

      rot: float
          Maximum parameter for random rotation in x, y and z.
          (default: 1)

      Returns
      -------
      htm : 4x4 numpy matrix
          A homogeneous transformation matrix.
      """

        x = np.random.uniform(trn_min[0],trn_max[0])
        y = np.random.uniform(trn_min[1],trn_max[1])
        z = np.random.uniform(trn_min[2],trn_max[2])
        ax = np.random.uniform(-rot / 2, rot / 2)
        ay = np.random.uniform(-rot / 2, rot / 2)
        az = np.random.uniform(-rot / 2, rot / 2)

        return Utils.trn([x,y,z]) * Utils.rotx(ax) * Utils.roty(ay) * Utils.rotz(az)

    @staticmethod
    def inv_htm(htm):
        """
      Given a homogeneous transformation matrix, compute its inverse.
      It is faster than using numpy.linalg.inv in the case of HTMs.
      
      Parameters
      ----------
      htm: 4X4 numpy array or nested list 
          Homogeneous transformation matrix of the rotation.

      Returns
      -------
      inv_htm: 4X4 numpy array
          The inverse of the transformation matrix.       
      """
        Q = htm[0:3, 0:3]
        p = htm[0:3, 3]

        inv_htm = np.matrix(np.zeros((4, 4)))
        inv_htm[0:3, 0:3] = Q.T
        inv_htm[0:3, 3] = -Q.T * p
        inv_htm[3, 3] = 1

        return inv_htm

    @staticmethod
    def axis_angle(htm):
        """
      Given an homogeneous transformation matrix representing a rotation, 
      return the rotation axis angle.
      
      Parameters
      ----------
      htm: 4X4 numpy array or nested list 
          Homogeneous transformation matrix of the rotation.

      Returns
      -------
      axis : 3D numpy vector
          The rotation axis.

      angle : float
          The rotation angle, in radians.        
      """
        Q = htm[0:3, 0:3]
        trace = Q[0, 0] + Q[1, 1] + Q[2, 2]
        angle = acos((trace - 1) / 2)
        G = Q * Q - 2 * cos(angle) * Q + np.identity(3)
        ok = False
        while not ok:
            v = np.matrix(np.random.uniform(-100, 100, size=(3,1)))
            w = np.matrix(np.random.uniform(-100, 100, size=(3,1)))
            r = G * v
            nr = np.linalg.norm(r)
            prod = w.T * r
            if nr > 0.01:
                ortr = w -  (r * prod) / (nr * nr)
                axis = Utils.S(ortr) * (Q * ortr)
                naxis = np.linalg.norm(axis)
                ok = naxis > 0.01

        axis = axis / naxis
        return axis, angle

    @staticmethod
    def euler_angles(htm):
        """
        Computer the Euler angles of a rotation matrix.
        Find alpha, beta and gamma such that.

        htm = Utils.rotz(alpha) * Utils.roty(beta) * Utils.rotx(gamma).

        Parameters
        ----------
        htm: 4X4 numpy array or nested list
            Homogeneous transformation matrix of the rotation.

        Returns
        -------
        alpha : float
            Rotation in z, in radians.
        beta : float
            Rotation in y, in radians.
        gamma : float
            Rotation in x, in radians.
        """

        Q = np.matrix(htm[0:3, 0:3])
        sy = sqrt(Q[0, 0] ** 2 + Q[1, 0] ** 2)

        if abs(sy) > 0.001:
            gamma = np.arctan2(Q[2, 1], Q[2, 2])
            beta = np.arctan2(-Q[2, 0], sy)
            alpha = np.arctan2(Q[1, 0], Q[0, 0])
        else:
            gamma = np.arctan2(-Q[1, 2], Q[1, 1])
            beta = np.arctan2(-Q[2, 0], sy)
            alpha = 0

        return alpha, beta, gamma


    @staticmethod
    def dp_inv(A, eps = 0.001):
        """
      Compute the damped pseudoinverse of the matrix 'mat'.
      
      Parameters
      ----------
      A: nxm numpy array
          The matrix to compute the damped pseudoinverse.
      
      eps: positive float
          The damping factor.
          (default: 0.001).

      Returns
      -------
      pinvA: mxn numpy array
          The damped pseudoinverse of 'mat'.
      """
        A_int = np.matrix(A)
        n = np.shape(A)[1]
        return np.linalg.inv(A_int.T * A_int + eps * np.identity(n)) * A_int.T

    @staticmethod
    def dp_inv_solve(A, b, eps = 0.001, mode='auto'):
        """
      Solve the problem of minimizing ||A*x-b||^2 + eps*||x||^2
      It is the same as dp_inv(A,eps)*b, but faster
      
      Parameters
      ----------
      A: nxm numpy array
          Matrix.
          
      b: nx1 numpy array
          Vector.
                
      eps: positive float
          The damping factor.
          (default: 0.001).

      Returns
      -------
      x: mx1 numpy array
          The solution to the optimization problem.
      """
      
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")

        n, m = A.shape
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            
            M = np.block([
                [eps * np.eye(m), -A.T],
                [A, np.eye(n)]
            ])
            
            rhs = np.concatenate((np.zeros((m,1)), b))
            solution = np.linalg.solve(M, rhs)
            
            return np.matrix(solution[:m]).reshape((n,1))
        else:
            return np.matrix(ub_cpp.dp_inv_solve(A,b,eps)).reshape((n,1))
                 

    @staticmethod
    def hierarchical_solve(mat_a, mat_b, eps=0.001):
        """
      Solve the lexicographical unconstrained quadratic optimization problem

      lexmin_x ||mat_a[i]*x - b[i]||² + eps*||x||²

      with lower indexes having higher priority than higher indexes.

      Parameters
      ----------
      mat_a: A list of matrices (double arrays or numpy matrices).
          The matrices mat_a[i]. All must have the same number of columns.

      mat_b: A list of column vectors (double arrays or numpy matrices).
          The vectors mat_b[i]. The number of rows of mat_b[i] must be equal to the number
          of rows of mat_a[i].

      eps: positive float
          Damping parameter.
          (default: 0.001).

      Returns
      -------
      x: numpy column vector
          The solution x. For positive eps, the solution is always unique.
      """


        x_sol = Utils.dp_inv(mat_a[0], eps) * mat_b[0]

        if len(mat_a) > 1:

            null_mat_a = null_space(mat_a[0])

            if np.shape(null_mat_a)[1] > 0:
                mat_a_mod = []
                mat_b_mod = []
                for i in range(1, len(mat_a)):
                    mat_a_mod.append(mat_a[i] * null_mat_a)
                    mat_b_mod.append(mat_b[i] - mat_a[i] * x_sol)

                y_sol = Utils.hierarchical_solve(mat_a_mod, mat_b_mod, eps)
                return x_sol + null_mat_a * y_sol
            else:
                return x_sol
        else:
            return x_sol


    @staticmethod
    def interpolate(points):
        """
      Create a function handle that generates an one-time differentiable interpolated data from 'points'.

      The simplest case in when 'points' is a list with m elements. In this case, it will output a function f.
      When this function is evaluated at a scalar t, it will coincide with points[i] when t = i/m, that is,
      f(i/m) = points[i]. This function is once differentiable and periodic with period 1, so f(t+k)=f(t) for
      an integer k.

      The function can also use a n x m numpy array or lists as 'points'. In this case, f(t) is a n dimensional
      column vector in which its i-th entry is the same as computing f_i = interpolate(points[i]) and then
      computing f_i(t).

      Finally, t can be a list of k elements instead of just a scalar. In this case, f(t) is a n x k numpy matrix
      in which the element at row i and column j is the same as computing f_i = interpolate(points[i]) and then
      computing f_i(t[k]).


      Parameters
      ----------
      points: a n x m numpy array or lists
          Points to be interpolated.

      Returns
      -------
      f: function handle
          The function handle that implements the interpolation.
      """

        if not Utils.is_a_matrix(points):
            raise Exception("The parameter 'points' should be a n x m numpy array of numbers.")

        def aux_interpolate_single(arg_points):

            def int_aux_simp(arg_t, arg_n, arg_c):
                tn = arg_n * (arg_t % 1)
                ti = floor(tn)
                coef = arg_c[4 * ti: 4 * ti + 5]
                return coef[0] + coef[1] * tn + coef[2] * tn ** 2 + coef[3] * tn ** 3

            def int_aux(arg_t, arg_n, arg_c):
                return [int_aux_simp(tt, arg_n, arg_c) for tt in arg_t] if str(
                    type(arg_t)) == "<class 'list'>" else int_aux_simp(arg_t, arg_n, arg_c)

            n = len(arg_points)
            xn = np.array(arg_points).tolist()
            xn.append(arg_points[0])

            t = range(n + 1)
            A = np.zeros((3 * n, 4 * n))
            b = np.zeros((3 * n, 1))

            # Equality at initial points
            for p in range(n):
                A[p, 4 * p] = 1
                A[p, 4 * p + 1] = t[p]
                A[p, 4 * p + 2] = t[p] ** 2
                A[p, 4 * p + 3] = t[p] ** 3
                b[p] = xn[p]
            # Equality at final points
            for p in range(n):
                A[n + p, 4 * p] = 1
                A[n + p, 4 * p + 1] = t[p + 1]
                A[n + p, 4 * p + 2] = t[p + 1] ** 2
                A[n + p, 4 * p + 3] = t[p + 1] ** 3
                b[n + p] = xn[p + 1]
            # Equality of the derivative in the initial points
            for p in range(n):
                if not (p == n - 1):
                    A[2 * n + p, 4 * p] = 0
                    A[2 * n + p, 4 * p + 1] = 1
                    A[2 * n + p, 4 * p + 2] = 2 * t[p + 1]
                    A[2 * n + p, 4 * p + 3] = 3 * t[p + 1] ** 2
                    A[2 * n + p, 4 * (p + 1)] = -0
                    A[2 * n + p, 4 * (p + 1) + 1] = -1
                    A[2 * n + p, 4 * (p + 1) + 2] = -2 * t[p + 1]
                    A[2 * n + p, 4 * (p + 1) + 3] = -3 * t[p + 1] ** 2
                else:
                    A[2 * n + p, 4 * p] = 0
                    A[2 * n + p, 4 * p + 1] = 1
                    A[2 * n + p, 4 * p + 2] = 2 * t[p + 1]
                    A[2 * n + p, 4 * p + 3] = 3 * t[p + 1] ** 2
                    A[2 * n + p, 0] = -0
                    A[2 * n + p, 1] = -1
                    A[2 * n + p, 2] = 0
                    A[2 * n + p, 3] = 0

            # Create the objective function

            H = np.zeros((0, 0))
            for p in range(n):
                M = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 6 * ((p + 1) ** 2 - p ** 2)],
                              [0, 0, 6 * ((p + 1) ** 2 - p ** 2), 12 * ((p + 1) ** 3 - p ** 3)]])
                nn = np.shape(H)[0]
                H = np.block([[H, np.zeros((nn, 4))], [np.zeros((4, nn)), M]])

            # Solve the optimization problem
            m1 = np.shape(A)[0]
            m2 = np.shape(A)[1]

            G = np.block([[H, np.transpose(A)], [A, np.zeros((m1, m1))]])
            g = np.block([[np.zeros((m2, 1))], [b]])
            y = np.linalg.solve(G, g)
            c = y[0: np.shape(H)[0]]
            c = c.reshape((1, np.shape(H)[0]))[0].tolist()

            # Create the function
            f = lambda ts: int_aux(ts, n, c)
            return f

        def aux_interpolate_multiple(arg_points, t):

            if not (Utils.is_a_vector(t) or Utils.is_a_number(t)):
                raise Exception(
                    "The parameter of the interpolation function must be either a number or a list of numbers.")

            y = np.zeros((0, len(t) if Utils.is_a_vector(t) else 1))
            for i in range(np.shape(arg_points)[0]):
                fun = aux_interpolate_single(arg_points[i])
                fun_out = fun(t)
                fun_out = np.array(fun_out).reshape((1, len(fun_out) if Utils.is_a_vector(fun_out) else 1))
                y = np.block([[y], [fun_out]])

            return np.matrix(y)

        return lambda t: aux_interpolate_multiple(points, t)

    @staticmethod
    def solve_qp(H, f, A, b):
        #Solves u^THu/2 + f^Tu subject to Au>=b
    
        H = 0.5 * (H + H.T)  

        m = H.shape[0]
        n = A.shape[0]

        f = np.asarray(f).reshape((m,))   
        b = np.asarray(b).reshape((n,))   

        C = A.T  
        
        result = quadprog.solve_qp(H, -f, C, b, 0)
        
        return np.matrix(result[0].reshape((m,1)))

    #######################################
    # Type check functions
    #######################################

    @staticmethod
    def is_a_number(obj):
        """
      Check if the argument is a float or int number
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        return str(type(obj)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"]

    @staticmethod
    def is_a_natural_number(obj):
        """
      Check if the argument is a natural number (integer and >=0)
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        return str(type(obj)) == "<class 'int'>" and obj >= 0


    @staticmethod
    def is_a_matrix(obj, n=None, m=None):
        """
        Check if the argument is an n x m matrix of integers or floats.

        Parameters
        ----------
        obj: object
            Object to be verified.

        n: positive int
            Number of rows
            (default: it does not matter).

        m: positive int
            Number of columns
            (default: it does not matter).

        Returns
        -------
        is_type: boolean
            If the object is of the type.
        """

        # Check if obj is a numpy array or matrix
        if isinstance(obj, (np.ndarray, np.matrix)):
            # Check shape if dimensions are provided

            shp = np.shape(obj)
            rows = shp[0]
            cols = shp[1] if len(shp)>1 else  1

            if ((n is not None) and (rows != n)) or ((m is not None) and (cols !=m)):
                return False

            # Ensure all elements are integers or floats
            return np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer)

        # Check if obj is a list (and therefore could represent a matrix)
        if isinstance(obj, list):
            try:
                return Utils.is_a_matrix(np.matrix(obj), n, m)
            except:
                return False
    
    @staticmethod
    def is_a_vector(obj, n=None):
        """
      Check if the argument is a n vector of floats.
      
      Parameters
      ----------
      obj: object
          Object to be verified.

      n: positive int
          Number of elements
          (default: it does not matter).

      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        return Utils.is_a_matrix(obj, n, 1) or Utils.is_a_matrix(obj, 1, n)

    @staticmethod
    def is_a_pd_matrix(obj, n=None):
        """
      Check if the argument is a symmetric nxn positive (semi)-definite matrix.
      
      Parameters
      ----------
      obj: object
          Object to be verified.

      n: positive int
          Dimension of the square matrix
          (default: it does not matter).
    
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        value = Utils.is_a_matrix(obj, n, n)

        if value:
            value = np.allclose(obj, np.transpose(obj), rtol=1e-05, atol=1e-08)

        if value:
            try:
                np.linalg.cholesky(obj)
            except:
                value = False

        return value

    @staticmethod
    def is_a_color(obj):
        """
      Check if the argument is a HTML-compatible string that represents a color.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        try:
            obj = obj.replace(" ", "")
            Color(obj)
            return True
        except:
            return False

    @staticmethod
    def is_a_name(string):
        """
      Check if the argument is a valid name for uaibot objects.
      Only characters [a-z], [A-z], [0-9] and '_' are allowed.
      However, variables should not begin with numbers.

      Parameters
      ----------
      string: string
          Name to be verified.

      Returns
      -------
      is_name: boolean
          If the name is a valid name.
      """

        if str(type(string)) == "<class 'str'>":
            if string=='':
                return True
            else:
                allowed1 = set(st.ascii_lowercase + st.ascii_uppercase + st.digits + "_")
                allowed2 = set(st.ascii_lowercase + st.ascii_uppercase + "_")

                return (set(string) <= allowed1) and (set(string[0]) <= allowed2)
        else:
            return False

    @staticmethod
    def get_uaibot_type(obj):
        """
      Return the UAIBot type of the object.
      Return the empty string if it is not an UAIBot object.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      obj_type: string
          UAIBot type.
      """
        type_str = str(type(obj))

        ind = -1
        k = 0
        while ind == -1 and k < len(Utils.UAIBOT_NAME_TYPES):
            ind = type_str.find(Utils.UAIBOT_NAME_TYPES[k])
            k += 1

        if ind == -1:
            return ""
        else:
            ind1 = type_str.rfind('.')
            ind2 = type_str.rfind('>')
            return "uaibot." + type_str[ind1 + 1:ind2 - 1]

    @staticmethod
    def is_a_simple_object(obj):
        """
      Check if the argument is a simple object.
      Check the constant 'Utils.IS_SIMPLE' for a list of simple objects.

      Parameters
      ----------
      obj: object
          Object to be verified.

      Returns
      -------
      is_type: boolean
          If the object is of the type.
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_SIMPLE

    @staticmethod
    def is_a_metric_object(obj):
        """
      Check if the argument is a metric object (if distance can be computed).
      Check the constant 'Utils.IS_METRIC' for a list of metric objects.

      Parameters
      ----------
      obj: object
          Object to be verified.

      Returns
      -------
      is_type: boolean
          If the object is of the type.
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_METRIC
    
    @staticmethod
    def is_a_groupable_object(obj):
        """
      Check if the argument is a groupable object.
      Check the constant 'Utils.IS_GROUPABLE' for a list of groupable objects.

      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_GROUPABLE

    @staticmethod
    def is_a_obj_sim(obj):
        """
      Check if the argument is an object that can be put into the simulator.
      Check the constant 'Utils.IS_OBJ_SIM' for a list of objects that can be put in the simulator.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_OBJ_SIM

    @staticmethod
    def is_url_available(url, types):
        """
      Try to access the content of the url 'url'. Also verifies if the content is one of the extensions contained in
      'types' (e.g, types = ['png', 'bmp', 'jpg', 'jpeg'] for images).

      Never throws an Exception, always returning a string with a message. Returns 'ok!' if and only if the
      url was succesfully acessed and has the correct file type.

      Parameters
      ----------
      url: string
          The url string.

      types: list of string
          The desired content extensions.

      Returns
      -------
      message: string
          Message.
      """

        if not (str(type(url)) == "<class 'str'>"):
            return " is not a valid url"

        ind = url.rfind('.')
        filetype = url[ind + 1:]

        if not (filetype in types):
            return " must be an file of the types: '" + str(types) + "'"

        try:
            h = httplib2.Http(disable_ssl_certificate_validation=True) #Change this later
            resp = h.request(url, 'HEAD')
            if int(resp[0]['status']) < 400:
                return "ok!"
            else:
                return " : not able to access '" + url + "'."
        except:
            return " : not able to access '" + url + "'."

    @staticmethod
    def get_environment():
        try:
            import google.colab
            return "Colab"
        except ImportError:
            try:
                from IPython import get_ipython
                if "google.colab" in sys.modules:
                    return "Colab"
                elif get_ipython() is not None:
                    return "Jupyter"
            except ImportError:
                return "Not in Jupyter/Colab"
        return "None"
    
    #######################################
    # Distance computation functions
    #######################################

    @staticmethod
    def softmin(x,h):
        minval = np.min(x)
        s=0

        for val in x:
            s+= exp(-h*(val-minval))

        return minval -(1/h)*np.log(s)

    @staticmethod
    def softselectmin(x, y, h):
        minval = np.min(x)
        s = 0

        coef = []
        for val in x:
            coef.append(exp(-(val - minval)/h))
            s += coef[-1]

        coef = [c/s for c in coef]

        sselect = 0 * y[0]
        for i in range(len(coef)):
            sselect += coef[i] * y[i]

        return sselect, minval - h * np.log(s/len(coef))


    @staticmethod
    def softmax(x, h):
        return Utils.softmin(x,-h)

    @staticmethod
    def softselectmax(x, y, h):
        return Utils.softselectmin(x,y,-h)

    @staticmethod
    def compute_aabbdist(obj1, obj2):

        w1, d1, h1 = obj1.aabb()
        w2, d2, h2 = obj2.aabb()
        delta = obj1.htm[0:3, -1] - obj2.htm[0:3, -1]

        dx = max(abs(delta[0, 0]) - (w1 + w2) / 2, 0)
        dy = max(abs(delta[1, 0]) - (d1 + d2) / 2, 0)
        dz = max(abs(delta[2, 0]) - (h1 + h2) / 2, 0)

        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)



    @staticmethod
    def compute_dist_python(obj_a, obj_b, p_a, tol=0.001, no_iter_max=20):

        converged = False
        i = 0

        while (not converged) and i < no_iter_max:
            p_a_ant = p_a
            p_b, _ = obj_b.projection(p_a)
            p_a, _ = obj_a.projection(p_b)
            converged = np.linalg.norm(p_a - p_a_ant) < tol
            i += 1

        dist = np.linalg.norm(p_a - p_b)

        return p_a, p_b, dist
    
    @staticmethod
    def obj_to_cpp(obj, htm=None):
            
            if Utils.get_uaibot_type(obj) != 'uaibot.PointCloud' and htm is None:
                htm = obj.htm
                
                
            if Utils.get_uaibot_type(obj) == 'uaibot.Ball':
                return ub_cpp.CPP_GeometricPrimitives.create_sphere(htm, obj.radius)
            if Utils.get_uaibot_type(obj) == 'uaibot.Box':
                return ub_cpp.CPP_GeometricPrimitives.create_box(htm, obj.width, obj.depth, obj.height)
            if Utils.get_uaibot_type(obj) == 'uaibot.Cylinder':
                return ub_cpp.CPP_GeometricPrimitives.create_cylinder(htm, obj.radius, obj.height)
            if Utils.get_uaibot_type(obj) == 'uaibot.PointCloud':
                return obj.cpp_pointcloud 
            if Utils.get_uaibot_type(obj) == 'uaibot.ConvexPolytope':
                Q = htm[0:3,0:3]
                p = htm[0:3,-1]
                A = obj.A * Q.transpose()
                b = obj.b + obj.A * Q.transpose()*p
                return ub_cpp.CPP_GeometricPrimitives.create_convexpolytope(htm, A, b)
            
    @staticmethod
    def compute_dist(obj_a, obj_b, p_a_init=None, tol=0.001, no_iter_max=20, h=0, eps = 0, mode='auto'):
        # Error handling

        if mode=='python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            if Utils.get_uaibot_type(obj_a) == 'uaibot.PointCloud' or Utils.get_uaibot_type(obj_b) == 'uaibot.PointCloud':
                raise Exception("Point cloud distance is not supported in Python mode!")
            
            if not Utils.is_a_metric_object(obj_a):
                raise Exception("In python mode, the parameter 'obj_a' must be one of the following: " + str(Utils.IS_METRIC) + ".")

            if not Utils.is_a_metric_object(obj_b):
                raise Exception("In python mode, the parameter 'obj_b' must be one of the following: " + str(Utils.IS_METRIC) + ".")
            
        if mode=='c++':
            if not (Utils.get_uaibot_type(obj_a) == 'uaibot.CPP_GeometricPrimitives' or Utils.is_a_metric_object(obj_a)):
                raise Exception("In c++ mode, the parameter 'obj_a' must be 'uaibot.CPP_GeometricPrimitives' or "+ str(Utils.IS_METRIC) + ".")
            if not (Utils.get_uaibot_type(obj_b) == 'uaibot.CPP_GeometricPrimitives' or Utils.is_a_metric_object(obj_b)):
                raise Exception("In c++ mode, the parameter 'obj_b' must be 'uaibot.CPP_GeometricPrimitives' or "+ str(Utils.IS_METRIC) + ".")            


        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            if Utils.is_a_metric_object(obj_a):
                obj_a_cpp = Utils.obj_to_cpp(obj_a) 
            else:
                obj_a_cpp = obj_a

            if Utils.is_a_metric_object(obj_b):
                obj_b_cpp = Utils.obj_to_cpp(obj_b) 
            else:
                obj_b_cpp = obj_b
           
            
        if p_a_init is None:
            if ((mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1')) and h >= 1e-5:
                dist_res = obj_a_cpp.dist_to(obj_b_cpp, 0.0, 0.0, tol, no_iter_max, np.matrix([0,0,0]).reshape((3,1)))
                p_a = np.matrix(dist_res.proj_A).reshape((3,1))
            else:
                p_a = np.random.uniform(-3, 3, size=(3,))
        else:
            p_a = np.array(p_a_init)


        if not (p_a_init is None or Utils.is_a_vector(p_a_init, 3)):
            raise Exception("The optional parameter 'p_a_init' must be a 3D vector or 'None'.")

        if not Utils.is_a_number(tol) or tol <= 0:
            raise Exception("The optional parameter 'tol' must be a nonnegative number.")

        if not Utils.is_a_natural_number(no_iter_max):
            raise Exception("The optional parameter 'no_iter_max' must be a nonnegative integer.")
        
        if not Utils.is_a_number(h) or h < 0:
            raise Exception("The optional parameter 'h' must be a nonnegative number.")

        if not Utils.is_a_number(eps) or eps < 0:
            raise Exception("The optional parameter 'eps' must be a nonnegative number.")
                
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        # end error handling

        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            if h >0 or eps > 0:
                raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
            
            return Utils.compute_dist_python(obj_a, obj_b, p_a_init, tol, no_iter_max)
        else:
            dist_res = obj_a_cpp.dist_to(obj_b_cpp, h, eps, tol, no_iter_max, p_a)
            return np.matrix(dist_res.proj_A).reshape((3,1)), np.matrix(dist_res.proj_B).reshape((3,1)), dist_res.dist, dist_res.hist_error








