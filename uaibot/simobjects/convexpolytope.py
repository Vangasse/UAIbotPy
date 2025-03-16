from utils import *
import numpy as np
from graphics.meshmaterial import *
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from simobjects.box import *
import os

def is_unbounded(A, b):
    n_dim = A.shape[1]

    n_dim = A.shape[1]

    for i in range(n_dim):
        c = np.zeros(n_dim)
        c[i] = 1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status == 3 or abs(res.x[i])>1e3: 
            return False  

        c[i] = -1  

        res = linprog(c, A_ub=A, b_ub=b, method='highs', bounds=[(None, None)] * n_dim)

        if res.status == 3 or abs(res.x[i])>1e3: 
            return False  

    return False

def compute_polytope(A, b):
    n_dim = A.shape[1]
    
    c = np.zeros(n_dim)
    res = linprog(c, A_ub=A, b_ub=b-1e-6, method='highs', bounds=(None,None))

    if not res.success:
        raise ValueError("The polytope is empty.")

    interior_point = res.x  

    if is_unbounded(A, b):
        raise ValueError("The polytope is unbounded.")
    

    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    hs = HalfspaceIntersection(halfspaces, interior_point)
    vertices = hs.intersections

    hull = ConvexHull(vertices)
    faces = hull.simplices
    
    for i, face in enumerate(faces):
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        center = np.mean(vertices[face], axis=0)
        if np.dot(normal, center) < 0:  
            faces[i] = face[::-1]  

    return vertices.tolist(), faces.tolist()

def compute_htm(points):

    points = np.array(points)
    
    center = np.mean(points, axis=0).reshape(3,1)  
    
    centered_points = points - center.T
    
    M = centered_points.transpose()@ centered_points 
    
    _, _, Vt = np.linalg.svd(M) 
    
    transformation_matrix = np.eye(4) 
    transformation_matrix[:3, :3] = Vt.T  
    transformation_matrix[:3, 3] = center.flatten()  
    
    return np.matrix(transformation_matrix)

class ConvexPolytope:
    """
  A convex polytope object.

  Parameters
  ----------
  htm : 4x4 numpy array or 4x4 nested list or None
      Decide the object's **initial** placement of its frame 
      If 'None', a frame placement is computed automatically using SVD
      (default: 'None').

  name : string
      The object's name.
      (default: '' (automatic)).

  A : (n,3)-shaped numpy matrix
      The A matrix that forms the convex polytope according to Ap<=b.


  b : (n,1)-shaped numpy matrix
      The b matrix that forms the convex polytope according to Ap<=b.
      
  mass : positive float
      The object's mass, in kg.
      (default: 1).

  color : string
      The object's color, a HTML - compatible string.
      (default: "red").

  opacity : float between 0 and 1
      The opacity. 1 = fully opaque, and 0 = transparent.

  mesh_material: 'MeshMaterial' object
      The object mesh material. If set to 'None', the default is used.
      (default: None).
  """

    #######################################
    # Attributes
    #######################################

    @property
    def A(self):
        """The A matrix that forms the convex polytope according to Ap<=b."""
        return self._A

    @property
    def b(self):
        """The b matrix that forms the convex polytope according to Ap<=b."""
        return self._b

    @property
    def vertexes(self):
        """All the vertexes of the polygon"""
        return self._vertexes

    @property
    def faces(self):
        """All the faces of the polygon"""
        return self._faces
        
    @property
    def name(self):
        """The object name."""
        return self._name

    @property
    def htm(self):
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return np.matrix(self._htm)

    @property
    def mass(self):
        """Mass of the object, in kg."""
        return self._mass

    @property
    def color(self):
        """Color of the object"""
        return self.mesh_material.color

    @property
    def mesh_material(self):
        """Mesh material properties of the object"""
        return self._mesh_material

    #######################################
    # Constructor
    #######################################

    def __init__(self, htm=None, name="", A=[], b=[], mass=1, color="red", opacity=1, \
                 mesh_material=None):

        # Error handling
        if not (Utils.is_a_matrix(htm, 4, 4) or htm is None):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix, or 'None'.")

        if not Utils.is_a_number(mass) or mass < 0:
            raise Exception("The parameter 'mass' should be a positive float.")

        if not Utils.is_a_matrix(A,None,3) :
            raise Exception("The parameter 'A' should be a matrix with 3 columns.")

        if not Utils.is_a_vector(b):
            raise Exception("The parameter 'b' should be a vector.")
        
        if not np.shape(A)[0]==np.shape(b)[0]:
            raise Exception("The number of rows of 'A' and 'b' should be the same.")
        
        

        if name=="":
            name="var_convexpolytope_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a HTML-compatible color.")

        if not ((mesh_material is None) or (Utils.get_uaibot_type(mesh_material) == "uaibot.MeshMaterial")):
            raise Exception(
                "The parameter 'mesh_material' should be either 'None' or a 'uaibot.MeshMaterial' object.")

        if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
            raise Exception("The parameter 'opacity' should be a float between 0 and 1.")
        # end error handling
        
        #If htm is not set, compute it automatically using SVD
        if htm is None:
            vertexes, _ = compute_polytope(A, b)
            htm = compute_htm(vertexes)
            
        #Transform the shape according the initial htm0
        n = np.shape(A)[0]
        htm_m = np.matrix(htm)
        Q = htm_m[0:3,0:3]
        pc = htm_m[0:3,-1]
        A_m = np.matrix(A)*Q
        b_m = np.matrix(b).reshape((n,1))-np.matrix(A)*pc
        
        self._vertexes, self._faces = compute_polytope(A_m, b_m)
        
        self._A = np.matrix(A_m)
        self._b = np.matrix(b_m).reshape((n,1))
        self._htm = htm_m
        self._name = name
        self._mass = 1
        self._frames = []
        self._max_time = 0
        
        xmin=1e6
        xmax=-1e6
        ymin=1e6
        ymax=-1e6
        zmin=1e6
        zmax=-1e6
        
        for v in self._vertexes:
            xmin = min(xmin,v[0])
            xmax = max(xmax,v[0])
            ymin = min(ymin,v[1])
            ymax = max(ymax,v[1])   
            zmin = min(zmin,v[2])
            zmax = max(zmax,v[2]) 
            
        self._lx = xmax-xmin
        self._ly = ymax-ymin
        self._lz = zmax-zmin
        self._p = np.matrix([(xmax+xmin)/2, (ymax+ymin)/2,  (zmax+zmin)/2]).transpose()
                 

        if mesh_material is None:
            self._mesh_material = MeshMaterial(color=color, opacity=opacity, side="DoubleSide")
        else:
            self._mesh_material = mesh_material

        # Set initial total configuration
        self.set_ani_frame(self._htm)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Convex Polytope with name '" + self.name + "': \n\n"
        string += " Number of vertexes: " + str(len(self.vertexes)) + "\n"
        string += " Number of faces: " + str(len(self.faces)) + "\n"
        string += " Color: " + str(self.color) + "\n"
        string += " Mass (kg): " + str(self.mass) + "\n"
        string += " HTM: \n" + str(self.htm) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time, htm=None):
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.
    htm : 4x4 numpy array or 4x4 nested list
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """
        if htm is None:
            htm = self._htm

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a positive float.")
        # end error handling

        f = [time, np.around(htm[0,0],4), np.around(htm[0,2],4), np.around(-htm[0,1],4), np.around(htm[0,3],4),
             np.around(htm[1,0],4), np.around(htm[1,2],4), np.around(-htm[1,1],4), np.around(htm[1,3],4),
             np.around(htm[2,0],4), np.around(htm[2,2],4), np.around(-htm[2,1],4), np.around(htm[2,3],4),
             0, 0, 0, 1]

        self._htm = htm
        self._frames.append(f)
        self._max_time = max(self._max_time, time)

    def set_ani_frame(self, htm=None):
        """
    Reset object's animation queue and add a single configuration to the 
    object's animation queue.

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        if htm is None:
            htm = self._htm

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        # end error handling

        self._frames = []
        self.add_ani_frame(0, htm)
        self._max_time = 0

    def gen_code(self):
        """Generate code for injection."""

        string = "\n"
        string += "//BEGIN DECLARATION OF THE CONVEX POLYTOPE '" + self.name + "'\n\n"
        string += self.mesh_material.gen_code(self._name) + "\n"
        string += "const var_" + self._name + " = new ConvexPolytope(" + str(self._vertexes) + "," + str(
            self._faces) + "," + str(self._frames) + ", material_" + self._name + ");\n"
        string += "sceneElements.push(var_" + self._name + ");\n"
        string += "//USER INPUT GOES HERE"

        return string

    def copy(self):
        """Return a deep copy of the object, without copying the animation frames."""
        return ConvexPolytope(self.htm, self.name + "_copy", self.A, self.b, self.color)

    def aabb(self, mode='auto'):
        """
    Compute an AABB (axis-aligned bounding box), considering the current orientation of the object.

    Parameters
    ----------
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto') 
            
    Returns
    -------
     aab: the AABB as a uaibot.Box object
    """

        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = Utils.obj_to_cpp(self) 
            
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            
            x = self.htm[0:3,0]
            y = self.htm[0:3,1]
            z = self.htm[0:3,2]
            Q = self.htm[0:3,0:3]
            p = self.htm[0:3,-1]
            
            p1 = self._lx * x + self._ly * y + self._lz * z
            p2 = -self._lx * x + self._ly * y + self._lz * z
            p3 = self._lx * x - self._ly * y + self._lz * z
            p4 = self._lx * x + self._ly * y - self._lz * z
            
            lx = max(abs(p1[0]), abs(p2[0]), abs(p3[0]), abs(p4[0]))
            ly = max(abs(p1[1]), abs(p2[1]), abs(p3[1]), abs(p4[1]))
            lz = max(abs(p1[2]), abs(p2[2]), abs(p3[2]), abs(p4[2]))
            
            lx = np.max([abs(p1[0, 0]), abs(p2[0, 0]), abs(p3[0, 0]), abs(p4[0, 0])])
            ly = np.max([abs(p1[1, 0]), abs(p2[1, 0]), abs(p3[1, 0]), abs(p4[1, 0])])
            lz = np.max([abs(p1[2, 0]), abs(p2[2, 0]), abs(p3[2, 0]), abs(p4[2, 0])])
            
            pc = Q*self._p + p
            
            return Box(name = "aabb_"+self.name, width= lx, depth=ly, height=lz, htm=Utils.trn(pc),opacity=0.5)

        else:
            aabb = obj_cpp.get_aabb()
            return Box(name = "aabb_"+self.name, width= aabb.lx, depth=aabb.ly, height=aabb.lz, htm=Utils.trn(aabb.p),opacity=0.5) 


    def to_point_cloud(self, disc=0.025, mode='auto'):
        """
    Transform the object into a PointCloud object using the discretization 'delta'.

    Parameters
    ----------
    
    disc: positive float
        Discretization.
        (default: 0.025)

    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto') 
            
    Returns
    -------
     pointcloud: the pointcloud object.
    """

        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = Utils.obj_to_cpp(self) 
            
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            eps = 1e-6  
            all_face_points = []
            
            A = np.array(self.A, dtype=float)
            b = np.array(self.b, dtype=float).flatten()
            
            vertices_np = [np.array(v, dtype=float) for v in self.vertexes]
            
            for i in range(A.shape[0]):
                a = A[i, :]      
                bi = b[i]        
                
                face_vertices = []
                for v in vertices_np:
                    if abs(np.dot(a, v) - bi) < eps:
                        face_vertices.append(v)
                        
                if not face_vertices:
                    continue  
                

                n = a / np.linalg.norm(a)
                if abs(n[0]) < 0.9:
                    arbitrary = np.array([1.0, 0.0, 0.0])
                else:
                    arbitrary = np.array([0.0, 1.0, 0.0])

                d1 = arbitrary - np.dot(arbitrary, n) * n
                d1 = d1 / np.linalg.norm(d1)
                d2 = np.cross(n, d1)
                
            
                origin = face_vertices[0]
                u_vals = []
                v_vals = []
                for v in face_vertices:
                    diff = v - origin
                    u_vals.append(np.dot(diff, d1))
                    v_vals.append(np.dot(diff, d2))
                    
                u_min, u_max = min(u_vals), max(u_vals)
                v_min, v_max = min(v_vals), max(v_vals)
                
                u_range = np.arange(u_min, u_max + disc, disc)
                v_range = np.arange(v_min, v_max + disc, disc)
                for u in u_range:
                    for v_coord in v_range:
                        candidate = origin + u * d1 + v_coord * d2
                        if np.all(np.dot(A, candidate) <= b + eps):
                            all_face_points.append(candidate.tolist())
                            
            all_face_points_transformed = []
            for points in all_face_points:
                tr_point = self.htm[0:3,0:3]*np.matrix(points).reshape((3,1))+self.htm[0:3,-1] 
                all_face_points_transformed.append(tr_point)
                            
            return PointCloud(points = all_face_points_transformed, color = self.color, size = disc/2)
        else:
            return PointCloud(points = obj_cpp.to_pointcloud(disc).points_gp, color = self.color, size = disc/2)
        
    # Compute distance to an object
    def compute_dist(self, obj,  p_init=None, tol=0.001, no_iter_max=20, h=0, eps = 0, mode='auto'):
        return Utils.compute_dist(self, obj, p_init, tol, no_iter_max, h, eps, mode)
    
    # Compute the projection of a point into an object
    def projection(self, point, h=0, eps = 0, mode='auto'):
        """
    The projection of a point in the object, that is, the
    closest point in the object to a point 'point'.

    Parameters
    ----------
    point : 3D vector
        The point for which the projection will be computed.

    h : positive float
        Smoothing parameter (only valid in c++ mode)
        (default: 0).            

    eps : positive float
        Smoothing parameter (only valid in c++ mode)
        (default: 0).      
        
    Returns
    -------
     proj_point : 3D vector
        The projection of the point 'point' in the object.

     d : positive float
        The distance between the object and 'point'.
    """


        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = Utils.obj_to_cpp(self)
            
        if ( ( h > 0 or eps > 0) and ((mode == 'python') or ((mode=='auto' and os.environ['CPP_SO_FOUND']=='0')))):
            raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
               
        if not Utils.is_a_number(h) or h < 0:
            raise Exception("The optional parameter 'h' must be a nonnegative number.")

        if not Utils.is_a_number(eps) or eps < 0:
            raise Exception("The optional parameter 'eps' must be a nonnegative number.")
        
        if not Utils.is_a_vector(point, 3):
            raise Exception("The parameter 'point' should be a 3D vector.")
        
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")


        # end error handling
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            tpoint = self._htm[0:3, 0:3].T * (point - self._htm[0:3, 3])
            
            tpi = Utils.solve_qp(np.identity(3), -tpoint, -self.A, -self.b)
            d = np.linalg.norm(tpoint-tpi)
            return self._htm[0:3, 0:3] * tpi + self._htm[0:3, 3], d
        else:
            pr = obj_cpp.projection(np.matrix(point).reshape((3,1)), h, eps)
            return np.matrix(pr.proj).transpose(), pr.dist            
