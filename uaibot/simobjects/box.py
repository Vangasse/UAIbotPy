from utils import *
import numpy as np
from graphics.meshmaterial import *
import os
from simobjects.pointcloud import *

class Box:
    """
  A box object.

  Parameters
  ----------
  htm : 4x4 numpy array or 4x4 nested list
      The object's configuration
      (default: the same as the current HTM).

  name : string
      The object's name.
      (default: '' (automatic)).

  width : positive float
      The object's width, in meters.
      (default: 1).    

  depth : positive float
      The object's depth, in meters.
      (default: 1).  

  height : positive float
      The object's height, in meters.
      (default: 1).  

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
    def width(self):
        """The box width, in meters."""
        return self._width

    @property
    def height(self):
        """The box height, in meters."""
        return self._height

    @property
    def depth(self):
        """The box depth, in meters."""
        return self._depth

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

    def __init__(self, htm=np.identity(4), name="", width=1, height=1, depth=1, mass=1, color="red", opacity=1, \
                 mesh_material=None):

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(mass) or mass < 0:
            raise Exception("The parameter 'mass' should be a positive float.")

        if not Utils.is_a_number(width) or width < 0:
            raise Exception("The parameter 'width' should be a positive float.")

        if not Utils.is_a_number(height) or height < 0:
            raise Exception("The parameter 'height' should be a positive float.")

        if not Utils.is_a_number(depth) or depth < 0:
            raise Exception("The parameter 'depth' should be a positive float.")

        if name=="":
            name="var_box_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a HTML-compatible color.")

        if not ((mesh_material is None) or (Utils.get_uaibot_type(mesh_material) == "uaibot.MeshMaterial")):
            raise Exception(
                "The parameter 'mesh_material' should be either 'None' or a 'uaibot.MeshMaterial' object.")

        if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
            raise Exception("The parameter 'opacity' should be a float between 0 and 1")
            # end error handling

        self._width = width
        self._height = height
        self._depth = depth
        self._htm = np.matrix(htm)
        self._name = name
        self._mass = 1
        self._frames = []
        self._max_time = 0

        if mesh_material is None:
            self._mesh_material = MeshMaterial(color=color, opacity=opacity)
        else:
            self._mesh_material = mesh_material

        # Set initial total configuration
        self.set_ani_frame(self._htm)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Box with name '" + self.name + "': \n\n"
        string += " Width (m): " + str(self.width) + "\n"
        string += " Depth (m): " + str(self.depth) + "\n"
        string += " Height (m): " + str(self.height) + "\n"
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
        string += "//BEGIN DECLARATION OF THE BOX '" + self.name + "'\n\n"
        string += self.mesh_material.gen_code(self.name) + "\n"
        string += "const var_" + self.name + " = new Box(" + str(self.width) + "," + str(
            self.height) + "," + str(self.depth) + "," + str(self._frames) + ", material_" + self.name + ");\n"
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"
        return string

    def copy(self):
        """Return a deep copy of the object, without copying the animation frames."""
        return Box(self.htm, self.name + "_copy", self.width, self.height, self.depth, self.mass, self.color)

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
            p1 = self.width * self.htm[:,0] + self.depth * self.htm[:,1] + self.height * self.htm[:,2]
            p2 = -self.width * self.htm[:, 0] + self.depth * self.htm[:, 1] + self.height * self.htm[:, 2]
            p3 = self.width * self.htm[:, 0] - self.depth * self.htm[:, 1] + self.height * self.htm[:, 2]
            p4 = self.width * self.htm[:, 0] + self.depth * self.htm[:, 1] - self.height * self.htm[:, 2]

            w = np.max([abs(p1[0, 0]), abs(p2[0, 0]), abs(p3[0, 0]), abs(p4[0, 0])])
            d = np.max([abs(p1[1, 0]), abs(p2[1, 0]), abs(p3[1, 0]), abs(p4[1, 0])])
            h = np.max([abs(p1[2, 0]), abs(p2[2, 0]), abs(p3[2, 0]), abs(p4[2, 0])])
            
            return Box(name = "aabb_"+self.name, width= w, depth=d, height=h, htm=Utils.trn(self.htm[0:3,-1]),opacity=0.5)
        
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
            P = np.matrix(np.zeros((3, 0)))

            T = round(2*np.pi*self.radius / disc)+1
            R = round(self.radius/disc)+1
            H = round(self.height / disc)+1


            for i in range(T):
                u = (2*np.pi)*i/(T-1)
                for j in range(H):
                    v = j/(H-1)

                    x = self.radius*np.cos(u)
                    y = self.radius*np.sin(u)
                    z = (-self.height/2 + v*self.height)
                    P = np.block([P, np.matrix([x,y,z]).transpose()])


            for i in range(R):
                v = self.radius * (i/(R-1))
                T = round(2 * np.pi * v / disc)
                for j in range(T):
                    u = (2*np.pi)*j/(T-1)

                    x = v * np.cos(u)
                    y = v * np.sin(u)
                    z = -self.height / 2
                    P = np.block([P, np.matrix([x, y, z]).transpose()])

                    x = v * np.cos(u)
                    y = v * np.sin(u)
                    z = self.height / 2
                    P = np.block([P, np.matrix([x, y, z]).transpose()])

            for i in range(np.shape(P)[1]):
                P[:,i] = self.htm[0:3,0:3]*P[:,i]+self.htm[0:3,-1]
                
            return PointCloud(points = P, color = self.color, size = disc/2)
        
        else:
            return PointCloud(points = obj_cpp.to_pointcloud(disc).points_gp, color = self.color, size = disc/2)
      
    # Compute distance to an object
    def compute_dist(self, obj,  p_init=None, tol=0.001, no_iter_max=20, h=0, eps = 0, mode='auto'):
        return Utils.compute_dist(self, obj, p_init, tol, no_iter_max, h, eps, mode)

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

            if abs(tpoint[0,0]) < self.width/2:
                x = tpoint[0,0]
                dx2 = 0
            else:
                x = self.width/2 if tpoint[0,0] > 0 else -self.width/2
                dx2 = (abs(tpoint[0,0]) - self.width/2)**2

            if abs(tpoint[1,0]) < self.depth/2:
                y = tpoint[1,0]
                dy2 = 0
            else:
                y = self.depth/2 if tpoint[1,0] > 0 else -self.depth/2
                dy2 = (abs(tpoint[1,0]) - self.depth/2)**2

            if abs(tpoint[2,0]) < self.height/2:
                z = tpoint[2,0]
                dz2 = 0
            else:
                z = self.height/2 if tpoint[2,0] > 0 else -self.height/2
                dz2 = (abs(tpoint[2,0]) - self.height/2)**2

            d = sqrt(dx2+dy2+dz2)

            return self._htm[0:3, 0:3] * np.matrix([[x], [y], [z]]) + self._htm[0:3, 3], d
        else:
            pr = obj_cpp.projection(np.matrix(point).reshape((3,1)), h, eps)
            return np.matrix(pr.proj).transpose(), pr.dist