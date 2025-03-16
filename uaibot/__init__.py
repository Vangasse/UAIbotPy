__version__ = '0.0.21'

import os
import sys

# Get the directory of the current __init__.py file
current_dir = os.path.dirname(__file__)

# Add the directory of __init__.py to sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)



try:
    import uaibot_cpp_bind as ub_cpp
    os.environ['CPP_SO_FOUND'] = '1'
except ImportError:
    print("INFO: CPP .so not found! Only python mode is available!")
    os.environ['CPP_SO_FOUND'] = '0'

from robot import *
from demo import *
from utils import *
from simulation import *
from graphics import *
from simobjects import *

import numpy as np
np.set_printoptions(
    precision=4,   
    linewidth=80,  
    suppress=True  
)




