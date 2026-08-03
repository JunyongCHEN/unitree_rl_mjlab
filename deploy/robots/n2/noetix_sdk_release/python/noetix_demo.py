from typing import Tuple, overload
import numpy as np
import ctypes
import time
# import numpy.typing as npt
from enum import Enum
import sys
import os
sys.path.append("./")
import noetix_interface 
noetix = noetix_interface.Controllerbase()

def main():
 
    noetix.init(noetix_interface.ControlMode.DEFAULT)
    noetix.loadModel("walk","../ning")
    noetix.loadModel("run","../ning")
    noetix.start()
    while True:
        time.sleep(0.00001)
   
    
main()



