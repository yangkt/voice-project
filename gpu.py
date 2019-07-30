import pycuda.driver as cuda
import torch

class GPU():
    
    def __init__(self):
        # the following functions test GPU availability    
        torch.cuda.is_available()
        cuda.init()
    
    @staticmethod
    def GPUCount():
        print('Number of GPU(s):' + cuda.Device.count())                # number of GPU(s)
    
    @staticmethod
    def GPUName():
        id = torch.cuda.current_device()   # Get Id of default device
        cuda.Device(id).name()             # Get the name for the device "Id"
    
    @staticmethod
    def GPUMem():
        print('GPU memory allocated:' + torch.cuda.memory_allocated())  # the amount of GPU memory allocated
        print('GPU memory cached:' + torch.cuda.memory_cached())    # the amount of GPU memory cached
    
    @staticmethod
    def GPUClear():
        torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.
        

