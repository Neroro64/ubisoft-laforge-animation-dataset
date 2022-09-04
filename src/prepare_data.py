# %%
# Scientific computing 
import numpy as np
import pandas as pd
from scipy import signal

# Plotting 
import plotly.express as ex
import plotly.graph_objects as go

# Python utilities
import os,bz2,re,traceback
import _pickle as pkl
import zipfile
from multiprocessing import Pool

# Laforge animation dataset
from lafan1.extract import Anim,read_bvh
from bvh2pbz2 import *

# %%
DATA_DIR = "data"
MOTION_ZIP = "lafan1/lafan1.zip"
DATASET_DIR = "dataset"
FRAME_TIME = 0.03333
FRAMES = 7840

# %%
# Unzip and extract the BVH motions files
if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    with zipfile.ZipFile(MOTION_ZIP, "r") as zip_ref:
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        zip_ref.extractall(DATA_DIR)

# %%
def read_pbz2(filename:str) -> Anim:
    '''load Bzip2 compressed binary file, uncompressed and unpickle it to return the original object'''
    if not os.path.exists(filename):
        return None

    with bz2.BZ2File(filename, 'rb') as f:
        data = pkl.load(f)
    return data 

def write_pbz2(filename:str, data:object, dirname:str=DATASET_DIR) -> Exception:
    '''dump an object as Bzip2 compressed binary file'''
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with bz2.BZ2File(os.path.join(dirname, filename + ".pbz2"), 'w') as file_ref:
            pkl.dump(data, file_ref)
        return None
    except Exception as exp:
        return exp

class DataPreprocess:
    '''
    a class for processing a raw data file into a data sample for training / validating.
    It does
        - reads a clip file and extract the hips positions
        - computes velocity, acceleration, speed, phase information based on speed.
        - stacks the feature vectors into a Fx12 matrix where F is total frames.
        - dumps the matrix to compressed binary file on disk
    
    One frame contains:
        - P     : R^3   : [x, y, z] 
        - V     : R^3   : [dx/dt, dy/dt, dz/dt]
        - A     : R^3   : [dVx/dt, dVy/dt, dVz/dt]
        - Phase : R^2   : [cos(S), sin(S)], where S = norm(sum(V^2))
    
    Parameter:
        - output_dir (str): the directory where the binary files will be written to.
        - write (bool): if not set, the processed data will be returned instead
    Returns:
        - None
    '''
    def __init__(self, output_dir:str=DATASET_DIR, write:bool=True):
        self.output_dir = output_dir
        self.write = write

    def __call__(self, bvh:str):
        '''
        Parameter:
            - bvh (str): BVH file path
        '''
        pose_data = extract.read_bvh(bvh)

        pos = pose_data.pos[:, 0, :] # only consider hip joint
        vel = np.diff(pos, prepend=0) / FRAME_TIME
        acc = np.diff(vel, prepend=0) / FRAME_TIME

        pos = self.normalize(pos[:-30])             # discard last 30 frames of data
        vel = self.normalize(vel[:-30])             
        acc = self.normalize(acc[30:])              # offset acceleration data by 30 frames, such that acc[30] is the input signal at t=0 
        hip_to_ground = pos[:,1]    # just using the y values from positions

        # Compute phases
        spd = np.sum(vel**2,axis=1)
        spd = spd / np.std(spd,keepdims=True)
        b,a = signal.butter(5,0.005,btype="low",analog=False)
        spd_filtered = signal.filtfilt(b,a,spd)     # apply Butterworth-filter to smooth out the response.
        phase_sin = np.sin(spd_filtered * 2*np.pi - np.pi/2)
        phase_cos = np.cos(spd_filtered * 2*np.pi)
        phase = np.column_stack((phase_cos, phase_sin))

        # stack the feature vectors together
        data = np.column_stack((pos,vel,acc,hip_to_ground,phase))

        filename = re.split(".*/(.+)\.bvh$", bvh)[1]
        if self.write:
            exp = write_pbz2(filename, data, self.output_dir)
            # Saves exception to be printed at the end
            if exp is not None:
                print(exp)
        else:
            return {"data":data, "spd":spd_filtered, "phase":phase, "filename":filename}

    def normalize(self, vec:np.ndarray) -> np.ndarray:
        '''normalizes the vector / matrix to a normal distribution N(0,1) '''
        std = np.std(vec, keepdims=True)    # standard deviation of the values per axis (x,y,z)
        std[std < 1e-6] = 1                 # ignoring small deviations
        return (vec - np.mean(vec, keepdims=True)) / std # ( X - E[X] ) / \sigma_x

def process_bvh(bvh_files:List[str], output_dir:str=DATASET_DIR, verbose:bool=False):
    '''Given a list of BVH files, process them and extract the features to compressed binary format'''
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception:
            traceback.print_exc()
            output_dir = "" # fallback to current folder
    if verbose:
        print("Extracing features from BVH files to {}".format(output_dir))
    
    num_cpus = os.cpu_count()
    if num_cpus is None:
        preprocessor = DataPreprocess(output_dir, write=True,)
        for bvh in bvh_files:
            preprocessor(bvh)
    else:
        # Parallel processing
        with Pool(processes=num_cpus) as p:
            p.map(DataPreprocess(output_dir, write=True), bvh_files)
