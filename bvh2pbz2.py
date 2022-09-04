import sys, bz2, os, re, traceback,argparse
import _pickle as pkl
from typing import List
from lafan1 import extract
from multiprocessing import Pool

def find_all_bvh(dirname: str, verbose:bool=False) -> List[str]:
    """
    It takes a directory name and returns a list of all the BVH files in that directory
    
    :param dirname: The directory to search for BVH files
    :type dirname: str
    :param verbose: If True, prints out the directory it's searching in, defaults to False
    :type verbose: bool (optional)
    :return: A list of strings, each string is the path to a bvh file in the directory.
    """
    if not os.path.exists(dirname):
        raise FileNotFoundError("Directory not found: {}".format(dirname))

    if verbose:
        print("Searching for BVH files in {}".format(dirname))

    cur_dirname = os.path.dirname(__file__)
    return [os.path.join(cur_dirname, dirname, f) for f in os.listdir(dirname)]

class Converter:
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
    def __call__(self, bvh:str):
        anim = extract.read_bvh(bvh)
        animName = re.split(".*/(.+)\.bvh$", bvh)[1]
        with bz2.BZ2File(os.path.join(self.output_dir, animName+".pbz2"), 'w') as f:
            pkl.dump(anim, f)


def save_to_pbz2(bvh_files:List[str], output_dir:str="output", verbose:bool=False):
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception:
            traceback.print_exc()
            output_dir = "" # fallback to current folder
    if verbose:
        print("Dumping BVH files as Anim objects to {}".format(output_dir))
    num_cpus = os.cpu_count()
    if num_cpus is None:
        for bvh in bvh_files:
            Converter(output_dir)(bvh)
    else:
        with Pool(processes=num_cpus) as p:
            p.map(Converter(output_dir), bvh_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="directory where BVH are located")
    parser.add_argument('--verbose',  type=bool, nargs='?')

    args = parser.parse_args()
    dirname = args.path
    verbose = args.__contains__("verbose")
    try:
        save_to_pbz2(find_all_bvh(dirname, verbose), verbose=verbose)
    except Exception:
        traceback.print_exc()
        traceback.print_stack()
