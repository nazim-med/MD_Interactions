#!/usr/bin/env python3

from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os

from plip.basic import config
from MD_PLIP.md_tools.md_tools import random_sampling, cluster_traj, load_traj, analysis

config.PISTACK_OFFSET_MAX = 3.5  #### Detect more pi-stacking interactions for profiling
config.NOHYDRO = True

def parse_args():
        #Load in files for input

    my_parser = argparse.ArgumentParser(description='Cluster a trajectory. Assigns correct chains to gromacs output pdb and cleans pdb from mdtraj')
    my_parser.add_argument('--traj', action='store', type=str, required=True, help="Trajectory for clustering")
    my_parser.add_argument('--top', action='store', type=str, required=True, help="topology matching the trajectory")
    my_parser.add_argument('--stride', action='store',type=int, required=False, default=1, help='Load every nth frame')
    my_parser.add_argument('--start', action='store',type=int, required=False, default=0, help='Analyse interactions from frame n')
    my_parser.add_argument('--sampling', action='store',type=str, required=False, default="cluster", help='Sampling method for interaction analysis. "cluster" = agglomerative clustering, "random" = random sampling for simulation length.')
    my_parser.add_argument('--nsample', action='store',type=int, required=False, default=20, help='Number of samples for random sampling.')
    my_parser.add_argument('--thresh', action='store',type=float, required=False, default=0.2, help='Distance threshold for agglomerative clustering.')
    my_parser.add_argument('--max_clust', action='store',type=int, required=False, default=20, help='Maximum number of clusters from agglomerative clustering - exits script if too many clusters are found.')
    my_parser.add_argument('--min_clust', action='store',type=int, required=False, default=1, help='Minimum number of clusters from agglomerative clustering - exits script if too few clusters are found.')
    my_parser.add_argument('--pymol', action='store',type=bool, required=False, default=True, help='If true, will generate Pymol  with interactions for each cluster/frame. For large numbers of frames this will increase computation time.')
    my_parser.add_argument('--rmres', action='store',type=str, required=False, default=None, help='Residues to remove for analysis - e.g. lipid molecules in membranes')
    my_parser.add_argument('--plot_thresh', action='store',type=float, required=False, default=0.2, help='Plot interactions present in greater than n frames of the trajectory.')

    args = my_parser.parse_args()

        ### Checking arguments

    if not os.path.isfile(args.traj) or not os.path.isfile(args.top):
        raise FileNotFoundError("Trajectory or Topology file does not exist.")

    if args.sampling not in ["cluster","random"]:
        raise ValueError("Invalid sampling type. Expected one of: %s" % ["cluster","random"])

    return args

def main():
    args = parse_args()

    sample = True

    if not os.path.exists("clusters"):
        os.mkdir("clusters")
    elif len(os.listdir("clusters")) != 0 and ".pdb" in [x[-4:] for x in os.listdir("clusters")]:
        user_input = ""
        while True:
            user_input = input("Detected previous samples. Do you want to remove files and resample the trajectory (y) or analyse existing samples (n)? y/n: ")

            if user_input.lower() == "y":
                for x in os.listdir("clusters"):
                    os.remove("clusters/"+x)
                break
            
            elif user_input.lower() == "n":
                sample = False
                break
            else:
                continue

    print("Loading trajectory")

    t, topology = load_traj(args.traj, args.top, args.stride, args.start, args.rmres)

    # # Calculate distance matrix and cluster trajectories

    if sample:
        if args.sampling == "cluster":
            print("Sampling by agglomerative clustering.\n")
            cluster_traj(t, args.thresh, args.min_clust, args.max_clust, nsample=args.nsample)

        else:
            print("Sampling {} frames by random sampling.\n".format(args.nsample))
            random_sampling(t, args.nsample, "clusters")

    ###Edit PDB files so PLIP won't return an error

    analyser = analysis(t, topology, "clusters")
    bsids = analyser.bsids

    for my_id in [x for x in bsids if x.split(":")[0] not in ["ARN", "ASH", "GLH", "LYN", "HIE", "HIP"]]:
        user_input = ''

        while True:
            user_input = input('Do you want analyse object {}? y/n: '.format(my_id))

            if user_input.lower() in ['n','y']:
                break
            else:
                print('Do you want analyse object {}? y/n: '.format(my_id))    

        if user_input == "n": continue

        else:
            analyser.analyse_bsid(my_id, save_files=False, pymol=True)
            analyser.plot_HPI(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_HB(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_PS(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_PC(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_SB(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_interaction_presence(plot_thresh=args.plot_thresh, save_files=True)
            analyser.plot_2D_interactions(plot_thresh=args.plot_thresh, save_png=True, out_name="{}_interactions.png".format(my_id.split(":")[0]))
            analyser.traj_ifp(save_files=True, thresh=args.plot_thresh, fraction=True)
            analyser.representative_frame(thresh=0.3, pymol=True, save_files=True, metric="tanimoto")

    print("Done.")

if __name__ =="__main__":
    main()