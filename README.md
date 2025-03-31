# MD-PLIP - An extension of the popular Protein Ligand Interaction Profiler (PLIP)

Aims of this project:
* Add MD analysis functionality to the validated protein interaction profiler [PLIP](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index) using their Python API and [MDtraj](https://github.com/mdtraj/mdtraj)
* Automatically detect stable interaction and generate useful plots to monitor interactions over time
* Generate 2D and 3D representations of interaction structures
* Produce a command line interface to MD analysis to automatically carry out the most useful analysis

Implementation/ Workflow:
1. Trajectory i/o
2. Sampling of frames by random sampling or clustering
3. Analysis of non-covalent interaction by PLIP
4. Monitoring of detected interactions over the course of the simulation
5. Generation of useful plots

## Installation
1. Create environment and install dependencies
```
conda create -n MD_PLIP -c conda-forge openbabel plip pymol-open-source matplotlib cairo rdkit scikit-learn
```
2. Clone this repo
```
git clone https://github.com/nazim-med/MD_Interactions.git
```
3. Install md_plip (tested on windows and Ubuntu on WSL2)
```
pip install .
```

## Command-Line Interface
Running MD_PLIP in commandline can be as simple as - 
```
md_plip --top "foo.pdb" --traj "bar.xtc"
```
Through using mdtraj, a variety of topology and trajectory file formats are available.

Output of ```md_plip -h```:
```
usage: md_plip [-h] --traj TRAJ --top TOP [--stride STRIDE] [--start START] [--sampling SAMPLING] [--nsample NSAMPLE] [--thresh THRESH]
               [--max_clust MAX_CLUST] [--min_clust MIN_CLUST] [--pymol PYMOL] [--rmres RMRES] [--plot_thresh PLOT_THRESH]

Cluster a trajectory. Assigns correct chains to gromacs output pdb and cleans pdb from mdtraj

optional arguments:
  -h, --help            show this help message and exit
  --traj TRAJ           Trajectory for clustering
  --top TOP             topology matching the trajectory
  --stride STRIDE       Load every nth frame
  --start START         Analyse interactions from frame n
  --sampling SAMPLING   Sampling method for interaction analysis. "cluster" = agglomerative clustering, "random" = random sampling for
                        simulation length.
  --nsample NSAMPLE     Number of samples for random sampling.
  --thresh THRESH       Distance threshold for agglomerative clustering.
  --max_clust MAX_CLUST
                        Maximum number of clusters from agglomerative clustering - exits script if too many clusters are found.
  --min_clust MIN_CLUST
                        Minimum number of clusters from agglomerative clustering - exits script if too few clusters are found.
  --pymol PYMOL         If true, will generate Pymol with interactions for each cluster/frame. For large numbers of frames this will increase
                        computation time.
  --rmres RMRES         Residues to remove for analysis - e.g. lipid molecules in membranes
  --plot_thresh PLOT_THRESH
                        Plot interactions present in greater than n frames of the trajectory.
```

## Python Module

MD_plip is available as an importable python module. This includes functions to load the trajectory and sample frames for interaction analysis. The analysis functions are within the analysis class. 

An example notebook is provided in this repo. Simple usage is shown below.

```
from MD_PLIP.md_tools.md_tools import load_traj, cluster_traj, analysis

### Load in and cluster Trajectory

t, topology = load_traj(top="topology.gro", traj="trajectory.xtc", start=25, rmres="POPC")
cluster_traj(t, nsample=50, save_files=False, outdir="../MD_dir/clusters")

### Perform Analysis

analyser = analysis(t, topology, "../MD_dir/clusters")
bsid = analyser.bsids[0]
analyser.analyse_bsid(bsid, save_files=False)
```

## To-do

- Benchmarking
- Make installable by pip
- Improve 2D interaction plot label positioning
