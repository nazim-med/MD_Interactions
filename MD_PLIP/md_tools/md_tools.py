import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import os
from plip.basic import config
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
from scipy.spatial.distance import euclidean, rogerstanimoto, minkowski, cityblock

from .utils import analyse_pi, add_chains, rm_model_tag, int_fp_matrix, int_fp_traj
from .figures import save_pymol, plot_HPI, plot_HB, plot_PC, plot_PS, plot_SB, draw_interaction_Graph, plot_interaction_presence
import logging

config.NOHYDRO = True
config.DEFAULT_LOG_LEVEL = logging.ERROR

### Loosen detection thresholds to detect more interactions
### 
config.PISTACK_OFFSET_MAX = 4.0
config.PISTACK_ANG_DEV = 45
config.HBOND_DIST_MAX = 5.0
config.HYDROPH_DIST_MAX = 4.5


keys = (
    "hydrophobic",
    "hbond",
    "waterbridge",
    "saltbridge",
    "pistacking",
    "pication",
    "halogen",
    "metal",
)

hbkeys = [
    "resnr",
    "restype",
    "reschain",
    "resnr_l",
    "restype_l",
    "reschain_l",
    "sidechain",
    "protisdon",
    "d_orig_idx",
    "a_orig_idx",
    "h"
]

plip_cutoffs = {"HB_dist_max":0.41,
                "HB_ang_min":100,
                "HB_dist_max_plotting":0.25,
                "HB_ang_min_plotting":120,
                "PS_dist_max":0.55,
                "PS_offset_max":0.2,
                "PC_dist_max":0.6,
                "SB_dist_max":0.55,
                "HP_dist_max":0.4,
                }

prolif_cutoffs = {"HB_dist_max":0.35,
                "HB_ang_min":130,
                "HB_dist_max_plotting":0.25,
                "HB_ang_min_plotting":120,
                "PS_dist_max":0.65,
                "PS_offset_max":0.65,
                "PC_dist_max":0.45,
                "SB_dist_max":0.45,
                "HP_dist_max":0.45,
                }

def plip_analysis(pdb_file, bsid, outdir, save_files=False, pymol=False):
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)

    my_mol.analyze()
    my_interactions = my_mol.interaction_sets[bsid]

    if pymol & save_files:
        save_pymol(my_mol, bsid, outdir)

    bsr = BindingSiteReport(my_interactions)

    interactions = {
        k: [getattr(bsr, k + "_features")] + getattr(bsr, k + "_info")
        for k in keys
    }

    hp_df = pd.DataFrame(interactions["hydrophobic"][1:], columns=interactions["hydrophobic"][0])
    hb_df = []
    for hb in my_interactions.all_hbonds_pdon + my_interactions.all_hbonds_ldon:
        hb_interactions = []
        for k in hbkeys:
            hb_interactions.append(getattr(hb, k))

        hb_df.append(np.array(hb_interactions))
    if len(hb_df) != 0:
        hb_df = pd.DataFrame(np.stack(hb_df), columns=hbkeys)
        hb_df["h"] = [x.idx for x in hb_df["h"]]
    else:
        hb_df = pd.DataFrame()
    ps_df = pd.DataFrame(interactions["pistacking"][1:], columns=interactions["pistacking"][0])
    pc_df = pd.DataFrame(interactions["pication"][1:], columns=interactions["pication"][0])
    sb_df = pd.DataFrame(interactions["saltbridge"][1:], columns=interactions["saltbridge"][0])

    return my_mol, hp_df, hb_df, ps_df, pc_df, sb_df 

def load_traj(traj, top, stride=1, start=0, rmres=None):
    t = md.load(traj, top=top, stride=stride)
    t = t.remove_solvent(exclude=["CA"])
    if rmres != None:
        for res in rmres.split(" "):
            t = t.atom_slice(t.topology.select("not resname {}".format(res)))
    topology = t.topology

    ##### Truncate trajectory for analysis

    t = t[int(np.round(start/stride)):]

    return t, topology

def cluster_traj(t, min_clust=1, max_clust=100, thresh=0.25,  save_files=True, nsample=None, outdir="clusters"):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    distances = np.empty((t.n_frames, t.n_frames))
    for i in range(t.n_frames):
        distances[i] = md.rmsd(t, t, i)

    print('Max pairwise rmsd: %f nm' % np.max(distances))

    #Save cluster summary and bar chart

    if nsample is not None:
        clust = AgglomerativeClustering(n_clusters=nsample, metric="precomputed", linkage="average")
    else:
        clust = AgglomerativeClustering(distance_threshold=thresh, n_clusters=None, metric="precomputed", linkage="average")
    assignments = clust.fit_predict(distances)

    if (len(np.unique(assignments)) > max_clust or len(np.unique(assignments)) <= min_clust) and nsample is None:
        raise ValueError("Unacceptable number of clusters obtained ({}), should be between {}–{}. If this is an acceptable value, please increase --max_clust or decrease --min_clust arguments. If not, try adjusting the clustering threshold (--thresh)".format(len(np.unique(assignments)), max_clust, min_clust))
    
    if save_files:
        summary = pd.DataFrame([np.unique(assignments, return_counts=True)[0], np.unique(assignments, return_counts=True)[1]], index=["Cluster","population"]).T.to_string()
        with open("summary.txt", "w") as f:
            f.write(summary)

        XY = np.array([np.arange(len(assignments)), assignments]).T

        for i in range(len(np.unique(assignments))):
            index = np.where(assignments == i)[0]
            plt.bar(XY[index,0], 1, label=str(i))
        plt.legend(loc="upper right")
        plt.savefig("./clusters.png")


    #Calculate and save central structure of clusters

    centroids = []

    for j in range(len(np.unique(assignments))):
        distances_0 = distances[:,np.where(assignments == j)[0]]
        distances_0 = distances_0[np.where(assignments == j)[0]]
        beta = 1
        index = np.exp(-beta*distances_0 / distances_0.std()).sum(axis=1).argmax()
        centroids.append(index)

    centroids = [centroids[i]+min(np.where(assignments == i)[0]) for i in range(len(centroids))]

    for i in range(len(centroids)):
        filename = "{}/cluster_{}.pdb".format(outdir,i)
        t[centroids[i]].save_pdb(filename)
        add_chains(filename)
    return assignments, centroids

def random_sampling(t, nsample, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for i in range(nsample):
        frame = np.random.randint(0,len(t))
        t[frame].save_pdb("{}/frame_{}.pdb".format(outdir,frame))

    for filename in [x for x in os.listdir(outdir) if "pdb" in x]:
        # rm_model_tag("{}/{}".format(outdir,filename))
        #### Add chain identifiers to pdb for use
        add_chains("{}/{}".format(outdir,filename))

class analysis():
    '''
    Provides an extension of PLIP's interaction analysis to molecular dynamics trajectories. 
    '''
    class Hbond:
        def __init__(self):
            self.hbond_df = None
            self.hb_dists = None
            self.hb_angles = None
            self.hb_presence = None

    class Hydrophobic:
        def __init__(self):
            self.hydrophobic_df = None
            self.hp_dists = None
            self.hp_presence = None

    class PiStacking:
        def __init__(self):
            self.pi_stacking_df = None
            self.ps_dists = None
            self.ps_angles = None
            self.ps_offset = None
            self.ps_presence = None

    class PiCation:
        def __init__(self):
            self.pi_cation_df = None
            self.pc_dists = None
            self.pc_presence = None

    class Saltbridge:
        def __init__(self):
            self.saltbridge_df = None
            self.sb_dists = None
            self.sb_presence = None
    
    bsid = None
    dirname = None

    def __init__(self, t, topology, sample_dir="clusters", cutoffs="PLIP"):
        """
        Initialize an instance of the Analysis class.

        Parameters:
        - t (mdtraj.core.trajectory.Trajectory): Mdtraj trajectory object.
        - topology (mdtraj.core.topology.Topology): Mdtraj topology object - must correspond to the trajectory object.
        - sample_dir (str, optional): Description of the parameter. Defaults to "clusters".
        - cutoffs (str, optional): must belong to "PLIP" or "PROLIF". Defines interaction cutoffs for detection of interactions. More information given in [REFERENCES]. Defaults to "PLIP".
        """
        if cutoffs == "PLIP":
            self.cutoffs = plip_cutoffs
        elif cutoffs == "PROLIF":
            self.cutoffs = prolif_cutoffs
            config.PICATION_DIST_MAX = 4.5
            config.PISTACK_DIST_MAX = 6.5
            config.PISTACK_OFFSET_MAX = 100
            config.SALTBRIDGE_DIST_MAX = 4.5
            config.HBOND_DIST_MAX = 3.5
            config.HBOND_DON_ANGLE_MIN = 130
            config.HYDROPH_DIST_MAX = 4.5

        self.t = t
        self.topology = topology
        self.sample_dir = sample_dir

        tempmol = PDBComplex()
        tempmol.load_pdb(sample_dir+"/"+[x for x in os.listdir(sample_dir) if "pdb" in x][0])
        self.bsids = [x for x in str(tempmol).split("\n")[1:]]

        # Initialize instance variables for each analysis type
        self.hbond = self.Hbond()
        self.hydrophobic = self.Hydrophobic()
        self.pi_stacking = self.PiStacking()
        self.pi_cation = self.PiCation()
        self.saltbridge = self.Saltbridge()

    def analyse_bsid(self, bsid, save_files=True, pymol=True):
        """
        Perform analysis of the chosen binding site and generate interaction distances. Populates interaction subclasses of the Analysis Class.

        Parameters:
        - bsid (str): Binding site to perform PLIP analysis on to generate interactions.
        - save_files (bool, optional): Save distances and analysis data. Defaults to True.
        - pymol (bool, optional): If save_files is True, also saves pymol session files for each analysed sample. Can lengthen analysis time for a large number of samples. Defaults to True
        """

        dirname = "_".join(bsid.split(":"))

        self.bsid=bsid
        self.dirname = dirname

        #### Create directories ###
        if save_files:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            if not os.path.isdir(dirname+"/Interaction_analysis"):
                os.mkdir(dirname+"/Interaction_analysis")

        hydrophobic_df = pd.DataFrame()
        hbond_df = pd.DataFrame()
        pi_stacking_df = pd.DataFrame()
        pi_cation_df = pd.DataFrame()
        saltbridge_df = pd.DataFrame()   

        for filename in [x for x in os.listdir(self.sample_dir) if "pdb" in x]:
            if save_files:
                f_name = filename.split(".")[0]+"_plip"
                if f_name not in os.listdir(dirname):
                    os.mkdir(dirname+"/"+f_name)
            
            _, hp_df, hb_df, ps_df, pc_df, sb_df = plip_analysis("./{}/{}".format(self.sample_dir, filename), bsid, outdir= dirname+"/"+filename.split(".")[0]+"_plip", save_files=save_files, pymol=pymol)

            hydrophobic_df = pd.concat([hydrophobic_df, hp_df], axis=0).reset_index(drop=True)
            hbond_df = pd.concat([hbond_df, hb_df], axis=0).reset_index(drop=True)
            pi_stacking_df = pd.concat([pi_stacking_df, ps_df], axis=0).reset_index(drop=True)
            pi_cation_df = pd.concat([pi_cation_df, pc_df], axis=0).reset_index(drop=True)
            saltbridge_df = pd.concat([saltbridge_df, sb_df], axis=0).reset_index(drop=True)

            if save_files:
                if len(hp_df) > 0:
                    hp_df.to_csv(dirname+"/{}/{}_hp.csv".format(f_name, filename.split(".")[0]), index=False)
                if len(hb_df) > 0:
                    hb_df.to_csv(dirname+"/{}/{}_hb.csv".format(f_name, filename.split(".")[0]), index=False)
                if len(ps_df) > 0:
                    ps_df.to_csv(dirname+"/{}/{}_ps.csv".format(f_name, filename.split(".")[0]), index=False)
                if len(pc_df) > 0:
                    pc_df.to_csv(dirname+"/{}/{}_pc.csv".format(f_name, filename.split(".")[0]), index=False)
                if len(sb_df) > 0:
                    sb_df.to_csv(dirname+"/{}/{}_sb.csv".format(f_name, filename.split(".")[0]), index=False)


        self.hydrophobic.hydrophobic_df=hydrophobic_df.drop("DIST", axis=1).drop_duplicates(subset=["LIGCARBONIDX","PROTCARBONIDX"]).reset_index(drop=True)
        self.hbond.hbond_df=hbond_df.drop_duplicates(subset=["d_orig_idx","a_orig_idx"]).reset_index(drop=True)
        self.pi_stacking.pi_stacking_df=pi_stacking_df.drop(["CENTDIST","ANGLE","OFFSET","TYPE"], axis=1).drop_duplicates(subset=["PROT_IDX_LIST","LIG_IDX_LIST"]).reset_index(drop=True)
        self.pi_cation.pi_cation_df=pi_cation_df.drop("DIST", axis=1).drop_duplicates(subset=["PROT_IDX_LIST","LIG_IDX_LIST"]).reset_index(drop=True)
        self.saltbridge.saltbridge_df=saltbridge_df.drop("DIST", axis=1).drop_duplicates(subset=["PROT_IDX_LIST","LIG_IDX_LIST"]).reset_index(drop=True)


        if len(hydrophobic_df) != 0:
            columns = []
            for row in self.hydrophobic.hydrophobic_df.iterrows():
                columns += [str(self.topology.atom(row[1]["LIGCARBONIDX"]-1))+"_"+str(self.topology.atom(row[1]["PROTCARBONIDX"]-1))]

            self.hydrophobic.hp_dists = pd.DataFrame(md.compute_distances(self.t,self.hydrophobic.hydrophobic_df[["LIGCARBONIDX","PROTCARBONIDX"]].to_numpy()-1), columns=columns)
            self.hydrophobic.hp_presence = (self.hydrophobic.hp_dists < self.cutoffs["HP_dist_max"])
            self.hydrophobic.hydrophobic_df["fpresent"] = (self.hydrophobic.hp_presence.sum(axis=0)/len(self.hydrophobic.hp_presence)).round(3).values
            self.hydrophobic.hydrophobic_df=self.hydrophobic.hydrophobic_df.sort_values("fpresent", ascending=False).reset_index(drop=True)
            columns = pd.DataFrame(self.hydrophobic.hp_presence.sum()).sort_values(0, ascending=False).index
            self.hydrophobic.hp_dists = self.hydrophobic.hp_dists[columns]
            self.hydrophobic.hp_presence = self.hydrophobic.hp_presence[columns]

            self.hydrophobic.hydrophobic_df["fpresent"] = (self.hydrophobic.hp_presence.sum(axis=0)/len(self.hydrophobic.hp_presence)).round(4).values
            if save_files:
                self.hydrophobic.hp_dists.to_csv(dirname+"/Interaction_analysis/hp_dists.csv",index=False)
                self.hydrophobic.hydrophobic_df.to_csv(dirname+"/Interaction_analysis/hp_summary.csv", index=False)

        if len(hbond_df) != 0:
            columns = []
            for row in self.hbond.hbond_df.iterrows():
                columns += [str(self.topology.atom(row[1]["a_orig_idx"]-1))+"_"+str(self.topology.atom(row[1]["h"]-1))]

            self.hbond.hb_dists = pd.DataFrame(md.compute_distances(self.t,self.hbond.hbond_df[["a_orig_idx","h"]].to_numpy()-1), columns=columns)
            self.hbond.hb_angles = pd.DataFrame(np.degrees(md.compute_angles(self.t,self.hbond.hbond_df[["a_orig_idx","h","d_orig_idx"]].to_numpy()-1)), columns=columns)
            self.hbond.hb_presence = (self.hbond.hb_dists < self.cutoffs["HB_dist_max_plotting"]) & (self.hbond.hb_angles > self.cutoffs["HB_ang_min_plotting"])
            self.hbond.hbond_df["fpresent"] = (self.hbond.hb_presence.sum(axis=0)/len(self.hbond.hb_presence)).round(3).values
            self.hbond.hbond_df=self.hbond.hbond_df.sort_values("fpresent", ascending=False).reset_index(drop=True)
            columns = pd.DataFrame(self.hbond.hb_presence.sum()).sort_values(0, ascending=False).index
            self.hbond.hb_dists = self.hbond.hb_dists[columns]
            self.hbond.hb_angles = self.hbond.hb_angles[columns]
            self.hbond.hb_presence = self.hbond.hb_presence[columns]

            if save_files:
                self.hbond.hb_dists.to_csv(dirname+"/Interaction_analysis/hb_dists.csv", index=False)
                self.hbond.hb_angles.to_csv(dirname+"/Interaction_analysis/hb_angles.csv", index=False)
                self.hbond.hbond_df.to_csv(dirname+"/Interaction_analysis/hb_summary.csv", index=False)

        if len(pi_stacking_df) != 0:
            columns = []
            ps_dists = []
            ps_angles = []
            ps_offset = []
            for row in self.pi_stacking.pi_stacking_df.iterrows():
                int_name = row[1]["RESTYPE_LIG"]+"_"+row[1]["RESTYPE"]+str(row[1]["RESNR"])
                if int_name not in columns:
                    columns += [int_name]
                else:
                    columns += [int_name+"_{}".format(columns.count(int_name))]
                PROT_COM = self.t.xyz[:,np.array(row[1]["PROT_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                LIG_COM = self.t.xyz[:,np.array(row[1]["LIG_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                ps_dists += [np.array([euclidean(tup[0],tup[1]) for tup in zip(PROT_COM,LIG_COM)])]
                
                angle_temp = []
                offset_temp = []

                for frame in range(len(self.t.xyz)):
                    prot_coo = self.t.xyz[frame,np.array(row[1]["PROT_IDX_LIST"].split(",")).astype(int)-1,:]
                    lig_coo = self.t.xyz[frame,np.array(row[1]["LIG_IDX_LIST"].split(",")).astype(int)-1,:]

                    angle, offset = analyse_pi(lig_coo, prot_coo)

                    angle_temp += [angle]
                    offset_temp += [offset]
                
                ps_angles += [np.array(angle_temp)]
                ps_offset += [np.array(offset_temp)]

            self.pi_stacking.ps_dists = pd.DataFrame(ps_dists, index=columns).T
            self.pi_stacking.ps_angles = pd.DataFrame(ps_angles, index=columns).T
            self.pi_stacking.ps_offset = pd.DataFrame(ps_offset, index=columns).T
            self.pi_stacking.ps_presence = (self.pi_stacking.ps_dists < self.cutoffs["PS_dist_max"]) & (self.pi_stacking.ps_offset < self.cutoffs["PS_offset_max"])
            self.pi_stacking.pi_stacking_df["fpresent"] = (self.pi_stacking.ps_presence.sum(axis=0)/len(self.pi_stacking.ps_presence)).round(3).values
            self.pi_stacking.pi_stacking_df=self.pi_stacking.pi_stacking_df.sort_values("fpresent", ascending=False).reset_index(drop=True)
            columns = pd.DataFrame(self.pi_stacking.ps_presence.sum()).sort_values(0, ascending=False).index
            self.pi_stacking.ps_dists = self.pi_stacking.ps_dists[columns]
            self.pi_stacking.ps_angles = self.pi_stacking.ps_angles[columns]
            self.pi_stacking.ps_offset = self.pi_stacking.ps_offset[columns]
            self.pi_stacking.ps_presence = self.pi_stacking.ps_presence[columns]
            if save_files:
                self.pi_stacking.ps_dists.to_csv(dirname+"/Interaction_analysis/ps_dists.csv", index=False)
                self.pi_stacking.ps_angles.to_csv(dirname+"/Interaction_analysis/ps_angles.csv", index=False)
                self.pi_stacking.ps_offset.to_csv(dirname+"/Interaction_analysis/ps_offset.csv", index=False)
                self.pi_stacking.pi_stacking_df.to_csv(dirname+"/Interaction_analysis/ps_summary.csv", index=False)


        if len(pi_cation_df) != 0:
            columns = []
            pc_dists = []

            for row in self.pi_cation.pi_cation_df.iterrows():
                int_name = row[1]["RESTYPE_LIG"]+"_"+row[1]["RESTYPE"]+str(row[1]["RESNR"])
                if int_name not in columns:
                    columns += [int_name]
                else:
                    columns += [int_name+"_{}".format(columns.count(int_name))]
                PROT_COM = self.t.xyz[:,np.array(row[1]["PROT_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                LIG_COM = self.t.xyz[:,np.array(row[1]["LIG_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                pc_dists += [np.array([euclidean(tup[0],tup[1]) for tup in zip(PROT_COM,LIG_COM)])]

            self.pi_cation.pc_dists = pd.DataFrame(pc_dists, index=columns).T
            self.pi_cation.pc_presence = (self.pi_cation.pc_dists < self.cutoffs["PC_dist_max"])
            self.pi_cation.pi_cation_df["fpresent"] = (self.pi_cation.pc_presence.sum(axis=0)/len(self.pi_cation.pc_presence)).round(3).values
            self.pi_cation.pi_cation_df=self.pi_cation.pi_cation_df.sort_values("fpresent", ascending=False).reset_index(drop=True)
            columns = pd.DataFrame(self.pi_cation.pc_presence.sum()).sort_values(0, ascending=False).index
            self.pi_cation.pc_dists = self.pi_cation.pc_dists[columns]
            self.pi_cation.pc_presence = self.pi_cation.pc_presence[columns]

            if save_files:
                self.pi_cation.pc_dists.to_csv(dirname+"/Interaction_analysis/pc_dists.csv", index=False)
                self.pi_cation.pi_cation_df.to_csv(dirname+"/Interaction_analysis/pc_summary.csv", index=False)

        if len(saltbridge_df) != 0:
            columns = []
            sb_dists = []

            for row in self.saltbridge.saltbridge_df.iterrows():
                int_name = row[1]["RESTYPE_LIG"]+"_"+row[1]["RESTYPE"]+str(row[1]["RESNR"])
                if int_name not in columns:
                    columns += [int_name]
                else:
                    columns += [int_name+"_{}".format(columns.count(int_name))]
                PROT_COM = self.t.xyz[:,np.array(row[1]["PROT_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                LIG_COM = self.t.xyz[:,np.array(row[1]["LIG_IDX_LIST"].split(",")).astype(int)-1,:].mean(axis=1)
                sb_dists += [np.array([euclidean(tup[0],tup[1]) for tup in zip(PROT_COM,LIG_COM)])]

            self.saltbridge.sb_dists = pd.DataFrame(sb_dists, index=columns).T
            self.saltbridge.sb_presence = (self.saltbridge.sb_dists < self.cutoffs["SB_dist_max"])
            self.saltbridge.saltbridge_df["fpresent"] = (self.saltbridge.sb_presence.sum(axis=0)/len(self.saltbridge.sb_presence)).round(3).values
            self.saltbridge.saltbridge_df=self.saltbridge.saltbridge_df.sort_values("fpresent", ascending=False).reset_index(drop=True)
            columns = pd.DataFrame(self.saltbridge.sb_presence.sum()).sort_values(0, ascending=False).index
            self.saltbridge.sb_dists = self.saltbridge.sb_dists[columns]
            self.saltbridge.sb_presence = self.saltbridge.sb_presence[columns]

            if save_files:
                self.saltbridge.sb_dists.to_csv(dirname+"/Interaction_analysis/sb_dists.csv", index=False)
                self.saltbridge.saltbridge_df.to_csv(dirname+"/Interaction_analysis/sb_summary.csv", index=False)

    def plot_HPI(self, plot_thresh, save_files=True):
        if self.hydrophobic.hp_dists is None:
            return None
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
        
        plot_HPI(self.hydrophobic.hp_dists, self.dirname, plot_thresh, save_files)
                
    def plot_HB(self, plot_thresh, save_files=True):
        if self.hbond.hb_dists is None:
            return None
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
        
        plot_HB(self.hbond.hb_dists, self.hbond.hb_angles, self.dirname, plot_thresh, save_files, HB_dist_max= self.cutoffs["HB_dist_max_plotting"], HB_ang_min= self.cutoffs["HB_ang_min_plotting"])
                            
    def plot_PS(self, plot_thresh, save_files=True):
        if self.pi_stacking.ps_dists is None:
            return None
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
        
        plot_PS(self.pi_stacking.ps_dists, self.pi_stacking.ps_offset, self.pi_stacking.ps_angles, self.dirname, plot_thresh, save_files, PS_dist_max= self.cutoffs["PS_dist_max"], PS_offset_max= self.cutoffs["PS_offset_max"])

    def plot_PC(self, plot_thresh, save_files=True):
        if self.pi_cation.pc_dists is None:
            return None
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
        
        plot_PC(self.pi_cation.pc_dists, self.dirname, plot_thresh,  save_files, PC_dist_max= self.cutoffs["PC_dist_max"])

    def plot_SB(self, plot_thresh, save_files=True):
        if self.saltbridge.sb_dists is None:
            return None
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
        
        plot_SB(self.saltbridge.sb_dists, self.dirname, plot_thresh,  save_files, SB_dist_max= self.cutoffs["SB_dist_max"])

    def plot_interaction_presence(self, plot_thresh=0.3, figsize=(6,8), save_files=True):
        if save_files:
            outpath = self.dirname+"/plots"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)

        ifp = self.get_ifp_matrix(save_files=False)

        plot_interaction_presence(ifp, self.dirname, self.bsid.split(":")[0], figsize, plot_thresh,  save_files)

    def get_ifp_matrix(self, save_files):
        '''
        Generates a matrix of interactions per frame, returns a pandas dataframe.
        
        Parameters:
        - plot_thresh (float): plot only interactions existing in more than a given fraction of frames.
        - save_files (bool): save interaction fingerprint matrix
        '''
        int_df = int_fp_matrix(self.bsid, self.sample_dir, 
                    self.hydrophobic.hydrophobic_df, self.hydrophobic.hp_presence,
                    self.hbond.hbond_df, self.hbond.hb_presence,
                    self.pi_stacking.pi_stacking_df, self.pi_stacking.ps_presence,
                    self.pi_cation.pi_cation_df, self.pi_cation.pc_presence,
                    self.saltbridge.saltbridge_df, self.saltbridge.sb_presence)
        
        if save_files:
            outpath = self.dirname+"/Interaction_analysis"
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)   

            int_df.to_csv(self.dirname+"/Interaction_analysis/IFP_matrix.csv")

        return int_df

    def traj_ifp(self, save_files, thresh, fraction=True):
        '''
        Generates a matrix of interactions per frame, returns a pandas dataframe.
        
        Parameters:
        - plot_thresh (float): plot only interactions existing in more than a given fraction of frames.
        - save_files (bool): save interaction fingerprint as a csv file.
        - fraction (bool, optional): output the trajectory fingerprint with values as fractions.
        '''

        int_df = int_fp_matrix(self.bsid, self.sample_dir, 
                    self.hydrophobic.hydrophobic_df, self.hydrophobic.hp_presence,
                    self.hbond.hbond_df, self.hbond.hb_presence,
                    self.pi_stacking.pi_stacking_df, self.pi_stacking.ps_presence,
                    self.pi_cation.pi_cation_df, self.pi_cation.pc_presence,
                    self.saltbridge.saltbridge_df, self.saltbridge.sb_presence)
        traj_fp = int_fp_traj(int_df, thresh, fraction)

        if save_files:
            if not os.path.isdir(self.dirname+"/Interaction_analysis"):
                os.mkdir(self.dirname+"/Interaction_analysis")         
            int_df.to_csv(self.dirname+"/Interaction_analysis/IFP_traj.csv")

        return traj_fp
    
    def representative_frame(self, thresh, pymol=True, save_files=False, metric="euclidean", out_name="Representative"):


        if metric.lower() == "euclidean":
            dist_metric = euclidean
        elif metric.lower() == "tanimoto":
            dist_metric = rogerstanimoto
        elif metric.lower() == "minkowski":
            dist_metric = minkowski
        elif metric.lower() == "manhattan":
            dist_metric = cityblock

        int_df = int_fp_matrix(self.bsid, self.sample_dir, 
                    self.hydrophobic.hydrophobic_df, self.hydrophobic.hp_presence,
                    self.hbond.hbond_df, self.hbond.hb_presence,
                    self.pi_stacking.pi_stacking_df, self.pi_stacking.ps_presence,
                    self.pi_cation.pi_cation_df, self.pi_cation.pc_presence,
                    self.saltbridge.saltbridge_df, self.saltbridge.sb_presence)

        traj_fp = int_fp_traj(int_df, thresh, fraction=False)

        rep_frame = np.argmin([dist_metric(traj_fp, x) for x in int_df.to_numpy()])

        if save_files:
            outpath = "{}/{}".format(self.dirname, out_name)
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)

            filename = "frame_{}".format(rep_frame)

            self.t[rep_frame].save_pdb("{}/{}.pdb".format(outpath,filename))

            add_chains("{}/{}.pdb".format(outpath,filename))
                
            _, hp_df, hb_df, ps_df, pc_df, sb_df = plip_analysis("{}/{}.pdb".format(outpath,filename), self.bsid, outpath, save_files=save_files, pymol=pymol)

            if len(hp_df) > 0:
                hp_df.to_csv("./{}/{}_hp.csv".format(outpath,filename), index=False)
            if len(hb_df) > 0:
                hb_df.to_csv("./{}/{}_hb.csv".format(outpath,filename), index=False)
            if len(ps_df) > 0:
                ps_df.to_csv("./{}/{}_ps.csv".format(outpath,filename), index=False)
            if len(pc_df) > 0:
                pc_df.to_csv("./{}/{}_pc.csv".format(outpath,filename), index=False)
            if len(sb_df) > 0:
                sb_df.to_csv("./{}/{}_sb.csv".format(outpath,filename), index=False)
        return rep_frame    
    
    def plot_2D_interactions(self, plot_thresh=0.2, canvas_height=500, canvas_width=800, padding=40, save_files=False, out_name=None):

        if save_files:
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)     
            if not os.path.isdir(self.dirname+"/plots"):
                os.mkdir(self.dirname+"/plots")         
    
        draw_interaction_Graph(self, plot_thresh, canvas_height, canvas_width, padding, save_files, out_name=self.dirname+"/plots/{}".format(out_name))
        