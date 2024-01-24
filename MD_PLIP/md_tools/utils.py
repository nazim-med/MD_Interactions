import numpy as np
import pandas as pd
import warnings
import os
from pymol import cmd
from scipy.spatial.distance import euclidean
from plip.basic.supplemental import vecangle, projection
from openbabel import pybel
from rdkit import Chem

def rm_model_tag(top):
        with open(top) as f:
            lines = f.readlines()
        lines_2 = [line for line in lines if line.find("MODEL") == -1 and line.find("ENDMDL") == -1]

        with open(top, "w") as f:
            for item in lines_2:
                f.write(item)

        mol = [x for x in pybel.readfile("pdb",top)][0]
        mol.write("pdb",top, overwrite=True)

def add_chains(top):
    filetype = top.split(".")[-1]
    filename = top.split(".")[-2]

    mol = [x for x in pybel.readfile(filetype,top)][0]

    chain_num = 1
    for i, res in enumerate(mol.residues):
        if res.OBResidue.GetName() in ["SOL","HOH","NA","CL"]:
            break
        if res.OBResidue.GetNum() < mol.residues[i-1].OBResidue.GetNum() and i != 0:
            chain_num += 1
            res = res.OBResidue.SetChainNum(chain_num)
        else:
            res = res.OBResidue.SetChainNum(chain_num)
    mol.write("pdb","./{}.pdb".format(filename), overwrite=True)


def analyse_pi(plane_1, plane_2):
    COM_1 =  np.mean(plane_1, axis=0)
    COM_2 =  np.mean(plane_2, axis=0)

    normal_1 = np.cross(plane_1[0]-COM_1,plane_1[1]-COM_1)
    normal_2 = np.cross(plane_2[0]-COM_2,plane_2[1]-COM_2)

    normal_1 = normal_1 / np.linalg.norm(normal_1)
    normal_2 = normal_2 / np.linalg.norm(normal_2)

    proj_1 = projection(normal_1, COM_1, COM_2)
    proj_2 = projection(normal_2, COM_2, COM_1)

    offset = min(euclidean(proj_1, COM_1), euclidean(proj_2, COM_2))
    angle = vecangle(normal_1, normal_2)
    angle = min(angle, 180-angle if not 180 - angle < 0 else angle)

    return angle, offset

def int_fp_matrix(my_id, clust_dir, 
                    hydrophobic_df, hp_presence,
                    hbond_df, hb_presence,
                    pi_stacking_df, ps_presence,
                    pi_cation_df, pc_presence,
                    saltbridge_df, sb_presence,):
    
    interaction_names = []

    mol = [x for x in pybel.readfile("pdb",clust_dir+"/"+os.listdir(clust_dir)[0])][0]
    for res in mol.residues:
        if res.OBResidue.GetName() != my_id.split(":")[0]:
            resname = res.OBResidue.GetName()+f"{res.OBResidue.GetNum()}_"+res.OBResidue.GetChain()
            interaction_names += [resname+"_HPI",
                                resname+"_HB",
                                resname+"_PS",
                                resname+"_PC",
                                resname+"_SB",]
            
    interaction_df = pd.DataFrame(np.zeros((len(hp_presence),len(interaction_names))), columns=interaction_names)

    for i in range(len(hydrophobic_df)):
        row =  hydrophobic_df.iloc[i]
        int_name = "{}{}_{}_HPI".format(row["RESTYPE"],row["RESNR"],row["RESCHAIN"])
        if interaction_df.iloc[0][int_name]  == 0:
            row_ids = np.where(((hydrophobic_df["RESTYPE"] == row["RESTYPE"])*(hydrophobic_df["RESNR"] == row["RESNR"])*(hydrophobic_df["RESCHAIN"] == row["RESCHAIN"])) == True)[0]

            int_presence = hp_presence.iloc[:,row_ids]
            interaction_df[int_name] = (int_presence.sum(axis=1) > 0)

    for i in range(len(hbond_df)):
        row =  hbond_df.iloc[i]
        int_name = "{}{}_{}_HB".format(row["restype"],row["resnr"],row["reschain"])
        if interaction_df.iloc[0][int_name]  == 0:
            row_ids = np.where(((hbond_df["restype"] == row["restype"])*(hbond_df["resnr"] == row["resnr"])*(hbond_df["reschain"] == row["reschain"])) == True)[0]

            int_presence = hb_presence.iloc[:,row_ids]
            interaction_df[int_name] = (int_presence.sum(axis=1) > 0)

    for i in range(len(pi_stacking_df)):
        row =  pi_stacking_df.iloc[i]
        int_name = "{}{}_{}_PS".format(row["RESTYPE"],row["RESNR"],row["RESCHAIN"])
        if interaction_df.iloc[0][int_name]  == 0:
            row_ids = np.where(((pi_stacking_df["RESTYPE"] == row["RESTYPE"])*(pi_stacking_df["RESNR"] == row["RESNR"])*(pi_stacking_df["RESCHAIN"] == row["RESCHAIN"])) == True)[0]

            int_presence = ps_presence.iloc[:,row_ids]
            interaction_df[int_name] = (int_presence.sum(axis=1) > 0)

    for i in range(len(pi_cation_df)):
        row =  pi_cation_df.iloc[i]
        int_name = "{}{}_{}_PS".format(row["RESTYPE"],row["RESNR"],row["RESCHAIN"])
        if interaction_df.iloc[0][int_name]  == 0:
            row_ids = np.where(((pi_cation_df["RESTYPE"] == row["RESTYPE"])*(pi_cation_df["RESNR"] == row["RESNR"])*(pi_cation_df["RESCHAIN"] == row["RESCHAIN"])) == True)[0]

            int_presence = pc_presence.iloc[:,row_ids]
            interaction_df[int_name] = (int_presence.sum(axis=1) > 0)
    
    for i in range(len(saltbridge_df)):
        row =  saltbridge_df.iloc[i]
        int_name = "{}{}_{}_SB".format(row["RESTYPE"],row["RESNR"],row["RESCHAIN"])
        if interaction_df.iloc[0][int_name]  == 0:
            row_ids = np.where(((saltbridge_df["RESTYPE"] == row["RESTYPE"])*(saltbridge_df["RESNR"] == row["RESNR"])*(saltbridge_df["RESCHAIN"] == row["RESCHAIN"])) == True)[0]

            int_presence = sb_presence.iloc[:,row_ids]
            interaction_df[int_name] = (int_presence.sum(axis=1) > 0)

    return interaction_df.astype(int)
    
def int_fp_traj(int_df, thresh, fraction=False):
    int_fp = int_df.sum()/len(int_df)

    if fraction == True:
        return int_fp
    else:
        return (int_fp > thresh).astype(int)
    

###### Function from  matteoferla on Github https://gist.github.com/matteoferla/94eb8e4f8441ddfb458bfc45722469b8 ######

def set_to_neutral_pH(mol: Chem):
    """
    Not great, but does the job.
    
    * Protonates amines, but not aromatic bound amines.
    * Deprotonates carboxylic acid, phosphoric acid and sulfuric acid, without ruining esters.
    """
    protons_added = 0
    protons_removed = 0
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('[N;D1]')):
        atom = mol.GetAtomWithIdx(indices[0])
        if atom.GetNeighbors()[0].GetIsAromatic():
            continue # aniline
        atom.SetFormalCharge(1)
        protons_added += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)[O;D1]')):
        atom = mol.GetAtomWithIdx(indices[2])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('P(=O)[Oh1]')):
        atom = mol.GetAtomWithIdx(indices[2])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    for indices in mol.GetSubstructMatches(Chem.MolFromSmarts('S(=O)(=O)[Oh1]')):
        atom = mol.GetAtomWithIdx(indices[3])
        # benzoic acid pKa is low.
        atom.SetFormalCharge(-1)
        protons_removed += 1
    return (protons_added, protons_removed)