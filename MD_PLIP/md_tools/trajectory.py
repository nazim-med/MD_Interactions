import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import os


def load_traj(args):
    t = md.load(args.traj, top=args.top, stride=args.stride)
    t = t.remove_solvent(exclude=["CA"])
    for res in args.rmres.split(" "):
        t = t.atom_slice(t.topology.select("not resname {}".format(res)))
    topology = t.topology

    ##### Truncate trajectory for analysis

    t = t[int(np.round(args.start/args.stride)):]

    return t, topology

def cluster_traj(t, args):
    if not os.path.exists("clusters"):
        os.mkdir("clusters")

    distances = np.empty((t.n_frames, t.n_frames))
    for i in range(t.n_frames):
        if i%100 == 0:
            print(str(i)+" of %i frames processed" % t.n_frames)
        distances[i] = md.rmsd(t, t, i)

    print('Max pairwise rmsd: %f nm' % np.max(distances))

    #Save cluster summary and bar chart

    clust = AgglomerativeClustering(distance_threshold=args.thresh, n_clusters=None, affinity="precomputed", linkage="average")
    assignments = clust.fit_predict(distances)

    if len(np.unique(assignments)) > args.max_clust or len(np.unique(assignments)) <= args.min_clust:
        raise ValueError("Unacceptable number of clusters obtained ({}), should be between {}–{}. If this is an acceptable value, please increase --max_clust or decrease --min_clust arguments. If not, try adjusting the clustering threshold (--thresh)".format(len(np.unique(assignments)), args.max_clust, args.min_clust))
    
    summary = pd.DataFrame([np.unique(assignments, return_counts=True)[0], np.unique(assignments, return_counts=True)[1]], index=["Cluster","population"]).T.to_string()
    with open("summary.txt", "w") as f:
        f.write(summary)

    XY = np.array([np.arange(len(assignments)), assignments]).T

    for i in range(len(np.unique(assignments))):
        index = np.where(assignments == i)[0]
        plt.bar(XY[index,0], 1, label=str(i))
    plt.legend(loc="upper right")
    plt.savefig("clusters.png")

    #Calculate and save central structure of clusters

    centroids = []

    for j in range(len(np.unique(assignments))):
        print("Calculating centroid of cluster ",str(j+1),"of ",str(len(np.unique(assignments))))
        distances_0 = distances[:,np.where(assignments == j)[0]]
        distances_0 = distances_0[np.where(assignments == j)[0]]
        beta = 1
        index = np.exp(-beta*distances_0 / distances_0.std()).sum(axis=1).argmax()
        centroids.append(index)

    centroids = [centroids[i]+min(np.where(assignments == i)[0]) for i in range(len(centroids))]

    for i in range(len(centroids)):
        filename = "clusters/cluster_"+str(i)+".pdb"
        t[centroids[i]].save_pdb(filename)

def random_sampling(t, args):
    if not os.path.exists("clusters"):
        os.mkdir("clusters")
    for i in range(args.nsample):
        frame = np.random.randint(0,len(t))
        t[frame].save_pdb("./clusters/frame_{}.pdb".format(frame))
