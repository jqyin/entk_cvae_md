import os
import tables
from mpi4py import MPI
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from itertools import chain

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
nloops = 10

def contact_maps_from_traj(pdb_file, traj_file, contact_cutoff=8.0, savefile=None):
    """
    Get contact map from trajectory.
    """
    
    mda_traj = mda.Universe(pdb_file, traj_file)
    traj_length = len(mda_traj.trajectory) 
    ca = mda_traj.select_atoms('name CA')
    
    if savefile and rank == 0:
        savefile = os.path.abspath(savefile)
        outfile = tables.open_file(savefile, 'w')
        atom = tables.Int8Atom()
        cm_table = outfile.create_earray(outfile.root, 'contact_maps', atom, shape=(traj_length, 0)) 

    contact_matrices = []
    # workaround mpi4py 2^32 limit on number of objects
    # and ib memory size limit 
    for loop in range(nloops): 
        contact_matrices_loop = []
    
        nframes = mda_traj.trajectory.n_frames//(size*nloops)
        start = (rank+loop*size)*nframes
        end = (rank+1+loop*size)*nframes 
        if loop == nloops -1 and rank == size - 1:
            end = mda_traj.trajectory.n_frames
        print("loop %d rank %d start %d end %d"%(loop, rank, start, end))
        for frame in mda_traj.trajectory[start:end]:
            cm_matrix = (distances.self_distance_array(ca.positions) < contact_cutoff) * 1.0
            contact_matrices_loop.append(cm_matrix)
        print("rank %d cm size %d"%(rank, len(contact_matrices_loop))) 
        contact_matrices_loop = comm.gather(contact_matrices_loop, root=0) 
        if rank == 0:
            contact_matrices.append(list(chain.from_iterable(contact_matrices_loop)))
            print("loop %d "%loop, len(contact_matrices_loop), len(contact_matrices_loop[0]))   
        comm.Barrier()
    if savefile and rank == 0:
        contact_matrices = list(chain.from_iterable(contact_matrices)) 
        print("All: ", len(contact_matrices))   
        cm_table.append(contact_matrices)
        outfile.close() 

    return contact_matrices
