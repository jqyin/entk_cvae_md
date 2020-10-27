import os
import tables
from mpi4py import MPI
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from itertools import chain
from scipy.optimize import brute

def best_loop(nloops, nframes, nranks):
    return abs(nframes - nframes//(nranks*nloops)*(nranks*nloops))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
loop_range = slice(1000,2000,1)

def contact_maps_from_traj(pdb_file, traj_file, savefile, contact_cutoff=8.0):
    """
    Get contact map from trajectory.
    """
    
    mda_traj = mda.Universe(pdb_file, traj_file)
    traj_length = len(mda_traj.trajectory) 
    nloops = int(brute(best_loop, (loop_range,), args=(traj_length, size), finish=None))
    print("traj_length: %d  nloop: %d"%(traj_length, nloops))
    write_freq = nloops // 5
    ca = mda_traj.select_atoms('name CA')
    dist_shape = distances.self_distance_array(ca.positions).shape[0]
 
    if rank == 0:
        savefile = os.path.abspath(savefile)
        outfile = tables.open_file(savefile, 'w')
        atom = tables.Int8Atom()
        cm_table = outfile.create_earray(outfile.root, 'contact_maps', atom, shape=(0, dist_shape)) 
        print("dist_shape ", dist_shape)
    contact_matrices = []
    # workaround mpi4py 2^32 limit on number of objects
    # and ib memory size limit 
    for loop in range(nloops): 
        contact_matrices_loop = []
    
        nframes = traj_length//(size*nloops)
        start = (rank+loop*size)*nframes
        end = (rank+1+loop*size)*nframes 
        if loop == nloops -1 and rank == size - 1:
            end = traj_length
        print("loop %d rank %d start %d end %d"%(loop, rank, start, end))
        for frame in mda_traj.trajectory[start:end]:
            cm_matrix = (distances.self_distance_array(ca.positions) < contact_cutoff) * 1.0
            contact_matrices_loop.append(cm_matrix.astype('int8'))
        print("rank %d cm size %d"%(rank, len(contact_matrices_loop))) 
        contact_matrices_loop = comm.gather(contact_matrices_loop, root=0) 
        if rank == 0:
            contact_matrices.append(list(chain.from_iterable(contact_matrices_loop)))
            print("loop %d "%loop, len(contact_matrices_loop), len(contact_matrices_loop[0]))   
            if (loop+1) % write_freq == 0:
                contact_matrices = list(chain.from_iterable(contact_matrices))
                cm_table.append(contact_matrices)
                contact_matrices = []
        comm.Barrier()
    if rank == 0:
        if len(contact_matrices) > 0:
            contact_matrices = list(chain.from_iterable(contact_matrices))
            cm_table.append(contact_matrices)
        outfile.close() 

