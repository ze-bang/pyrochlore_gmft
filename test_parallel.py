from mpi4py import MPI
import numpy as np
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size() # new: gives number of ranks in comm
# rank = comm.Get_rank()
#
# numDataPerRank = 10
# data = None
# if rank == 0:
#     data = np.linspace(1,size*numDataPerRank,numDataPerRank*size)
#     # when size=4 (using -n 4), data = [1.0:40.0]
#
# recvbuf = np.empty(numDataPerRank, dtype='d') # allocate space for recvbuf
# comm.Scatter(data, recvbuf, root=0)
#
# print('Rank: ',rank, ', recvbuf received: ',recvbuf)

omega = 0
tempE = np.array([1,2,3])
tempQ = np.array([1,2,3])

gauss = omega - np.einsum('i,j-> ij', tempE, tempQ)

print(gauss)