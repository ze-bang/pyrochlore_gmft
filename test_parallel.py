from mpi4py import MPI
import numpy as np
import sys
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size() # new: gives number of ranks in comm
# rank = comm.Get_rank()

# # print(size)

# a = np.linspace(0,10, 1000)

# n = 1000/size

# left = int(rank*n)
# right = int((rank+1)*n)

# for i in range(left, right):
#     print(i)


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

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(100, dtype='i') + rank
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 100], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)
# if rank == 0:
#     for i in range(size):
#         assert np.allclose(recvbuf[i,:], i)

MPI.Finalize()

print(recvbuf)