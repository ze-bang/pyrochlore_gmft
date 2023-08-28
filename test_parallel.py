from mpi4py import MPI
import numpy as np
import sys
#
# comm = MPI.COMM_WORLD

client_script = 'test_parallel.py'
comm = MPI.COMM_SELF.Spawn(sys.executable, args=[client_script], maxprocs=5)

size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

print(size)

a = np.linspace(0,10, 1000)

size = 1000/size

left = int(rank*size)
right = int((rank+1)*size)

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
