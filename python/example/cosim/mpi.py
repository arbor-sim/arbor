from mpi4py import MPI

world = MPI.COMM_WORLD

color = None
leader = None
if world.rank == 0:
    color = 0
    leader = 1
else:
    color = 1
    leader = 0

group = world.Split(color)
inter = group.Create_intercomm(0, world, leader, 42)

if __name__ == '__main__':
    print(f"{world.rank:2d}/{world.size:2d} {group.rank:2d}/{group.size:2d}")

    if color == 0:
        print(f"[NMM] {world.rank:2d}/{world.size:2d}")
    else:
        print(f"[ARB] {world.rank:2d}/{world.size:2d}")
