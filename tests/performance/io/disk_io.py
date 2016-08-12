import matplotlib.pyplot as plt
import subprocess
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

spikes_to_save = 100000

range_nr_rank = [1, 2, 4, 8, 16, 24, 32, 48, 64]
mean = []
std = []
for n_rank in range_nr_rank:
    # open the disk_io executable
    p1 = subprocess.Popen(["mpirun", "-n",str(n_rank),
                           os.path.join(current_script_dir, "disk_io.exe"),
                           str(spikes_to_save), str(10), "true" ,"true"],
                          stdout=subprocess.PIPE)
    
    #and grab the raw stats
    stats =  p1.communicate()[0]

    # convert into list
    stats = stats.split(",")

    mean.append(float(stats[1]))
    std.append(float(stats[2]))

    print ("performed test for n_rank= " + str(n_rank))

print (range_nr_rank)
print (mean)
print (std)

plt.errorbar(range_nr_rank, mean, yerr=std, fmt='-o')
plt.show()