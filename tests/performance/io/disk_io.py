import matplotlib.pyplot as plt
import subprocess
import os



current_script_dir = os.path.dirname(os.path.abspath(__file__))

spikes_to_save = 1000000

print ( "Simple performance runner for spike output to file. \n" +
        str(spikes_to_save) + " spikes will be written to a file and the duration of this \n" +
        "operation measured for different number of ranks\n" )


range_nr_rank = [1, 2, 4, 8, 16, 24, 32, 48, 64]
mean = []
std = []
min = []
max = []
for n_rank in range_nr_rank:
    # open the disk_io executable
    p1 = subprocess.Popen(["mpirun", "-n",str(n_rank),
                           os.path.join(current_script_dir, "disk_io.exe"),
                           str(spikes_to_save), str(10), "true"],
                          stdout=subprocess.PIPE)
    
    #and grab the raw stats
    stats =  p1.communicate()[0]

    # convert into list
    stats = stats.split(",")

    mean.append(float(stats[1]))
    std.append(float(stats[2]))
    min.append(float(stats[3]))
    max.append(float(stats[4]))

    print ("performed test for n_rank= " + str(n_rank))

print (range_nr_rank)
print (mean)
print (std)

plt.errorbar(range_nr_rank, mean, yerr=std, fmt='-o', label="mean (std)")
plt.errorbar(range_nr_rank, min, fmt='-', label="min")
plt.errorbar(range_nr_rank, max, fmt='-', label="max")
plt.legend()
plt.show()
