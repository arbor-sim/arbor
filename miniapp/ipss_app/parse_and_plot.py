import matplotlib.pyplot as plt
import collections
import math

default_spike_file_path = "./spikes.gdf"




def parse_spike_file(path):
    """ Simple spike file parsing function. Assumes no errors and might fail 
    silently on errors"""

    spikes_per_cell = collections.defaultdict(list)

    with open(path , "r") as f:
        for line in f:
            tokens = line.split(",")
            try:
                gid = int(tokens[0].strip())
                time = float(tokens[1].strip())
            except:
                print tokens[0], ", ", tokens[1]
                exit()

            spikes_per_cell[gid].append(time)

    return spikes_per_cell


def binning(spikes_per_cell):
    bins=[0]*1000

    for gid, spikes in spikes_per_cell.items():
        for spike in spikes:
            bin = int(math.floor(spike))


            bins[bin] += 1
    return bins  

        




def main(path):
    spikes_per_cell = parse_spike_file(path)

    bins = binning(spikes_per_cell)
    plt.hist([idx for idx in range(len(bins))], len(bins), weights = bins)

    plt.show()









if __name__ == "__main__":
    main(default_spike_file_path)



