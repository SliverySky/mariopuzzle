import matplotlib.pyplot as plt
import csv
import numpy as np
names = ["13", "20","21", "22"]
color_set=['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585', '#939393', '#c4c4c4']
for i in range(len(names)):
    exp = names[i]
    exp = "experiment" + exp
    with open('logs/'+exp+'/result.csv') as f:
        f_csv = csv.reader(f)
        x, y, std, novelty = [], [] ,[], []
        first = True
        for row in f_csv:
            if first:
                first = False
                continue
            x.append(float(row[0]) )
            y.append(float(row[3]))
            std.append(float(row[2]))
            if i==0:novelty.append(float(row[4]))
            else: novelty.append(float(row[4])/0.3)
        y, std = np.array(y), np.array(std)
        plt.plot(x, y, color = color_set[i*2])
        plt.plot(x,novelty, color = color_set[i*2+1])
        #plt.plot(x, z, c="r")
        #plt.fill_between(x, y-std, y+std, facecolor="yellow", alpha=0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
plt.legend(["Similarity_part","Novel_part(Not be rewarded)","Similarity_part","Novel_part","Similarity_part","Novel_part(use archive)","add all","add all"])
#plt.title("Training graph for agents with different reward function")
plt.savefig("archive_analyze_C")
plt.clf()