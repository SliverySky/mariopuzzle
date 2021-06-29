import matplotlib.pyplot as plt
import csv
import numpy as np
names = ["3", "9", "10", "11", "12", "4", "5", "8"]
symbol = ["s=14,n=2(h=14,w=28)", "s=14,n=4", "s=14,n=6", "s=28,n=1", "s=1,n=28", "h=14,w=14", "h=14,w=7", "with playability test"]
for i in range(len(names)):
    exp = names[i]
    if exp == "8":  exp = "exp3_3_a"
    else:exp = "experiment" + exp
    with open('logs/'+exp+'/result.csv') as f:
        f_csv = csv.reader(f)
        x, y, std = [], [] ,[]
        for row in f_csv:
            x.append(float(row[0]) )
            y.append(float(row[1]))
            std.append(float(row[2]))
        y, std = np.array(y), np.array(std)
        plt.plot(x, y)
        #plt.plot(x, z, c="r")
        #plt.fill_between(x, y-std, y+std, facecolor="yellow", alpha=0.5)
    plt.legend([symbol[i]], loc="lower right")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    #plt.title("Training graph for agents with different reward function")
    plt.savefig("train_graph"+str(names[i]))
    plt.clf()