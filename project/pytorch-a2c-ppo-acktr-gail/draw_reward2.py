import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.interpolate import make_interp_spline
names = ["26"]
color_set=['#ff000a', '#0000ff', '#00ff00'] #['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585', '#939393', '#c4c4c4']
for i in range(len(names)):
    exp = names[i]
    exp = "experiment" + exp
    with open('logs/'+exp+'/result.csv') as f:
        f_csv = csv.reader(f)
        x, y, std, novelty,variance, p= [], [] ,[], [], [], []
        first = True
        for row in f_csv:
            if first:
                first = False
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
            novelty.append(float(row[3]))
            variance.append(float(row[4]))
            p.append( (float(row[1])-float(row[3])-float(row[4])) )
        x, novelty, variance, p =np.array(x), np.array(novelty), np.array(variance), np.array(p)
        xnew = np.linspace(min(x), max(x),300)
        #print(p.shape,x.shape)
        p_new = make_interp_spline(x,p)(xnew)
        n_new = make_interp_spline(x,novelty)(xnew)
        v_new = make_interp_spline(x,variance)(xnew)
        plt.plot(xnew, p_new,color = color_set[0])#, marker='+')
        plt.plot(xnew,n_new, color = color_set[1])#,marker='o')
        plt.plot(xnew,v_new, color = color_set[2])#,marker='x')
        #plt.plot(x, z, c="r")
        #plt.fill_between(x, y-std, y+std, facecolor="yellow", alpha=0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
plt.legend(["Rew P","Rew N","Rew V"])
#plt.title("Training graph for agents with different reward function")
plt.savefig("reward2")
plt.clf()
