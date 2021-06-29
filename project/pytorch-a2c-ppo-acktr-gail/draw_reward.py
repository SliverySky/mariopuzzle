import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.interpolate import make_interp_spline
names = ["7"]
color_set=['#ff000a', '#0000ff', '#00ff00','#939393'] #['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585', '#939393', '#c4c4c4']
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
#plt.tight_layout()

fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
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
            variance.append(float(row[2]))
            p.append(float(row[4]))
        x, novelty, variance, p =np.array(x), np.array(novelty), np.array(variance), np.array(p)
        y = np.array(y)
        x=x
        xnew = np.linspace(min(x), max(x),300)
        #print(p.shape,x.shape)
        y_new = make_interp_spline(x,y)(xnew)
        p_new = make_interp_spline(x,p)(xnew)
        n_new = make_interp_spline(x,novelty)(xnew)
        v_new = make_interp_spline(x,variance)(xnew)
        plt.xlabel("Epoch",font1)
        plt.ylabel("Reward", font1)
        ap_val=0.8
        plt.plot(x, y,color = color_set[3], alpha=ap_val)
        plt.plot(x, variance, color = color_set[1], alpha=ap_val)#,marker='x')
        plt.plot(x, novelty, color = color_set[2], alpha=ap_val)#,marker='o')
        plt.plot(x, p,color = color_set[0], alpha=ap_val)#, marker='+')
        #plt.plot(x, z, c="r")
        #plt.fill_between(x, y-std, y+std, facecolor="yellow", alpha=0.5)
#plt.legend(["$R_{DNP}$","$R_P$","$R_D$","$R_N$"])
plt.legend(["$\sum (F+H+P)$", "$\sum F$","$\sum H$","$\sum P$"], prop=font2, frameon=False)
#plt.legend(["N"], prop=font1)
plt.tick_params(labelsize=15)
#plt.title("Training graph for agents with different reward function")
plt.savefig("graph/reward")
plt.clf()
