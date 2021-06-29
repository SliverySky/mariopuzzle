import os,sys
os.chdir(sys.path[0])
import matplotlib.pyplot as plt
import json


def drawPointGraph(total):
    x, y = [], []
    for i in range(Iteration):
        for j in range(Lamda):
            x.append(i)
            y.append(score[i][j]['fit'] / total)
    plt.scatter(x, y, s=10, marker='x', label='Fitness')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.savefig("point.png")
    plt.show()


def get_max(iter, attri):
    return max([score[iter][j][attri] for j in range(Lamda)])


def get_sum(iter, attri):
    return sum([score[iter][j][attri] for j in range(Lamda)])


def drawPic2(total, isBest=False):
    x, y1, y2, y3, y4 = [], [], [], [], []
    for i in range(Iteration):
        x.append(i)
        if (isBest):
            y1.append(get_max(i, 'fit') / total)
            y2.append(5 * get_max(i, 'wrong') / total)
            y3.append(3 * get_max(i, 'replace') / total)
            y4.append(get_max(i, 'value') / total)
        else:
            y1.append(get_sum(i, 'fit') / Lamda / total)
            y2.append(5 * get_sum(i, 'wrong') / Lamda / total)
            y3.append(3 * get_sum(i, 'replace') / Lamda / total)
            y4.append(get_sum(i, 'value') / Lamda / total)
    plt.plot(x, y1, color='r', linestyle='--', label='Fitness(R)')
    plt.plot(x, y2, 'b--', label='5*#Wrong(R)')
    plt.plot(x, y3, 'g--', label='3*#Repalce(R)')
    plt.plot(x, y4, 'y--', label='UV(R)')
    plt.title("Best" if isBest else "Mean")
    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("value")
    plt.savefig("best.png" if isBest else "mean.png")
    plt.show()


if __name__ == '__main__':
    global Iteration, Lamda
    with open("result//json//data.json") as f:
        score = json.load(f)
    Iteration = len(score)
    Lamda = len(score[0])
    drawPointGraph(1)
    drawPic2(1, True)
    drawPic2(1, False)
    print('figures are saved in GA folder.')
