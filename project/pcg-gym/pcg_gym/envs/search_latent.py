import numpy as np
class Searcher():
    def __init__(self,generator):
        self.generator = generator
    def find_latent(self,lv):
        self.lv = lv
        return self.search(32, -1, 1, self.evaluate)
    def evaluate(self,v):
        lv = self.generator.generate(v)
        return np.sum(np.where(lv==self.lv, 1, 0))
    def search(self,n, lb, ub, objective):
        mu = 100
        q = 10
        tot_iter = 200
        eta = np.ones((mu,n))*3
        new_eta = np.ones((mu,n))*3
        population = np.random.rand(mu,n) * (ub-lb) + lb
        new_population = np.random.rand(mu,n) * (ub-lb) + lb
        best = -100000
        bestx = np.zeros(n)
        fitness = np.zeros(2*mu)
        new_fitness = np.zeros(2*mu)
        for i in range(mu):
            fitness[i] = objective(population[i])
        for iter in range(tot_iter):
            #print(population)
            rate = self.get_rate(iter, tot_iter)
            children, children_eta = self.mutation(mu, n, lb, ub, population, eta, rate)
            win = np.zeros(2*mu)
            for i in range(mu):
                fitness[mu+i] = objective(children[i])
            for i in range(2*mu):
                for k in range(q):
                    s = int(np.floor(np.random.rand(1)*2*mu))
                    if fitness[i] >= fitness[s]:
                        win[i] += 1
            rank = np.argsort(-win)
            new_fitness[:] = fitness[:]
            new_population[:] = population[:]
            new_eta[:] = eta[:]
            #print(new_eta.shape, eta.shape)
            for i in range(mu):
                if rank[i] >=mu:
                    new_population[i] = children[rank[i]-mu]
                    #print(new_eta.shape, children_eta.shape, rank.shape)
                    new_eta[i] = children_eta[rank[i]-mu]
                else:
                    new_population[i] = population[rank[i]]
                    new_eta[i] = eta[rank[i]]
                new_fitness[i] = fitness[rank[i]]
            fitness[:] = new_fitness[:]
            population[:] = new_population[:]
            eta[:] = new_eta[:]
            t = np.argmax(fitness)
            if fitness[t]>best:
                best=fitness[t]
                bestx=population[t]
            print('max=', np.max(fitness), 'avg=', np.mean(fitness))
        return bestx
    def get_rate(self,k, tot_iter):
        rate = 10**(-1-k/tot_iter*4)
        rate= 0.00001
        return rate
    def mutation(self,mu, n, lb, ub, x, eta, rate):
        y = x+ eta * np.random.standard_cauchy((mu,n))
        #print(np.mean(eta), np.std(eta) )
        y = np.maximum(np.ones((mu,n)) * lb, y)
        y = np.minimum(np.ones((mu,n))*ub, y)
        tao = 1/np.sqrt(2*np.sqrt(n))
        tao_pie = 1/np.sqrt(2*n)
        new_eta = eta * np.exp(tao_pie*np.repeat(np.random.randn(mu,1), n, axis=1)+tao*np.random.randn(mu,n))
        new_eta = np.maximum(np.ones((mu,n))*rate, new_eta)
        #print(x,y)
        return y, new_eta
