import gym
import pcg_gym
import numpy as np
env = gym.make("mario_gan-v0")
lva = env.get_level(np.array([0.5]*32))
lvs = []
kl_val=[]
kls_val=[]
tot = 1000
for i in range(tot):
    lvs.append(env.get_level(env.sample_random_action()))
    kl_val.append(env.cal_kl_divergence_pairwise(lva, lvs[i]))
    kls_val.append(env.cal_kl_divergence(lva, lvs[i]))
kl_val, kls_val=np.array(kl_val), np.array(kls_val)
kl_rank, kls_rank = np.argsort(kl_val), np.argsort(kls_val)
print("kl",np.mean(kl_val), np.std(kl_val))
print("kls",np.mean(kls_val), np.std(kls_val))

dif=0
x = [0]*tot
for i in range(tot):
    x[kl_rank[i]]=i
for i in range(tot):
    dif += abs(x[kls_rank[i]]-i)
print('dif=', dif)
'''sum1,sum2=0,0
for i in range(100):
    sum1 += env.entropy(env.get_pattern_map(lvs[kl_rank[i+900]]))
    sum2 += env.entropy(env.get_pattern_map(lvs[kls_rank[i+900]]))
print(sum1, sum2)'''
#for i in [960,970,980,990,1000]:
#    env.saveWholeLevelAsImage([lva,lvs[kl_rank[i-1]]], "kl_"+str(i)+".jpg")
#    env.saveWholeLevelAsImage([lva,lvs[kls_rank[i-1]]], "kls_"+str(i)+".jpg")