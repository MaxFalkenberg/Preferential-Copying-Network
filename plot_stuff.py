import corr_copy
import numpy as np
import matplotlib.pyplot as plt
import time

def run_for_time(p,t=1):
    G = corr_copy.cc_graph(p=p, statistics=True)
    start = time.time()
    while time.time()-start < t:
        G.add_nodes(10)
    plt.plot(np.arange(3,len(G.k1_inf)+3),np.array(G.k1_inf)/2,'.',ms=3,label=r'$p$ = %4.4f'%(p))

plt.figure()
t = 10*60
run_for_time(p=0.1,t=t)
run_for_time(p=0.01,t=t)
run_for_time(p=0.001,t=t)
run_for_time(p=0.0001,t=t)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=2)
plt.show()

# G = corr_copy.cc_graph(p=0.1, statistics=True)

# t1 = 10**4
# G.add_nodes(t1)
# G.degree_dist(mode='obs')
# plt.savefig('smalldist')

# t2 = 10**5
# G.add_nodes(t2-t1)
# G.degree_dist(mode='obs')
# plt.savefig('mediumdist')

# t3 = 10**6
# G.add_nodes(t3-t2)
# G.degree_dist(mode='obs')
# plt.savefig('largedist')

# G.plot_edge_growth()
# G.plot_averages(savefig=True)