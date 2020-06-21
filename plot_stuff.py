import corr_copy

G = corr_copy.cc_graph(p=0.1, statistics=True)

t1 = 10**4
G.add_nodes(t1)
G.degree_dist(mode='obs',savefig=True)

t2 = 10**5
G.add_nodes(t2-t1)
G.degree_dist(mode='obs',savefig=True)

t3 = 10**6
G.add_nodes(t3-t2)
G.degree_dist(mode='obs',savefig=True)

G.plot_edge_growth(savefig=True)
G.plot_averages(savefig=True)