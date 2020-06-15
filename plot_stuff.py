import corr_copy

G = corr_copy.cc_graph(p=0.1, statistics=True)
G.add_nodes(10**5)
G.plot_averages(log='log')
