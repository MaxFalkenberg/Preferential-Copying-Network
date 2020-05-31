# -*- coding: utf-8 -*-
#    Copyright (C) 2020 by
#    Max Falkenberg <max.falkenberg13@imperial.ac.uk>
#    All rights reserved.
#    BSD license.
"""
Correlated copying graph generator.

"""

import numpy as np
import matplotlib.pyplot as plt
import time

class cc_graph:
    """
    Creates a preferential attachment graph object for BA model or k2 model.

    Attributes
    ----------
    t: int
        Timestep. Equal to number of nodes in the network.
    p: float
        Copying probability.
    seed: int
        Random number seed
    T:  int
        Number of edges in the current influence network.
    T_track: list
        Number of edges in the influence network for full evolution of
        network. Only accessible is statistics==True.
    k: list
        Influence degree of node i at index i.
    obs_k: list
        Observed degree of node i at index i.
    adjlist: list of lists
        Adjacency list for influence network.
    obs_adjlist: list of lists
        Adjacency list for observed network.

    Methods
    -------

    add_nodes(N)
        Add N nodes to the network.
    degree_dist(mode='inf',plot=True)
        Export degree distribution for influence network if mode == 'inf'
        or observed network if mode == 'obs'. Plot if plot == True.
    kernel(mode='inf',plot=True)
        Calculate the relative attachment kernel for influence network
        if mode == 'inf' or observed network if mode == 'obs'. Plot
        if plot == True.
    """

    def __init__(self,p=0,seed = None,statistics = False):
        """
        Class for undirected correlated copying model.


        Parameters
        ----------

        p    :  float, optional, default = 0.
                copying probability
        seed : integer, random_state, or None (default)
               Indicator of random number generation state.
               See :ref:`Randomness<randomness>`.
        statistics : boolean, default = True.
               Boolean to indicate whether to store statistics or not.


        Returns
        -------
        G : Graph


        Raises
        ------
        Error
            If `p` not in range ``0 <= p <= 1``.
        """

        if p < 0 or p > 1:
            raise Exception("Probability `p` not in range `0 <= p <= 1`.")


        self.t = 2 #time step. Equals total number of nodes.
        self.p = p #copying probability
        self.seed = seed #Random seed
        np.random.seed(seed = seed) #set random seed
        self.__statistics = statistics #Track statistics?
        self.__targets = [0,1] #Target list
        self.T = 2 #Number of targets (including repeats)
        self.adjlist = [[1],[0]] #Adjacency list where nth list is a list of direct neighbors of node n
        self.obs_adjlist = [[1],[0]] #Adjacency list for the observed network
        self.k =[1,1] #Degree of nodes in influence network
        self.obs_k = [1,1] #Degree of nodes in observed network
        self.T_track = [1,1] #Track number of edges in influence network over time

    def add_nodes(self,N):
        """
        Add N nodes to the network.

        Parameters
        ----------

        N: int
            Number of nodes to add to the network.
        """
        start = time.time() #Time counter
        for i in range(N):
            target = self.__targets[np.random.randint(self.T)] #Initial target
            self.obs_adjlist[target] += [self.t] #Updates neighbors in observed network
            self.obs_adjlist += [[target]]
            self.obs_k += [1]
            self.obs_k[target] += 1

            copy_candidates = self.adjlist[target] #Neighbors of target which may be copied
            copy_nodes = [target]
            for j in copy_candidates:
                if np.random.rand() < self.p:
                    copy_nodes.append(j)
            # copy_nodes = copy_candidates[np.random.rand(len(copy_candidates)) < self.p].astype('list') + [target] #Nodes to be copied
            self.__targets += copy_nodes #New copied targets
            self.__targets += [self.t]*len(copy_nodes) #New node targets
            self.T += 2*len(copy_nodes) #Total number of targets
            self.adjlist += [copy_nodes] #Adjust adjacency lists
            for j in copy_nodes: #Adjust adjacency lists
                self.adjlist[j] += [self.t]
                self.k[j] += 1
            self.k += [len(copy_nodes)]
            self.t += 1
            if self.__statistics:
                self.T_track += [self.T]
                # self.var_k += [np.var(self.k)]
                # self.var_obs_k += [np.var(self.obs_k)]
        print(time.time()-start)

    def degree_dist(self,mode = 'inf',plot=True):
        """
        Export degree distribution for the observed or influence network.

        Parameters
        ----------

        mode: 'inf' or 'obs':
            Export degree distribution for influence network if mode == 'inf'.
            Export degree distribution for observed network if mode == 'obs'.
            Plot degree distribution if plot == True.
        plot: boolean
            Plot degree distribution if True.

        Returns
        -------

        x: ndarray
            Degree of nodes for degree distribution.
        y: ndarray
            Probability that nodes in the network have a specific degree
            corresponding to equivalent index in x.

        """
        if mode == 'inf':
            y,x = np.histogram(self.k,bins=int(np.max(self.k)) - int(np.min(self.k)))
        else:
            y,x = np.histogram(self.obs_k,bins=int(np.max(self.obs_k)) - int(np.min(self.obs_k)))
        x = x[:-1]
        x = x[y != 0]
        y = y[y != 0]
        y = y.astype('float')
        y /= np.sum(y)

        if plot:
            plt.figure()
            plt.plot(x,y,ls='',marker='.')
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(x,4./(x*(x+1)*(x+2)),ls='--',color = 'k')
            plt.xlabel(r'$k$',fontsize = 21)
            plt.ylabel(r'$P(k)$',fontsize = 21)
            # plt.tight_layout()
            plt.title(mode,fontsize=15)
            plt.tick_params(labelsize='large',direction='out',right = False,top=False)
            plt.show()
        return x,y

    def kernel(self,mode='inf',plot=True):
        """
        Export relative attachment kernel for the network.

        Parameters
        ----------

        mode: 'inf' or 'obs':
            Export degree distribution for influence network if mode == 'inf'.
            Export degree distribution for observed network if mode == 'obs'.
            Plot degree distribution if plot == True.
            Note, the relative attachment kernel for the influence network is
            not the rescaled probability, but the rescaled expected number of
            edges attached to nodes with specific degree at time t.
        plot: boolean
            Plot degree distribution if True.

        Returns
        -------

        x: ndarray
            Degree of nodes for relative attachment kernel.
        y: ndarray
            Rescaled attachment probability for nodes in the network.
        """

        if mode == 'obs': #Observed network
            k1 = np.array(self.obs_k) #Degree in obs
            k2 = np.array(self.k) #Attractiveness in obs
        elif mode == 'inf': #Influence network
            k1 = np.array(self.k) #Degree in inf
            k2_raw = np.array(self.k)/self.T #Expected number of edges from direct attachment
            copy_prob = np.zeros_like(k1,dtype='float64')
            for count,i in enumerate(self.adjlist): #Expected number of edges from copying
                for j in i:
                    copy_prob[count] += self.p* k2_raw[j] #Contribution from each neighbour
            k2 = k2_raw + copy_prob
        else:
            raise Exception("Require mode == 'inf' or mode == 'obs'.")
        k1_bin = np.arange(np.min(k1),np.max(k1)+1) #Range of first degrees
        k2_bin = []
        for i in k1_bin: #Calculate average second degree for nodes with specific first degree
            if i in k1:
                k2_bin.append(np.mean(k2[k1==i]))
            else:
                k2_bin.append(None)
        k2_bin = np.array(k2_bin)
        k1_bin = k1_bin[k2_bin != None]
        k2_bin = k2_bin[k2_bin != None]
        k2_bin /= k2_bin[0]
        k2_bin *= k1_bin[0] #Rescale relative attachment kernel such that k2_bin[0] = 1
        if plot:
            plt.plot(k1_bin,k2_bin,marker='.',ls='',label = r'$N = $' + str(self.t))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$k$',fontsize=24)
            plt.ylabel(r'$\phi(k,1)$',fontsize=24)
            linear = np.arange(np.min(k1_bin),np.max(k1_bin)+1)
            plt.plot(linear,linear,ls='--',marker='',color='k',label = r'$\phi(k,1) \propto k$') #Expected scaling for BA model
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
        return k1_bin,k2_bin
