# -*- coding: utf-8 -*-
#    Copyright (C) 2020 by
#    Max Falkenberg <max.falkenberg13@imperial.ac.uk>
#    All rights reserved.
#    BSD license.
"""
Correlated copying graph generator.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time

class cc_graph:
    """
    Creates a preferential attachment graph object for BA model or k2 model.
    Attributes
    ----------
    t: int
        Timestep. Equal to number of nodes in the network.
    p: float or 'k2'
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
    second_moment: int
        Current average second moment of degree. Only recorded if
        statistics == True. Not normalised by N.
    second_moment_track: list
        Average second moment of degree over time. Only recorded if
        statistics == True. Not normalised by N.
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
    plot_edge_growth(scaling=None)
        Plots edge growth if statistics have been recorded.
    """

    def __init__(self,p=0,seed = None,statistics = False):
        """
        Class for undirected correlated copying model.
        Parameters
        ----------
        p    :  float or 'k2', optional, default = 0.
                copying probability
                If p == 'k2', copy probability automatically set as ratio of
                observed to influence degree.
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
        if p == 'k2':
            pass
        elif p < 0 or p > 1:
            raise Exception("Probability `p` not in range `0 <= p <= 1`.")


        self.t = 2 #time step. Equals total number of nodes.
        self.p = p #copying probability
        self.seed = seed #Random seed
        random.seed(seed)
        self.__statistics = statistics #Track statistics?
        self.__targets = [0,1] #Target list
        self.T = 2 #Number of targets (including repeats)
        self.adjlist = [[1],[0]] #Adjacency list where nth list is a list of direct neighbors of node n
        self.obs_adjlist = [[1],[0]] #Adjacency list for the observed network
        self.k =[1,1] #Degree of nodes in influence network
        self.obs_k = [1,1] #Degree of nodes in observed network
        self.T_track = [] #Track number of edges in influence network over time
        self.second_moment = 2 #Second moment currently
        self.second_moment_track = [] #Second moment over time.

        self.k1_obs = [2] # total observed first degree over time
        self.k1_inf = [2] # total influence first degree over time
        self.k2_obs = [2] # total observed second degree over time
        self.k2_inf = [2] # total infuence second degree over time
        self.twomoment_obs = [2] # total observed second moment over time
        self.twomoment_inf = [2] # total influence second moment over time
        self.neighborsum_obs = [2] # total observed neighbour degree sum over time 
        self.neighborsum_inf = [2] # total obversed neighbor degree sum over time

    def add_nodes(self,N):
        """
        Add N nodes to the network.
        Parameters
        ----------
        N: int
            Number of nodes to add to the network.
        """
        start_time = time.time()
        for i in range(N):
            target = random.choice(self.__targets) #Initial target
            self.obs_adjlist[target] += [self.t] #Updates neighbors in observed network
            self.obs_adjlist += [[target]]
            self.obs_k += [1]
            self.obs_k[target] += 1

            copy_candidates = self.adjlist[target] #Neighbors of target which may be copied
            copy_nodes = [target]
            for j in copy_candidates:
                if isinstance(self.p,str):
                    if random.random() < self.obs_k[j]/self.k[j]:
                        copy_nodes.append(j)
                else:
                    if random.random() < self.p:
                        copy_nodes.append(j)
            self.__targets += copy_nodes #New copied targets
            self.__targets += [self.t]*len(copy_nodes) #New node targets
            self.adjlist += [copy_nodes] #Adjust adjacency lists
            for j in copy_nodes: #Adjust adjacency lists
                self.adjlist[j] += [self.t]
                self.k[j] += 1
            self.k += [len(copy_nodes)]
            self.t += 1
            if self.__statistics:
                self.T += 2*len(copy_nodes)
                self.T_track += [self.T] #Track number of edges
                self.second_moment += len(copy_nodes)**2 #Change in sum from new node
                for j in copy_nodes: #Change in second moment from existing nodes
                    self.second_moment += (2*self.k[j])-1 #+k**2 - (k-1)**2
                self.second_moment_track += [self.second_moment]

                self.k1_obs += [self.k1_obs[-1]+2*self.obs_k[-1]] # each new observed edge adds 2 to the sum
                self.k1_inf += [self.k1_inf[-1]+2*self.k[-1]] # each new influence edge adds 2 to the sum
                self.twomoment_obs += [self.twomoment_obs[-1]+self.obs_k[-1]**2+2*self.obs_k[target]-1] # add new node value and modified (single) target value minus previous target value
                self.twomoment_inf += [self.twomoment_inf[-1]+self.k[-1]**2] # add new node contribution first then...
                self.k2_obs += [self.k2_obs[-1]+2*(self.obs_k[target])] # (new node adds 1 to k2 of target and each neighbor of target excluding new node) = new node's k2
                self.neighborsum_obs += [self.neighborsum_obs[-1]+1+self.obs_k[target]-1+self.obs_k[target]] # add new node's contribution to target's neighbor sum and target's neighbors' sums excluding new node, and (new node's neighbor sum = target's obs_k), respectively
                k2_inf = set()
                twosteps = 0
                neighborsum_inf = 0
                for j in copy_nodes:
                    self.twomoment_inf[-1] += 2*self.k[j]-1
                    k2_inf |= set([j]+self.adjlist[j]) # all nodes 1 or 2 steps from new node
                    twosteps += len(set(self.adjlist[j]).difference(copy_nodes))-1 # new node contributes 1 to nodes 2 steps away via nodes 1 step away (exclude new node)
                    neighborsum_inf += self.k[j]
                self.k2_inf += [self.k2_inf[-1] + 2*(len(k2_inf)-1)] # (new node adds 1 to k2 of all nodes 1 or 2 steps from new node) = k2 of new node
                self.neighborsum_inf += [self.neighborsum_inf[-1]+len(copy_nodes)*self.k[-1]+twosteps+neighborsum_inf] # adds new node's contribution to nodes 1 step away and 2 steps away, and new node's neighbor sum, respectively
        print(time.time()-start_time)

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
            plt.title(mode,fontsize=15)
            plt.tick_params(labelsize='large',direction='out',right = False,top=False)
            plt.tight_layout()
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
            if isinstance(self.p,str):
                raise Exception("Influence kernel not available for self.p = \'k2\'")
            else:
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

    def plot_edge_growth(self,scaling=None):
        """
        Plot number of edges in the influence network over time.
        Parameters
        ----------
        scaling: float
            Plot hypothetical rescaled scaling for growth with exponent 'scaling'.
            Require 1<=scaling<=2.
            Ignored if float == None.
        """
        if self.__statistics != True:
            raise Exception('Statistics for edge growth not recorded.')
        x = np.arange(2,len(self.T_track)+2,dtype='uint64') #time
        x_track = np.arange(3,len(self.T_track)+3,dtype='uint64') #time

        plt.plot(x,x-1,color='k',ls='--',label=r'$\propto t$') #linear scaling
        plt.plot(x,(x*(x-1))/2,color='k',ls='-.',label=r'$\propto t^{2}$') #complete graph
        if scaling != None: #Add trendline for custom scaling
            if scaling < 1. or scaling > 2.:
                raise Exception('Require 1. <= scaling <= 2.')
            else:
                p_scaling = x ** (scaling) #Assumed scaling
                p_scaling /= p_scaling[0]
                plt.plot(x,p_scaling,color='k',ls=':',label=r'$\propto t^{scale}$') #p scaling
        T_track = np.array(self.T_track)/2
        plt.plot(x_track,T_track) #edge growth
        if isinstance(self.p,str):
            pass
        else:
            k_mom = self.p * np.array(self.second_moment_track)/(2*T_track)
            #Ratio of second to first moment scaled by p
            crossover = np.argmin(k_mom<1) #Index where k_mom exceeds 1
            if k_mom[crossover]>1: #Only plot if crossover reached
                plt.plot([x_track[crossover],x_track[crossover]],[T_track[0]-1,T_track[crossover]],ls=':',color='k')
                plt.plot([x_track[0]-1,x_track[crossover]],[T_track[crossover],T_track[crossover]],ls=':',color='k')
        plt.xlabel(r'$t$',fontsize = 21)
        plt.ylabel(r'$E(t)$',fontsize = 21)
        plt.xlim((2,None))
        plt.ylim((1,None))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def plot_averages(self,log=None):
        """
        Plot averages of interest over time
        Parameters
        ----------
        log: string
            Plot log-log if log == 'log'
            Plot log y if log == 'y'
            Ignored if log == None.
        """

        k1_obs = np.array(self.k1_obs)/np.arange(1,self.t,dtype=float)
        k1_inf = np.array(self.k1_inf)/np.arange(1,self.t,dtype=float)
        k2_obs = np.array(self.k2_obs)/np.arange(1,self.t,dtype=float)
        k2_inf = np.array(self.k2_inf)/np.arange(1,self.t,dtype=float)
        twomoment_obs = np.array(self.twomoment_obs)/np.arange(1,self.t,dtype=float)
        twomoment_inf = np.array(self.twomoment_inf)/np.arange(1,self.t,dtype=float)
        neighborsum_obs = np.array(self.neighborsum_obs)/np.arange(1,self.t,dtype=float)
        neighborsum_inf = np.array(self.neighborsum_inf)/np.arange(1,self.t,dtype=float)

        self.k1_inf = [2] # total influence first degree over time
        self.k2_obs = [2] # total observed second degree over time
        self.k2_inf = [2] # total infuence second degree over time
        self.twomoment_obs = [2] # total observed second moment over time
        self.twomoment_inf = [2] # total influence second moment over time
        self.neighborsum_obs = [2] # total observed neighbour degree sum over time 
        self.neighborsum_inf = [2] # total obversed neighbor degree sum over time

        plt.figure()
        plt.plot(k1_obs,'.',label='k1_obs') # plot average observed first degree
        if log is not None: plt.yscale('log')
        if log == 'log': plt.xscale('log')
        plt.xlabel(r'$t$')
        plt.legend()

        plt.figure()
        plt.plot(k1_inf,'.',label='k1_inf') # plot average influence first degree
        plt.plot(k2_obs,'.',label='k2_obs') # plot average observed second degree
        plt.plot(twomoment_obs,'.',label='twomoment_obs') # plot average observed second moment
        plt.plot(neighborsum_obs,'.',label='neighborsum_obs') # plot average observed neighbor degree sum
        if log is not None: plt.yscale('log')
        if log == 'log': plt.xscale('log')
        plt.xlabel(r'$t$')
        plt.legend()

        plt.figure()
        plt.plot(k2_inf,'.',label='k2_inf') # plot average influence second degree
        plt.plot(twomoment_inf,'.',label='twomoment_inf') # plot average influence second moment
        plt.plot(neighborsum_inf,'.',label='neighborsum_inf') # plot average influence neighbor degree sum
        if log is not None: plt.yscale('log')
        if log == 'log': plt.xscale('log')
        plt.xlabel(r'$t$')
        plt.legend()

        plt.show()

G = cc_graph(p=0.1, statistics=True)
G.add_nodes(10**5)
G.plot_averages(log='log')
