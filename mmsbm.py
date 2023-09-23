import numpy as np
from random import *
from math import *
from scipy.stats import sem
from itertools import permutations
from numba import jit
import time


PADDING_ELEMENT = 0

eps= 1e-10
max_list = 0
max_set = 0


def ReadEdges(inputfile,maxS):
    nodes=[]#for each node it stores the total number of hyperedges of each size
    node2id=[]##store for each node number 0,1,2,... its corresponding real string
    edges=[] ##list of hyperedges of size k-1
    for s in range(maxS+1):edges.append([])
      
    with open(inputfile) as infile:
        data = infile.readlines()
        for line in data:
            e,val = line.strip().split() # get values
            nlist = e.split('_') # Obtain edge nodes
            
            nlist = list(map(int, nlist))
            
            s = len(nlist) # hyperedge size
            nidlist = [] # list of node ids
            
            for n in nlist: # Values od hyperedge nodes
                if n not in node2id: # if label no in list, assign new id 
                    nid = len(node2id) #  Num current labels
                    nn = [0]*(maxS+1) # node representation
                    nodes.append(nn) # add to the list of hyperedges in which node nid participates
                    nodes[nid][s] += 1 # array hacia eladd to the counter of hyperedges of size s of node nid
                    node2id.append(int(n)) # add label
                    
                else:
                    nid = node2id.index(n)
                    nodes[nid][s] += 1 ##update counter
                nidlist.append(str(nid))
            eid='_'.join(nidlist)
                        
            edges[s].append((eid,int(val))) ##edges[0] and edges[1] are just empty lists
    sizes=np.zeros(maxS+1) ##we keep track of edge sizes we have
    for s in range(len(edges)):
        sizes[s]=len(edges[s]) ##we have to keep track of how many edges of each type we have, due to fixed size of arrays
            
    global max_list
    global max_set
    max_list = 0
    max_set = 0
    edges = list(map(split_edges, edges))
    edges = list(map(reshape_edge, edges))
            
    return np.array(edges), np.array(nodes), np.array(node2id), sizes

def reshape_edge(list_e):
    global max_list
    
    if len(list_e) != max_list:
        for _ in range(max_list - len(list_e)):
            list_e.append(([0,0,0], 0))
            
    list_e = list(map(reshape_set, list_e))
    
    return list_e

def reshape_set(set_e):
    global max_set
    
    if len(set_e[0]) != max_set:
        for _ in range(max_set - len(set_e[0])):
            set_e[0].append(0)
    
    set_e[0].append(set_e[1])
    
    return set_e[0]
        

def split_edges(edge):
    global max_list
    
    edge_r = []
    if len(edge) == 0:
        return edge_r
    else:
        result_split = list(map(process_edge, edge))
        max_list = len(result_split) if (max_list < len(result_split)) else max_list
        return result_split

def process_edge(set_e):
    global max_set
    
    edges = set_e[0].split('_')
    max_set = len(edges) if max_set < len(edges) else max_set
    edges = list(map(int, edges))
    return (edges, set_e[1])
                         
#@jit
def InitializeParameters(nodes,maxS,K):
    #K is the latent dimension
    
    theta=[]
    for i in range(len(nodes)):
        
        t=np.array([random() for i in range(K)])
        t=[i/np.sum(t) for i in t]
        theta.append(t)
    
    
    q=[[] for s in range(maxS+1)]
    l2ps=[[],[]]
    ups=[[],[]]
    for s in range(2,maxS+1):
        l2p,up =GetPermutations(s,K)
        q[s]=np.zeros((K**s,2))
        for p in up:
            qq= random()
            for r in p:
                q[s][r]=[qq,1-qq]
        l2ps.append(l2p)
        ups.append(up)
    
    q = preprocess_q(q)
    l2ps = preprocess_l2ps(l2ps)
    ups = preprocess_ups(ups)
    
    return np.array(theta), np.array(q), np.array(l2ps), np.array(ups)


def preprocess_ups(ups):
    max_ele = 0
    max_inner = 0
    new_ups = []
    
    for item in ups:
        max_ele = len(item) if max_ele < len(item) else max_ele
        for item_inner in item:
            max_inner = len(item_inner) if max_inner < len(item_inner) else max_inner
            
    for item in ups:
        if len(item) < max_ele:
            for _ in range(max_ele - len(item)):
                item.append([])
        for item_inner in item:
            if len(item_inner) < max_inner:
                item_inner += [0] * (max_inner - len(item_inner))
        
        new_ups.append(item)
    
    return new_ups
    
def preprocess_l2ps(l2ps):
    max_len = 0
    new_l2ps = []
    
    for item in l2ps:
        max_len = len(item) if max_len < len(item) else max_len
    
    for item in l2ps:
        item_a = item + [0] * (max_len - len(item))
        new_l2ps.append(item_a)
    
    return new_l2ps
    

def preprocess_q(q):
    max_row = 0
    new_q = []
    
    for item in q:
        if len(item) != 0:
            max_row = item.shape[0] if max_row < item.shape[0] else max_row
                
    for item in q:
        if len(item) == 0:
            item_a = np.zeros((max_row, 2))
            new_q.append(item_a)
        else:
            item_a = np.pad(item, ((0, max_row - item.shape[0]), (0, 0)), 'constant')
            new_q.append(item_a)
    
    return new_q


#@jit
def GetPermutations(S,K):
    
    label2perm=[0]*K**S
    foundind=[]
    uperms=[]
    upermsi=[]
    for k in range(K**S):
        if k not in foundind:
            indexes=GetIndexes(k,S,K)
            
            permstr= [ ''.join(p) for p in permutations(''.join([str(x) for x in indexes])) ]
            perms= [ np.sum([int(p[i])*K**i for i in range(len(p)) ]) for p in permutations(''.join([str(x) for x in indexes])) ]
            p0=perms[0]
            foundind+=set(perms)
            for p in perms: label2perm[p]=p0
            uperms.append([s for s in set(perms)])
            upermsi.append(p0)

    return label2perm ,uperms

@jit
def GetIndexes(n,s,k):
    indexes=[]
    num=n
    exp=s-1
    for i in range(s):
        index=int(num/k**exp)
        num-=index*k**exp
        indexes.append(index)
        exp-=1
    return(indexes)


@jit(nopython=True, nogil=True, cache=False, parallel=False) 
def ComputeLikelihood(edges,theta,q,l2ps,sizes):
    maxS = len(edges)-1
    K = len(theta[0])
    logL = 0
    
    for s in range(2, maxS + 1):
        size=sizes[s]
        for val in edges[s][:size]:

            p = 0

            for k in range(K**s):

                index = GetIndexes(k,s,K)

                kp = l2ps[s][k] #we point at the corresponidng permutation value

                pl = q[s][kp][val[-1]]

                for i in range(s):
                    ni = val[:-1][i]
                    pl *= theta[ni][index[i]]
                    #print('here',s,k,i,index[i],theta[i],len(theta))
                p += pl
            #print(s,val[:s],val[-1],p)
            logL += log(p)

    return logL

@jit(nopython=True, nogil=True, cache=False, parallel=False)                
def PerformIteration(theta,q,nodes,edges,l2ps,sizes):
    K = len(theta[0])
    maxS = len(edges)-1
    N = len(theta)
    #print('maxS',maxS)
    ##computenew values according to update equations
    
    #thetanew=[np.zeros(K) for i in range(len(nodes))]
    thetanew = np.zeros((len(nodes), K)) 
        
    #qnew=[np.zeros((K**s,2)) for s in range(maxS+1)]
    qnew = np.zeros((maxS + 1, K**(maxS + 1), 2))
    
    for s in range(2,maxS + 1):
        size=sizes[s]
        for val in edges[s][:size]: #iterate over edges 
            #print(e)
            nlist = val[:s]
            #print(nlist,val[-1])
            w = np.zeros(K**s)
            wt = np.zeros((len(theta),K)) ##overkill in the dimensions here, could be optimized
            sw = 0

            for k in range(K**s): ##iteration over all possible index combinations
                kp = l2ps[s][k]##symmetry enforcer
                #print('kp',kp)
                w[k] = q[s][kp][val[-1]]
                #print('q',q[s][kp],val)
                index = GetIndexes(k,s,K)##get index combinations
                #print(index)
                for i in range(s):
                    ni = nlist[i]
                    w[k] *= theta[ni][index[i]]
                for i in range(s):
                    ni = nlist[i]
                    ii = index[i]
                    wt[ni][ii] += w[k] ##get the terms that contribute to each theta_ialpha                    
                sw += w[k]##compute normalization
            for i in nlist: 
                for j in range(K):
                    thetanew[i][j] += wt[i][j] / sw #compute thetanew
            for k in range(K**s):
                qnew[s][k][val[-1]] += w[k] / sw ##qnew 
        for k in range(K**s):
            sq = np.sum(qnew[s][k]) + eps
            qnew[s][k] /= sq
    
    for i in range(N):
        st=np.sum(nodes[i])
        thetanew[i]=[t/st for t in thetanew[i]]
        
    return thetanew,qnew


@jit(nopython=True, nogil=True, cache=False, parallel=False) 
def ObtainEMPars(theta,q,nodes,edges,l2ps,niter,sizes):
    
    logL0 = ComputeLikelihood(edges, theta, q, l2ps,sizes)
    for iter in range(niter):
        theta,q=  PerformIteration(theta,q,nodes,edges,l2ps,sizes)
        if iter % 50 == 0:
            logL = ComputeLikelihood(edges,theta,q,l2ps,sizes)
            if (logL-logL0) < 1e-3:
            #    print('break',iter)
                break
            logL0=logL
            
    return theta,q

#@jit
def MakePredictions(tfile,n2id,theta,q,l2ps,outfile=None):
    
    n2id = n2id.tolist()
    theta = theta.tolist()
    #print('nodelist',n2id)
    #print('hh')
    K = len(theta[0])
    #print('hhh')
    #N=len(theta)
    #print('here')
    ##coldstart - make average theta
    thetaav = np.zeros(K)
    for k in range(K):
        thetaav[k]=np.mean([t[k] for t in theta])
      
    preds=[]
            
    with open(tfile) as infile:
        data= infile.readlines()
        for line in data:
            e,val= line.strip().split()
            #print(e,val)
            nlist=e.split('_')
            s=len(nlist) ##compute size of edge 
            nidlist=[]
            for n in nlist:
                try:
                    nidlist.append(n2id.index(int(n)))
                except:
                    n2id.append(int(n)) # Tamaño fijo list de nodesfixed size
                    theta.append(thetaav)
                    nidlist.append(len(n2id)-1)
            p=0
            for k in range(K**s):
     
                index=GetIndexes(k,s,K)
                kp=l2ps[s][k] #we point at the corresponding index permutation value
                pl=q[s][kp][1]
                for i in range(s):
                    ni=nidlist[i]
                    pl*=theta[ni][index[i]]
                p+=pl
            preds.append((e,p,val))
    if outfile:
        with open(outfile,'w') as outf:
            for e,pl,val in preds:
                outf.write(' '.join([e,str(pl),val])+'\n')
    return preds


def ComputeAUCv(vpreds):
    
    nreps=len(vpreds)
    evals=[[],[]]
    for e,s,val in vpreds[0]:
        evals[int(val)].append(e)
    escore={}
    for r in range(nreps):
        for e,s,val in vpreds[r]:
            try:
                escore[e]+=float(s)/nreps
            except:
                escore[e]=float(s)/nreps
    auc=0
    n0=len(evals[0])
    n1=len(evals[1])
    for e1 in evals[1]:
        s1=escore[e1]
        for e0 in evals[0]:
            s0=escore[e0]
            if fabs(s1-s0)<1e-8: auc+=0.5/n0/n1
            elif s1 > s0 : auc+=1./n0/n1
    
    return auc


    
def ComputeAUC(preds):
    
    evals=[[],[]]
    escore={}
    for e,s,val in preds:
        evals[int(val)].append(e)
        escore[e]=float(s)

    auc=0
    n0=len(evals[0])
    n1=len(evals[1])
    for e1 in evals[1]:
        s1=escore[e1]
        for e0 in evals[0]:
            s0=escore[e0]
            if fabs(s1-s0)<1e-8: auc+=0.5/n0/n1
            elif s1 > s0 : auc+=1./n0/n1
    
    return auc
