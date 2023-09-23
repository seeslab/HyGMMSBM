import numpy as np
from random import random
from math import *
from scipy.stats import sem
eps= 1e-10

def ReadEdges(inputfile,maxS):
    nodes=[]#for each node it stores the total number of hyperedges of each size
    node2id=[]##store for each node number 0,1,2,... its corresponding real string
    edges=[] ##list of hyperedges of size k-1
    for s in range(maxS+1):edges.append([])
      
    with open(inputfile) as infile:
        data= infile.readlines()
        for line in data:
            e,val= line.strip().split()
            nlist=e.split('_')
            s=len(nlist) ##compute size of 
            nidlist=[]
            for n in nlist:
                if n not in node2id:
                    nid=len(node2id)
                    nn=[0]*(maxS+1)
                    nodes.append(nn)
                    nodes[nid][s]+=1
                    node2id.append(n)
                    
                else:
                    nid=node2id.index(n)
                    nodes[nid][s]+=1
                nidlist.append(str(nid))
            eid='_'.join(nidlist)
                        
            edges[s].append((eid,int(val))) ##edges[0] and edges[1] are just empty lists
    return edges,nodes,node2id

def InitializeParameters(nodes,maxS,K):
    #L is the latent dimension
    theta=[]
    for i in range(len(nodes)):
        
        t=np.array([random() for i in range(K)])
        t=[i/sum(t) for i in t]
        theta.append(t)
    q=[[] for k in range(maxS+1)]
    for k in range(2,maxS+1):
        for l in range(K):
            qq= random()
            q[k].append([qq,1-qq])
    
    return theta,q

def ComputeLikelihood(edges,theta,q):
    maxS=len(edges)-1
    K=len(theta[0])
    logL=0
    for s in range(2,maxS+1):
        for e,val in edges[s]:
            nlist=[int(i) for i in e.split('_')]
            p=0
            for k in range(K):
                pl=q[s][k][val]
                for i in nlist:pl*=theta[i][k]
                p+=pl
            #print(e,val,p)
            #for i in nlist:print('thetas',i,theta[i])
            logL+=log(p)
            
    return logL
                
def PerformIteration(theta,q,nodes,edges):
    K=len(theta[0])
    maxS=len(edges)-1
    ##computenew values according to update equations
    thetanew=[np.zeros(K) for i in range(len(nodes))]
    qnew=[[] for s in range(maxS+1)]
    for s in range(2,maxS+1):
        for k in range(K):
            qnew[s].append(np.zeros(2))            

    for s in range(2,maxS+1):
        for e,val in edges[s]:
            nlist=[int(i) for i in e.split('_')]
            w=np.zeros(K)
            sw=0
            for k in range(K):
                w[k]=q[s][k][val]
                for i in nlist:w[k]*=theta[i][k]
                sw+=w[k]
            for i in nlist: thetanew[i]=np.add(thetanew[i],w/sw)
            for k in range(K):qnew[s][k][val]+=w[k]/sw
        for k in range(K):
            sq=sum(qnew[s][k])+eps
            #print('B',k,l,qnew[k][l],sq)
            qnew[s][k]/=sq
            #print(qnew[s][k],sum(qnew[s][k]))
            
    for i in range(len(nodes)):
        thetanew[i]/=sum(nodes[i]) 
        #print(i,sum(thetanew[i]))
        
    
    return thetanew,qnew


def ObtainEMPars(theta,q,nodes,edges,niter,check=None):
    
    logL0=ComputeLikelihood(edges,theta,q)
        
      
    for iter in range(niter):
        theta,q=  PerformIteration(theta,q,nodes,edges)
        if iter % 50 == 0:
            logL = ComputeLikelihood(edges,theta,q)
            
            if (logL-logL0) < 1e-3:
                print(iter,'break')
                break
            logL0=logL
    return theta,q

def MakePredictions(tfile,n2id,theta,q,outfile=None):
    K=len(theta[0])
    ##coldstart - make average theta
       
    preds=[]
    
    thetaav=np.zeros(K)
    N=len(theta)
    for k in range(K):
        thetaav[k]=np.mean([t[k] for t in theta])
        
    with open(tfile) as infile:
        data= infile.readlines()
        for line in data:
            e,val= line.strip().split()
            nlist=e.split('_')
            s=len(nlist) ##compute size of hyoeredge
            nidlist=[]
            for n in nlist:
                try:
                    nidlist.append(n2id.index(n))
                except:#if not in list, coldstart, assign average theta
                    n2id.append(n)
                    theta.append(thetaav)
            p=0
            for k in range(K):
                pl=q[s][k][1]
                for n in nidlist: 
                    pl*=theta[n][k]
                p+=pl
            preds.append((e,pl,val))
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
    
 
