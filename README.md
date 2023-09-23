# Tutorial

The purpose of this code is to make predictions of unobserved binary hyperedges in a hypergraph with binary hyperedges.
Note that the training contains data about all observations of hyperedges and non-hyperedges. 
Hyperedges in the test set should not appear in the training set neither as 1s nor as 0s in the adjacency matrix.

There are two opstions of model (code) for which we give an implementation:

    - a mixed-membership stochastic block model for hypergraphs in the Bernouilli formulation
    
    - a mixed-membership stochastic block model for hypergraphs in the Bernouilli formulation assuming an assortative connection probability matrix 

For both options we give two possible example implementations for both cases:

    - Predictions using a single run of the EM algorithm
    
    - Predictions using several runs of the EM  algorithm so that probabilities for each hyperedge in the test set are the averages over probabilities of each model.

To run this you should have installed python3 in your computer. The necessary modules are listed in the file requirements.txt
    

### Import the modules
First we import the necessary modules and update sys.path to include our working directory (if needed)


```python
import sys
##add current directory to the path so that you can find the modules
#sys.path+=['./'] 
from mmsbm import *
import assortative as ast
from random import seed
```

### Data format for train and test files
The input file (train.dat) should have a two-column format separated by a space ' ':

First column: hyperedge, string concatenating the different labels of the nodes participating in the hyperedge with '_'

Second column: 1/0 for edges and nonedges, respectively

Example:

sub1_sub2 0

sub3_sub_4_sub5_sub7 1

sub1_sub3_sub7 0

### Input parameters
You need to specify 

maxS - the largest hyperedge size

cdin - directory wehre the input data is

cdout - directory where the output files should go 

k - the number of latent groups in the MMSBM 

nruns - number of models to make predictions with 


```python
maxS=3 ## maximum size of the hyperedges in the dataset - in the example 3
cdin='./Data/' ##directory where to read the data from
cdout='./Output/'##optional directory for saving output files
k=3 ##latent dimension (number of groups)
nruns=5 # number of times we run the EM algorithm to find different models and average over them to make predictions
```

## Mixed-Membership SBM for binary hypergraphs - Bernouilli formulation

### Predictions from single run of the EM algorithm Mixed-Membership SBM



```python
seed(1) ##just for control, this line can be commented

fname = cdin+'train.dat'

edges, nodes, n2id, sizes = ReadEdges(fname,maxS) ##I have to do it here to restart n2id

print('Edges in training',sizes [2:])

theta, q, l2ps, ups = InitializeParameters(nodes, maxS, k)

logL0 = ComputeLikelihood(edges,theta,q,l2ps,sizes)

max_iter=5000

start_time1 = time.time()
theta,q = ObtainEMPars(theta,q,nodes,edges,l2ps,max_iter,sizes)

end_time1 = time.time()
print(f"time elapsed: {end_time1 - start_time1}")

logL = ComputeLikelihood(edges,theta,q,l2ps,sizes)

outfile = cdout+'out_nonassort.dat' 

test_fname = cdin+'test.dat'

predictions = MakePredictions(test_fname,n2id,theta,q,l2ps,outfile=outfile)

auc = ComputeAUC(predictions)

print('AUC',auc)
print('Theta',theta[0:10])
print('Q - affinity tensor, for hyperedges of size 3')
##note that affinity tensors have to elements corespoding to (p[0],p[1])
##since affinity tensors are flattened into a 1 dimensional array, we need to do some work to obtain the indices (kis)
ss=3
for ki in range(k**ss):
    kis=[]
    ki0=ki
    for i in range(ss):
        i1= int(ki0/k**(ss-i))
        ki0-=i1*k**(ss-i)
        kis.append(str(i1))
    print(' '.join(kis),q[ss][ki])
        

```

    Edges in training [1000. 1000.]
    time elapsed: 9.093852043151855
    AUC 0.8887161413099083
    Theta [[1.18848410e-01 4.30793099e-01 4.50358492e-01]
     [4.36194464e-01 3.12132860e-06 5.63802415e-01]
     [6.37565198e-01 3.19261844e-01 4.31729587e-02]
     [3.88986061e-02 6.43323844e-01 3.17777550e-01]
     [8.30607944e-01 2.47102502e-03 1.66921031e-01]
     [4.99882257e-01 1.76156431e-01 3.23961311e-01]
     [9.80502681e-01 3.90975053e-03 1.55875683e-02]
     [4.60151868e-01 3.09589534e-01 2.30258598e-01]
     [6.79642564e-01 2.38401002e-01 8.19564342e-02]
     [3.68950761e-01 2.60107079e-01 3.70942160e-01]]
    Q - affinity tensor, for hyperedges of size 3
    0 0 0 [1. 0.]
    0 0 0 [1. 0.]
    0 0 0 [1. 0.]
    0 0 1 [1. 0.]
    0 0 1 [1. 0.]
    0 0 1 [1. 0.]
    0 0 2 [1. 0.]
    0 0 2 [1. 0.]
    0 0 2 [1. 0.]
    0 1 0 [1. 0.]
    0 1 0 [1. 0.]
    0 1 0 [1. 0.]
    0 1 1 [1. 0.]
    0 1 1 [0. 1.]
    0 1 1 [1. 0.]
    0 1 2 [1. 0.]
    0 1 2 [1. 0.]
    0 1 2 [1. 0.]
    0 2 0 [1. 0.]
    0 2 0 [1. 0.]
    0 2 0 [1. 0.]
    0 2 1 [1. 0.]
    0 2 1 [1. 0.]
    0 2 1 [1. 0.]
    0 2 2 [1. 0.]
    0 2 2 [1. 0.]
    0 2 2 [1. 0.]


### Predictions from averages over nruns of the EM algorithm for Mixed-Membership SBM



```python
seed(1) ## for control only 

fname = cdin+'train.dat'

edges,nodes,n2id_ini,sizes = ReadEdges(fname,maxS)

print('Edges in training',sizes)

preds=[] ## Here we will store predictions for each 

for rep in range(nruns):

    n2id=np.array([i for i in n2id_ini]) ##We need to restart n2id each time

    theta, q, l2ps, ups = InitializeParameters(nodes, maxS, k)


    logL0 = ComputeLikelihood(edges,theta,q,l2ps,sizes)

    max_iter=5000

    start_time1 = time.time()
    theta,q = ObtainEMPars(theta,q,nodes,edges,l2ps,max_iter,sizes)

    end_time1 = time.time()
    print(f"time elapsed: {end_time1 - start_time1}")

    logL = ComputeLikelihood(edges,theta,q,l2ps,sizes)

    outfile = cdout+'out_nonassort_fold%d.dat' %(rep)

    test_fname = cdin+'test.dat'
    
    predictions = MakePredictions(test_fname,n2id,theta,q,l2ps,outfile=None)

    auct = ComputeAUC(predictions)

    print('Rep',rep,logL0,logL,auct)
    #print(theta[0])
    #print(q[2])
    preds.append(predictions)

    
    
auc=ComputeAUCv(preds)
print('AUC',auc)

```

    Edges in training [   0.    0. 1000. 1000.]
    time elapsed: 8.919795989990234
    Rep 0 -1681.4732578403007 -0.005905945380862406 0.8887161413099083
    time elapsed: 9.597474813461304
    Rep 1 -1390.5699607575582 -0.006581726190664013 0.9023320082003582
    time elapsed: 10.103657007217407
    Rep 2 -1701.970605758891 -0.006132701990427036 0.922974700140461
    time elapsed: 9.73670768737793
    Rep 3 -1234.7962107078279 -0.005918033072432983 0.9359596555277835
    time elapsed: 10.46608304977417
    Rep 4 -1486.229416933768 -0.0055888301207673375 0.9091809958712922
    AUC 0.9569093225308581


## Assortative Mixed-Membership SBM for binary hypergraphs - Bernouilli formulation


### Predictions from single run of the EM algorithm 



```python
seed(11)

fname = cdin+'train.dat'

edges,nodes,n2id = ast.ReadEdges(fname,maxS)
##n2id is a list that keeps track of the relationship between internal ids for nodes and real labels form the string

theta,q = ast.InitializeParameters(nodes,maxS,k)
##model parameters at random

logL0 = ast.ComputeLikelihood(edges,theta,q)

max_iter = int(5000*sqrt(k))##maximum number of iterations, in cas the internal check is never fulfilled
theta,q = ast.ObtainEMPars(theta,q,nodes,edges,max_iter)

logL = ast.ComputeLikelihood(edges,theta,q)

outfile = cdout+'output_assort.dat' #optional if you want to output model parameters

test_fname = cdin+'test.dat'

predictions = ast.MakePredictions(test_fname,n2id,theta,q,outfile=None)

auc = ast.ComputeAUC(predictions)

print('AUC',auc)
print('Theta')
for i in range(10):
    print(theta[i])
print('Q - affinity tensor, for hyperedges of size 3')
##note that affinity tensors have to elements corespoding to (p[0],p[1])
for i in range(k):
    print(i,q[3][i])
```

    150 break
    AUC 0.6700251889523441
    Theta
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    [0. 1. 0.]
    Q - affinity tensor, for hyperedges of size 3
    0 [0. 1.]
    1 [0.98135718 0.01864282]
    2 [0.08039591 0.91960409]


### Predictions from averages over nruns of the EM algorithm


```python
seed(11)

fname = cdin+'train.dat'

edges,nodes,n2id = ast.ReadEdges(fname,maxS)

preds=[] ## Here we will store predictions for each 

for rep in range(nruns):

    n2id=[i for i in n2id_ini] ##We need to restart n2id each time
    theta,q= ast.InitializeParameters(nodes,maxS,k)
    logL0=ast.ComputeLikelihood(edges,theta,q)
    max_iter=int(5000*sqrt(k))##maximum number of iterations, in cas the internal check is never fulfilled
    theta,q=ast.ObtainEMPars(theta,q,nodes,edges,max_iter)
    logL=ast.ComputeLikelihood(edges,theta,q)
    outfile= cdout+'output%d.dat' %(rep) #optional if you want to output model parameters
    test_fname = cdin+'test.dat'
    predictions= ast.MakePredictions(test_fname,n2id,theta,q,outfile=None)
    auct=ast.ComputeAUC(predictions)
    print(logL0,logL,auct) ##print out likelihoods and the auc for this particular model
    preds.append(predictions)

auc=ast.ComputeAUCv(preds) ##Compute auc by averaging prediction scores over models
print('AUC',auc)
```

    150 break
    -5383.720933403015 -183.28239400599048 0.6700251889523441
    150 break
    -6085.242905112643 -179.5709855782157 0.6410579345349798
    200 break
    -4954.258627773241 -173.0033474536133 0.6863979849222416
    100 break
    -6840.319376662561 -177.09061947039268 0.54534005037423
    450 break
    -4401.408416110124 -329.30721940223617 0.4999999999614363
    AUC 0.6863979849222416



```python

```
