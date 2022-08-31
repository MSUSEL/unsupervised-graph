# Binary file analysis using Graph2Vec and unsupervised clustering

The repository contains programs for
1. Creating Control Flow Graphs (CFG) from binary files using 'angr' package.
2. Create vector representation for CFGs using 'Graph2Vec' graph embedding.
3. Unsupervised clustering algorithm training with hold out validation.
4. Cluster prediction for Test dataset

## Dataset

Dataset is currently shared via the following link.

'https://montanaedu-my.sharepoint.com/:f:/g/personal/t41n359_msu_montana_edu/Etvup_aPEjRDs0VBTEi40_UBALmLdSShC2mch8DmuILZ8w?e=rzkdWv'

Datasets composition is as follows 

|       | Malware | Benign |
| ------|---------|--------|
| Train | 3000    | 3000   |
| Test  | 1000    | 1000   |

## 1. Creating Control Flow Graphs (CFG) from binary files using 'angr' package.

Creating CFGs using binary files. Due to the security risk of sharing Malware binary files online. These binary files 
are not shared. However, the hash values of the binaries are given for information. 

## 2. Create vector representation for CFGs using 'Graph2Vec' graph embedding.