# Binary file analysis using Graph2Vec and unsupervised clustering

The repository contains 'main.py' programs for
1. Creating Control Flow Graphs (CFG) from binary files using 'angr' package.
2. Create vector representation for CFGs using 'Graph2Vec' graph embedding.
3. Unsupervised clustering algorithm training with hold out validation.
4. Cluster prediction for Test dataset

## Dataset

Due to the security risk of sharing Malware binary files online. These binary files 
are not shared. However, the hash values of the binaries are given for information. 

Dataset is permanently archived at the following link.

'https://doi.org/10.5281/zenodo.7630371'

Datasets composition is as follows 

|       | Malware | Benign |
| ------|---------|--------|
| Train | 3000    | 3000   |
| Test  | 1000    | 1000   |

Types of Malware according to NIST is  

| **Type**          | **Train** | **Test** | **Type**    | **Train** | **Test** |
|-------------------|-----------|----------|-------------|-----------|----------|
| adware            | 443       | 163      | krypt       | 1         | 0        |
| artemis           | 1         | 1        | kryptik     | 1         | 1        |
| banker            | 3         | 1        | midgare     | 0         | 1        |
| bechiro           | 0         | 1        | morstar     | 3         | 3        |
| bundler           | 1         | 1        | pua         | 5         | 2        |
| bundlore          | 3         | 1        | qakbot      | 1         | 0        |
| casino            | 1         | 0        | ransomware  | 2         | 0        |
| cinmus            | 1         | 0        | sefnit      | 1         | 0        |
| dialer            | 0         | 1        | softpulse   | 3         | 1        |
| domaiq            | 8         | 3        | solimba     | 8         | 4        |
| downloader        | 44        | 16       | squarenet   | 1         | 0        |
| dropper           | 2         | 10       | suspect     | 1         | 0        |
| fakeav            | 0         | 2        | symmi       | 1         | 0        |
| eurezo            | 1         | 0        | susppack    | 0         | 1        |
| file              | 1         | 0        | trojan      | 1792      | 652      |
| firseria          | 2         | 1        | ulpm        | 1         | 0        |
| firseriainstaller | 4         | 0        | unwanted    | 1         | 0        |
| hacktool          | 5         | 0        | virus       | 114       | 49       |
| installcore       | 5         | 2        | worm        | 202       | 68       |
| installerex       | 1         | 0        | Unspecified | 333       | 23       |
| kazy              | 2         | 0        |             |           |          |
