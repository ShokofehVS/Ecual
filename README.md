# Ecual
Ecual (a cybErseCurity platform for biclUstering ALgorithms): privacy-preserving gene expression data analysis by biclustering algorithm -- Cheng and Church algorithm -- over gene expression data performing Homomorphic Encryption operations in Python under the MIT license. We apply Zama's variant of TFHE (concrete-numpy) using a subset of numpy
that compile to FHE.

## Installation
First you need to ensure that all packages have been installed.
+ See `requirements.txt`
+ numpy>=1.22.3
+ setuptools>=60.2.0
+ pandas>=1.4.2
+ scikit-learn>=1.0.2
+ Bottleneck>=1.3.4
+ matplotlib>=3.5.2
+ scipy>=1.8.0
+ munkres>=1.1.4

You can clone this repository:

	   > git clone https://github.com/ShokofehVS/Ecual.git

If you miss something you can simply type:

	   > pip install -r requirements.txt

If you have all dependencies installed:

	   > python setup.py install

To install Concrete Numpy from PyPi, run the following command:  (more information regarding [installation of Concrete Numpy](https://github.com/zama-ai/concrete-numpy))

	   > pip install concrete-numpy

## Biclustering Algorithm
Biclustering or simultaneous clustering of both genes and conditions as a new paradigm was introduced by [Cheng and Church's Algorithm (CCA)](https://www.researchgate.net/profile/George_Church/publication/2329589_Biclustering_of_Expression_Data/links/550c04030cf2063799394f5e.pdf). The concept of bicluster refers to a subset of
genes and a subset of conditions with a high similarity score, which measures the coherence of the genes and conditions in the bicluster. It also returns the list of biclusters for the given data set. 

## Gene Expression Data Set
Our input data is synthetic data sets based on bicluster models (e.g., constant, shift, scale, shift-scale, and plaid) according to a procedure developed by [Victor A. Padilha et al. (2017)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1487-1)
taken from [Tavazoie et al. (1999)](https://pubmed.ncbi.nlm.nih.gov/10391217/) which was used in the orginal study by [Cheng and Church](https://www.researchgate.net/profile/George_Church/publication/2329589_Biclustering_of_Expression_Data/links/550c04030cf2063799394f5e.pdf);

## External Evaluation Measure
To measure the similarity of encrypted biclusters with non-encrypted version, we use Clustering Error (CE) as an external evaluation measure that was proposed by [Patrikainen and Meila (2006)](http://ieeexplore.ieee.org/abstract/document/1637417/) and Campello Soft Index (CSI) by [Horta and Campello, (2014)](https://horta.github.io/biclustering/paper/manuscript.pdf);
