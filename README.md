# Principles of Machine Learning for Bioinformatics

This four-day course introduces a selection of machine learning
methods used in bioinformatic analyses with a focus on RNA-seq gene
expression data. Topics covered include: unsupervised learning,
dimensionality reduction and clustering; feature selection and
extraction; and supervised learning methods for classification (e.g.,
random forests, SVM, LDA, kNN, etc.) and regression (with an emphasis
on regularization methods appropriate for high-dimensional
problems). Participants have the opportunity to apply these methods as
implemented in R and python to publicly available data.

Lecture notes are provided in the four slide decks:
- [maclearn-1.pdf](maclearn-1.pdf)
- [maclearn-2.pdf](maclearn-2.pdf)
- [maclearn-3.pdf](maclearn-3.pdf)
- [maclearn-4.pdf](maclearn-4.pdf)

The directories **microarray**, **pcr**, and **rnaseq** contain
example data sets. Most of the remaining files in the repository are R
or python scripts (most scripts are available in essentially
equivalent form in both languages).

## Suggested prerequisites

Recommended for students with some prior knowledge of either R or
python. **Participants are expected to provide their own laptops with
recent versions of R and/or python installed.** Students will be
instructed to download several free software packages (including R
packages and/or python libraries such as including pandas and
sklearn).

## R packages used:

### from CRAN

The command below can be run within an R session to install most of
the required packages from CRAN; **some of these may take a while to
install, recommend installation prior to class.**

```R
install.packages(c('ada', 'caret', 'devtools', 'e1071', 'ggplot2',
                   'ggrepel', 'GGally', 'glmnet', 'MASS', 'matrixStats',
                   'pheatmap', 'randomForest', 'rpart', 'Rtsne', 'tidyr'))
```

### from Bioconductor

The package genefilter can be installed from Bioconductor using the
following code again run within an R session.

```R
install.packages('BiocManager')
BiocManager::install('genefilter')
```

### from github

The package sparsediscrim can be installed from github using the
following code again run within an R session.

```R
devtools::install_github('ramhiser/sparsediscrim')
```

## Python modules used:

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- plotnine
- seaborn

## Scripts to study by day

### Day 1: Loading Data, Normalization, Unsupervised Analysis
| R                                  | Python                                 | Notes                               |
|------------------------------------+----------------------------------------+-------------------------------------|
| [LoadData.R](LoadData.R)           | [LoadData.py](LoadData.py)             |                                     |
| [NormalizeData.R](NormalizeData.R) | [NormalizedData.py](NormalizedData.py) | RLE- and mean-center-normalization  |
| [Clustering.R](Clustering.R)       | [Clustering.py](Clustering.py)         | k-means and hierarchical clustering |
| [PCA_intro.R](PCA_intro.R)         |                                        |                                     |
| [PCA.R](PCA.R)                     | [PCA.py](PCA.py)                       |                                     |

### Day 2: knn classification, overfitting, cross-validation, feature selection
| R                            | Python                         | Notes                                                  |
|------------------------------+--------------------------------+--------------------------------------------------------|
| [KnnSim.R](KnnSim.R)         | [KnnSim.py](KnnSim.py)         | compare resub vs. test performance on simulated data   |
| [KnnSimCV.R](KnnSimCV.R)     | [KnnSimCV.py](KnnSimCV.py)     | show cross-validation (cv) removes resub bias          |
| [BadFeatSel.R](BadFeatSel.R) | [BadFeatSel.py](BadFeatSel.py) | supervised feature selection must be done under cv     |
| [KnnGrid.R](KnnGrid.R)       | [KnnGrid.py](KnnGrid.py)       | compare cv acc for varying k parameter on real data    |
| [KnnReal.R](KnnReal.R)       | [KnnReal.py](KnnReal.py)       | t-test feature selection/extraction + knn on real data |

### Day 3: linear models, regularization, naive bayes
| R                                                        | Python                                                     | Notes |
|----------------------------------------------------------+------------------------------------------------------------+-------|
| [TTesting.R](TTesting.R)                                 | [TTesting.py](TTesting.py)                                 |       |
| [PredictingGeneExpression.R](PredictingGeneExpression.R) | [PredictionGeneExpression.py](PredictionGeneExpression.py) |       |
| [WhyRegularize.R](WhyRegularize.R)                       | [WhyRegularize.py](WhyRegularize.py)                       |       |
| [LogisticReal.R](LogisticReal.R)                         | [LogisticReal.py](LogisticReal.py)                         |       |
| [LdaIsLikeLogistic.R](LdaIsLikeLogistic.R)               |                                                            |       |
  
### Day 4: svm, bootstrap, trees, random forests, boosting
| R                                            | Python                                         | Notes                                        |
|----------------------------------------------+------------------------------------------------+----------------------------------------------|
| [SvmReal.R](SvmReal.R)                       | [SvmReal.py](SvmReal.py)                       |                                              |
| [bootstrap_examples.R](bootstrap_examples.R) |                                                | mostly taken from package bootstrap examples |
| [KnnSimBoot.R](KnnSimBoot.R)                 |                                                |                                              |
| [RandomForestReal.R](RandomForestReal.R)     | [RandomForestReal.py](RandomForestReal.py)     |                                              |
| [AdaBoostReal.R](AdaBoostReal.R)             | [AdaBoostReal.py](AdaBoostReal.py)             |                                              |
| [CompareModelStrats.R](CompareModelStrats.R) | [CompareModelStrats.py](CompareModelStrats.py) |                                              |
