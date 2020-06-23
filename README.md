# Principles of Machine Learning for Bioinformatics

This four-day course introduces a selection of machine learning
methods used in bioinformatic analyses with a focus on gene expression
data. Topics covered include: unsupervised learning, dimensionality
reduction and clustering; feature selection and extraction; and
supervised learning methods for classification (e.g., random forests,
support vector machines, knn, etc.) and regression (with an emphasis
on regularization methods appropriate for high-dimensional
problems). Participants have the opportunity to apply these methods as
implemented in R and python to publicly available data.

## Course materials
Lecture notes are provided in three different formats:

### pdf document (R version)
- [maclearn-2020.pdf](maclear-2020.pdf)

### jupyter notebook (R version)
- [1-algorithms-that-learn-R.ipynb](1-algorithms-that-learn-R.ipynb)
- [2-unsupervised-R.ipynb](2-unsupervised-R.ipynb]
- [3-knn-and-cross-validation-R.ipynb](3-knn-and-cross-validation-R.ipynb)
- [4-feature-selection-extraction-R.ipynb](4-feature-selection-extraction-R.ipynb)
- [5-regression-models-R.ipynb](5-regression-models-R.ipynb)
- [6-svm-bootstrap-random-forests-auc-R.ipynb](6-svm-bootstrap-random-forests-auc-R.ipynb)

### jupyter notebook (python version)
- [1-algorithms-that-learn-Python.ipynb](1-algorithms-that-learn-Python.ipynb)
- [2-unsupervised-Python.ipynb](2-unsupervised-Python.ipynb]
- [3-knn-and-cross-validation-Python.ipynb](3-knn-and-cross-validation-Python.ipynb)
- [4-feature-selection-extraction-Python.ipynb](4-feature-selection-extraction-Python.ipynb)
- [5-regression-models-Python.ipynb](5-regression-models-Python.ipynb)
- [6-svm-bootstrap-random-forests-auc-Python.ipynb](6-svm-bootstrap-random-forests-auc-Python.ipynb)

### data
The directories **data** contains two example data sets (described in
the lecture notes). The remaining files in the repository are small R
or python scripts either `source`d (R) or `import`ed (Python) at
various points in the jupyter notebooks.

## Suggested prerequisites

Recommended for students with some prior knowledge of either R or
python. **Participants are expected to provide their own laptops with
recent versions of R and/or python installed.** Students will be
instructed to download several free software packages (including R
packages and/or python libraries such as including pandas and
sklearn).

## R packages

### from CRAN

The command below can be run within an R session to install most of
the required packages from CRAN; **some of these may take a while to
install, recommend installation prior to class if you intend to run
the R scripts.**

```R
install.packages(c("caret", "clue", "ggplot2", "glmnet", "HiDimDA",
                   "kernlab", "pheatmap", "pROC", "randomForest",
                   "rpart", "tidyr"))
```

### from Bioconductor

The package **genefilter** can be installed from Bioconductor using the
following code again run within an R session.

```R
install.packages("BiocManager")
BiocManager::install("genefilter")
```

## Python modules

The following Python modules are used in the included scripts; again I
would **recommend installing prior to class if you intend to run the
Python scripts**:
- matplotlib
- mlextend
- numpy
- pandas
- plotnine
- scikit-learn (a.k.a. sklearn)
- scipy
- seaborn
