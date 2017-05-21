import pandas as pd
from pandas import DataFrame
from pandas import Series

def readTab(file):
    out = pd.read_csv(file, sep="\t", header=0, index_col=0)
    out.index.name = None
    return out

xFiles = {
    "bottomly" : "rnaseq/bottomly/bottomly_count_table.tsv.gz",
    "patel" : "rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz",
    "montastier" : "pcr/GSE60946/GSE60946-raw.tsv.gz",
    "hess" : "microarray/Hess/HessTrainingData.tsv.gz"
}
xs = {k : readTab(xFiles[k]) for k in xFiles}
## data files are all in genes-as-rows, samples-as-columns format
## for this class, we will work with opposite format
## (samples-as-rows, genes-as-columns):
xs = {k : xs[k].transpose() for k in xs}

annotFiles = {
    "bottomly" : "rnaseq/bottomly/bottomly_annot.tsv",
    "patel" : "rnaseq/GSE57872/GSE57872_MappedSampleAnnotation.tsv",
    "montastier" : "pcr/GSE60946/GSE60946-annot.tsv",
    "hess" : "microarray/Hess/HessTrainingAnnotation.tsv"
}
annots = {k : readTab(annotFiles[k]) for k in annotFiles}

## check that data objects (xs) are aligned with annot objects (annots)
all([all(annots[k].index.to_series() == xs[k].index.to_series()) for k in xs])
