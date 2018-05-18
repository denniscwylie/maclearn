import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

def readTab(file):
    out = pd.read_csv(file, sep='\t', header=0, index_col=0)
    out.index.name = None
    return out

xFiles = {
    'shen' : 'rnaseq/shen2012/19-tissues-expr.tsv.gz',
    'patel' : 'rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz',
    'montastier' : 'pcr/GSE60946/GSE60946-raw.tsv.gz',
    'hess' : 'microarray/Hess/HessTrainingData.tsv.gz'
}
xs = {k : readTab(xFiles[k]) for k in xFiles}
## data files are all in genes-as-rows, samples-as-columns format
## for this class, we will work with opposite format
## (samples-as-rows, genes-as-columns):
xs = {k : xs[k].transpose() for k in xs}

annotFiles = {
    'patel' : 'rnaseq/GSE57872/GSE57872_MappedSampleAnnotation.tsv',
    'montastier' : 'pcr/GSE60946/GSE60946-annot.tsv',
    'hess' : 'microarray/Hess/HessTrainingAnnotation.tsv'
}
annots = {k : readTab(annotFiles[k]) for k in annotFiles}

annots['shen'] = pd.DataFrame({
    'Tissue' : xs['shen'].index.str.replace(r'\d*(-.*)?', '')
}, index = xs['shen'].index)
annots['shen']['System'] = pd.Series({
    'boneMarrow' : 'lymphatic',
    'brain' : 'nervous',
    'cerebellum' : 'nervous',
    'cortex' : 'nervous',
    'heart' : 'circulatory',
    'intestine' : 'digestive/excretory',
    'kidney' : 'digestive/excretory',
    'limb' : 'other',
    'liver' : 'digestive/excretory',
    'lung' : 'respiratory',
    'mef' : 'other',
    'mESC' : 'other',
    'olfactory' : 'nervous',
    'placenta' : 'other',
    'spleen' : 'lymphatic',
    'testes' : 'other',
    'thymus' : 'lymphatic'
}).reindex(annots['shen']['Tissue'].values).values
annots['shen']['Nervous'] = annots['shen']['System'] == 'nervous'

## check that data objects (xs) are aligned with annot objects (annots)
np.all([np.all(annots[k].index.to_series() == xs[k].index.to_series()) for k in xs])
