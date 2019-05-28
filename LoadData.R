## -----------------------------------------------------------------
## load data
## -----------------------------------------------------------------
readTab = function(file) {
    read.table(file, sep="\t",
               header=TRUE, row.names=1, check.names=FALSE)
}

xFiles = c(
    shen = "rnaseq/shen2012/19-tissues-expr.tsv.gz",
    patel = "rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz",
    montastier = "pcr/GSE60946/GSE60946-raw.tsv.gz",
    hess = "microarray/Hess/HessTrainingData.tsv.gz"
)
xs = lapply(X=xFiles, FUN=readTab)
## data files are all in genes-as-rows, samples-as-columns format
## for this class, we will work with opposite format
## (samples-as-rows, genes-as-columns):
xs = lapply(X = xs,
            FUN = function(x) {data.frame(t(x), check.names=FALSE)})

annotFiles = c(
    patel = "rnaseq/GSE57872/GSE57872_MappedSampleAnnotation.tsv",
    montastier = "pcr/GSE60946/GSE60946-annot.tsv",
    hess = "microarray/Hess/HessTrainingAnnotation.tsv"
)
annots = lapply(X=annotFiles, FUN=readTab)

annots$shen = data.frame(
    Tissue = gsub('\\d*(-.*)?', '', rownames(xs$shen)),
    row.names =  rownames(xs$shen)
)
annots$shen$System = c(
    'boneMarrow' = 'lymphatic',
    'brain' = 'nervous',
    'cerebellum' = 'nervous',
    'cortex' = 'nervous',
    'heart' = 'circulatory',
    'intestine' = 'digestive/excretory',
    'kidney' = 'digestive/excretory',
    'limb' = 'other',
    'liver' = 'digestive/excretory',
    'lung' = 'respiratory',
    'mef' = 'other',
    'mESC' = 'other',
    'olfactory' = 'nervous',
    'placenta' = 'other',
    'spleen' = 'lymphatic',
    'testes' = 'other',
    'thymus' = 'lymphatic'
)[annots$shen$Tissue]
annots$shen$Nervous = (annots$shen$System == 'nervous')

annots = annots[c('shen', 'patel', 'montastier', 'hess')]

## check that data objects (xs) are aligned with annot objects (annots)
mapply(
    FUN = function(x, annot) {all(rownames(x) == rownames(annot))},
    xs,
    annots
)
