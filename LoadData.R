## -----------------------------------------------------------------
## load data
## -----------------------------------------------------------------
readTab = function(file) {
    if (grepl("gz$", file)) {
        file = gzfile(file)
    }
    read.table(file, sep="\t",
            header=TRUE, row.names=1, check.names=FALSE)
}

xFiles = c(
    bottomly = "rnaseq/bottomly/bottomly_count_table.tsv.gz",
    patel = "rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz",
    montastier = "pcr/GSE60946/GSE60946-raw.tsv.gz",
    hess = "microarray/Hess/HessTrainingData.tsv.gz"
)
xs = lapply(X=xFiles, FUN=readTab)
## data files are all in genes-as-rows, samples-as-columns format
## for this class, we will work with opposite format
## (samples-as-rows, genes-as-columns):
xs = lapply(X=xs,
        FUN=function(x) {data.frame(t(x), check.names=FALSE)})

annotFiles = c(
    bottomly = "rnaseq/bottomly/bottomly_annot.tsv",
    patel = "rnaseq/GSE57872/GSE57872_MappedSampleAnnotation.tsv",
    montastier = "pcr/GSE60946/GSE60946-annot.tsv",
    hess = "microarray/Hess/HessTrainingAnnotation.tsv"
)
annots = lapply(X=annotFiles, FUN=readTab)

## check that data objects (xs) are aligned with annot objects (annots)
mapply(
    FUN = function(x, annot) {all(rownames(x) == rownames(annot))},
    xs,
    annots
)
