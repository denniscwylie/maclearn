rt = function(f) {
    read.table(f, sep="\t", row.names=1, header=TRUE,
               check.names=FALSE, comment.char="", quote="")
}

 ## training set:
hessTrain = rt("../data/HessTrainingData.tsv.gz")
hessTrainAnnot = rt("../data/HessTrainingAnnotation.tsv")
 ## align annotation data.frame with expression data:
hessTrainAnnot = hessTrainAnnot[colnames(hessTrain), ]

 ## test set:
hessTest = rt("../data/HessTestData.tsv.gz")
hessTestAnnot = rt("../data/HessTestAnnotation.tsv")
 ## align annotation data.frame with expression data:
hessTestAnnot = hessTestAnnot[colnames(hessTest), ]

probeAnnot = rt("../data/U133A.tsv.gz")
 ## align hessTrain and hessTest to probeAnnot:
hessTrain = hessTrain[rownames(probeAnnot), ]
hessTest = hessTest[rownames(probeAnnot), ]

hessTrainY = factor(hessTrainAnnot$pCRtxt)
names(hessTrainY) = rownames(hessTrainAnnot)
hessTestY = factor(hessTestAnnot$pCRtxt)
names(hessTestY) = rownames(hessTestAnnot)
