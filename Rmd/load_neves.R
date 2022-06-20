rt = function(f) {
    read.table(f, sep="\t", row.names=1, header=TRUE,
               check.names=FALSE, comment.char="", quote="")
}

nevesExpr = log2(rt("../data/gse120430_deseq_normalized.tsv.gz") + 1)
 ## simplify nevesExpr by removing genes with no data:
nevesExpr = nevesExpr[rowSums(nevesExpr) > 0, ]
nevesAnnot = rt("../data/gse120430_sample_annotation.tsv")
dmGenes = rt("../data/d_melanogaster_gene_annotations.saf.gz")

 ## align sample annotations to expression data:
nevesAnnot = nevesAnnot[colnames(nevesExpr), , drop=FALSE]
 ## align dmGenes to expression data:
dmGenes = dmGenes[rownames(nevesExpr), ]

 ## use more descriptive names for samples
betterSampleNames = paste0(nevesAnnot$group, "-", 1:3)
colnames(nevesExpr) = betterSampleNames
rownames(nevesAnnot) = betterSampleNames

 ## use more descriptive names for genes
rownames(nevesExpr) = dmGenes$GeneName
