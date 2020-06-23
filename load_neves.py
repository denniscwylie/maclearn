import numpy as np
import pandas as pd

def rt(f):
    return pd.read_csv(f, sep="\t", index_col=0, header=0)

nevesExpr = np.log2(rt("data/gse120430_deseq_normalized.tsv.gz") + 1)
 ## simplify nevesExpr by removing genes with no data:
nevesExpr = nevesExpr.loc[nevesExpr.sum(axis=1) > 0]
nevesAnnot = rt("data/gse120430_sample_annotation.tsv")
dmGenes = rt("data/d_melanogaster_gene_annotations.saf.gz")

 ## align sample annotations to expression data:
nevesAnnot = nevesAnnot.loc[nevesExpr.columns]
 ## align dmGenes to expression data:
dmGenes = dmGenes.loc[nevesExpr.index]

 ## use more descriptive names for samples
betterSampleNames = [nevesAnnot["group"].iloc[i] + "-" + str(1+i%3)
                     for i in range(nevesAnnot.shape[0])]
nevesExpr.columns = betterSampleNames
nevesAnnot.index = betterSampleNames

 ## use more descriptive names for genes
nevesExpr.index = dmGenes["GeneName"]
