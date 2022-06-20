import pandas as pd

def rt(f):
    return pd.read_csv(f, sep="\t", index_col=0, header=0)

 ## training set:
hessTrain = rt("data/HessTrainingData.tsv.gz")
hessTrainAnnot = rt("data/HessTrainingAnnotation.tsv")
 ## align annotation data.frame with expression data:
hessTrainAnnot = hessTrainAnnot.loc[hessTrain.columns]

 ## test set:
hessTest = rt("data/HessTestData.tsv.gz")
hessTestAnnot = rt("data/HessTestAnnotation.tsv")
 ## align annotation data.frame with expression data:
hessTestAnnot = hessTestAnnot.loc[hessTest.columns]

probeAnnot = rt("data/U133A.tsv.gz")
 ## align hessTrain and hessTest to probeAnnot:
hessTrain = hessTrain.loc[probeAnnot.index]
hessTest = hessTest.loc[probeAnnot.index]

hessTrainY = pd.Series({"pCR":0, "RD":1}).loc[hessTrainAnnot["pCRtxt"]]
hessTrainY.index = hessTrainAnnot.index
hessTestY = pd.Series({"pCR":0, "RD":1}).loc[hessTestAnnot["pCRtxt"]]
hessTestY.index = hessTestAnnot.index
