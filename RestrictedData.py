import numpy
import pandas
from pandas import DataFrame
from pandas import Series

import MaclearnUtilities

import NormalizedData
xs = NormalizedData.xs
xnorms = NormalizedData.xnorms
annots = NormalizedData.annots


patelSubtype = annots['patel'].SubType
patelKeepers = ((patelSubtype == 'subtype: Mes') |
                (patelSubtype == 'subtype: Pro'))
patelKeepers = annots['patel'].index[patelKeepers]

xs['patel'] = xs['patel'].ix[patelKeepers]
xnorms['patel'] = xnorms['patel'].ix[patelKeepers]
annots['patel'] = annots['patel'].ix[patelKeepers]


montastierTime = annots['montastier'].Time
montastierKeepers = ((montastierTime == 'C1') |
                     (montastierTime == 'C2'))

xs['montastier'] = xs['montastier'].ix[montastierKeepers]
xnorms['montastier'] = xnorms['montastier'].ix[montastierKeepers]
annots['montastier'] = annots['montastier'].ix[montastierKeepers]


## -----------------------------------------------------------------
## extract ys
## -----------------------------------------------------------------
ys = {
    "bottomly" : annots["bottomly"].strain,
    "patel" : annots["patel"].SubType,
    "montastier" : annots["montastier"].Time,
    "hess" : annots["hess"].pCRtxt
}

ynums = {k : MaclearnUtilities.safeFactorize(ys[k]) for k in ys}
