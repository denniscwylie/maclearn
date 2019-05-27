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

xs['patel'] = xs['patel'].loc[patelKeepers]
xnorms['patel'] = xnorms['patel'].loc[patelKeepers]
annots['patel'] = annots['patel'].loc[patelKeepers]


montastierTime = annots['montastier'].Time
montastierKeepers = ((montastierTime == 'C1') |
                     (montastierTime == 'C2'))

xs['montastier'] = xs['montastier'].loc[montastierKeepers]
xnorms['montastier'] = xnorms['montastier'].loc[montastierKeepers]
annots['montastier'] = annots['montastier'].loc[montastierKeepers]


## -----------------------------------------------------------------
## extract ys
## -----------------------------------------------------------------
ys = {
    'shen' : annots['shen'].Nervous,
    'patel' : annots['patel'].SubType,
    'montastier' : annots['montastier'].Time,
    'hess' : annots['hess'].pCRtxt
}

ynums = {k : MaclearnUtilities.safeFactorize(ys[k]) for k in ys}
