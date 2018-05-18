import matplotlib.pyplot as plt

import MaclearnUtilities
from MaclearnUtilities import safeFactorize, ggpca

plt.ion()

import RestrictedData
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys



## -----------------------------------------------------------------
plt.close()
ggpca(xnorms['shen'], annots['shen']['System'],
      rlab=True,
      cshow=0, colscale=['firebrick', 'goldenrod', 'lightseagreen',
                          'darkorchid', 'darkslategray', 'dodgerblue'])


plt.close()
ggpca(xnorms['shen'], annots['shen']['System'],
      rlab=True, clab=True,
      cshow=25, clightalpha=0.1,
      colscale=['firebrick', 'goldenrod', 'lightseagreen',
                'darkorchid', 'darkslategray', 'dodgerblue', 'gray'])


plt.close()
ggpca(xnorms['patel'], ys['patel'],
      rlab=False, clab=True, cshow=10, clightalpha=0)
