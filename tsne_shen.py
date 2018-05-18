import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.manifold import TSNE

plt.ion()


import RestrictedData
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots


tsne = TSNE(n_components=2, verbose=1,
            perplexity=10, method='barnes_hut', angle=0.5,
            init='pca', early_exaggeration=12, learning_rate=200,
            n_iter=1000, random_state=123)
tsneResults = tsne.fit_transform(xnorms['shen'].values)


ggd = pd.DataFrame({'sample' : xnorms['shen'].index,
                    'system' : annots['shen'].reindex(xnorms['shen'].index)['System'],
                    'coord1' : tsneResults[:, 0],
                    'coord2' : tsneResults[:, 1]})
plt.close()
ggo = gg.ggplot(ggd, gg.aes(x='coord1', y='coord2', color='system', label='sample'))
ggo += gg.geom_point()
ggo += gg.geom_text(nudge_y=9, show_legend=False)
ggo += gg.scale_color_manual(values=['firebrick', 'goldenrod', 'lightseagreen',
                                     'darkorchid', 'darkslategray', 'dodgerblue'])
ggo += gg.theme_bw()
ggo += gg.xlab('tSNE coordinate 1')
ggo += gg.ylab('tSNE coordinate 2')
print(ggo)
