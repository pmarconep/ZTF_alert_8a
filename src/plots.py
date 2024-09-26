import numpy as np
import matplotlib.pyplot as plt

def plot_example(dataset, sample='random', sample2='random', n=1):
    if sample == 'random' and sample2 == 'random':
        sample = np.random.choice(dataset['Train']['class'].size, n)
        sample2 = np.random.choice(5, n)
    else:
        n = len(sample)
         
    fig, ax = plt.subplots(n, 3, figsize=(5, 1.8 * n))
    
    for i in range(n):
        ax[i, 0].imshow(dataset['Train']['images'][sample[i]][sample2[i]])
        ax[i, 0].set_title(f'Image {i + 1}')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        
        ax[i, 1].imshow(dataset['Train']['template'][sample[i]])
        ax[i, 1].set_title(f'Template {i + 1}')
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        
        ax[i, 2].imshow(dataset['Train']['difference'][sample[i]][sample2[i]])
        ax[i, 2].set_title(f'Difference {i + 1}')
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    
    plt.show()