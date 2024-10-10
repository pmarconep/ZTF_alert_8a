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

def plot_example_dataset(dataset1, dataset2, dataset3, sample='random', n=1):
    if sample == 'random':
        sample = np.random.choice(dataset1['Train']['class'].size, n)
    else:
        n = len(sample)
         
    fig, ax = plt.subplots(n, 3, figsize=(5, 1.8 * n))
    
    for i in range(n):
        ax[i, 0].imshow(dataset1['Train']['images'][sample[i]])
        ax[i, 0].set_title(f'Low Res {i + 1}')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        
        ax[i, 1].imshow(dataset2['Train']['images'][sample[i]])
        ax[i, 1].set_title(f'Mid Res {i + 1}')
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        
        ax[i, 2].imshow(dataset3['Train']['images'][sample[i]])
        ax[i, 2].set_title(f'High Res {i + 1}')
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    
    plt.show()
    
    
def plot_two_images(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img1)
    ax[0].set_title('Input')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].imshow(img2)
    ax[1].set_title('Reconstructed')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    plt.show()