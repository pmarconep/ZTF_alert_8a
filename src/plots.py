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
    
import torch
def plot_train_example(models, dataset, gaussian_dataset):
    sample = np.random.choice(len(dataset), len(models))
    sample_gaus = np.random.choice(len(gaussian_dataset), len(models))
    
    fig, axes = plt.subplots(len(models), 4, figsize=(10, 15))
    for j, model in enumerate(models):
        img = dataset[sample[j]][0]  # Obtener la imagen original del dataset
        img_gaus = gaussian_dataset[sample_gaus[j]][0].squeeze()
        
        # Mostrar la imagen original
        axes[j, 0].imshow(img)
        axes[j, 0].set_title(f'Original')
        axes[j, 0].set_xticks([])
        axes[j, 0].set_yticks([])
        
        axes[j, 2].imshow(img_gaus)
        axes[j, 2].set_title(f'Gaussian')
        axes[j, 2].set_xticks([])
        axes[j, 2].set_yticks([])
        
        model.eval()
        with torch.no_grad():
            img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Convertir la imagen a tensor
            reconstructed_img = model(img_tensor)[0].squeeze().detach().numpy()
            
        with torch.no_grad():
            gaus_tensor = torch.tensor(img_gaus).unsqueeze(0).unsqueeze(0)  # Convertir la imagen a tensor
            reconstructed_gaus = model(gaus_tensor)[0].squeeze().detach().numpy()
        
        axes[j, 1].imshow(reconstructed_img)
        axes[j, 1].set_title(f'Reconstructed')
        axes[j, 1].set_xticks([])
        axes[j, 1].set_yticks([])
        
        axes[j, 3].imshow(reconstructed_gaus)
        axes[j, 3].set_title(f'Reconstructed Gaussian')
        axes[j, 3].set_xticks([])
        axes[j, 3].set_yticks([])

        # Add a general title for each row
        axes[j, 0].set_ylabel(f'Model {j}', fontsize=16, rotation=0, labelpad=50)

    plt.show()