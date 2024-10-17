import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_example(models, test_dataset, n_examples = 2):

    fig, ax = plt.subplots(4, len(models)*n_examples, figsize=((13/2)*4, 5*len(models)*n_examples), dpi = 300)

    random_indices = np.random.randint(0, len(test_dataset), size=(len(models), n_examples))

    imgs = [test_dataset.tensors[0][i] for i in random_indices]
    labels = [test_dataset.tensors[1][i] for i in random_indices]

    for i, model in enumerate(models):
        model.eval()

        reconstructed = model(imgs)
        
        for j in range(n_examples):

            ax[i*n_examples+j, 0].imshow(imgs[j][0, :, :], cmap='gray')
            ax[i*n_examples+j, 0].set_xticks([])
            ax[i*n_examples+j, 0].set_yticks([])

            ax[i*n_examples+j, 1].imshow(imgs[j][1, :, :], cmap='gray')
            ax[i*n_examples+j, 1].set_xticks([])
            ax[i*n_examples+j, 1].set_yticks([])

            ax[i*n_examples+j, 2].imshow(reconstructed[j][0, :, :], cmap='gray')
            ax[i*n_examples+j, 2].set_xticks([])
            ax[i*n_examples+j, 2].set_yticks([])

            ax[i*n_examples+j, 3].imshow(reconstructed[j][1, :, :], cmap='gray')
            ax[i*n_examples+j, 3].set_xticks([])
            ax[i*n_examples+j, 3].set_yticks([])

        # Center the ylabel between the current row and the row below
        ax[i*n_examples+j, 0].set_ylabel(f'Model {model.name}', fontsize=16, rotation=0, labelpad=50, va='center')
        ax[i*n_examples+j, 0].yaxis.set_label_coords(-0.1, -0.5)

        # Add a line separator between each model change
        for k in range(4):
            ax[i*n_examples+j, k].axhline(y=0, color='black', linewidth=1)

        # Add a vertical line separator between each column
        for col in range(1, 4):
            for row in range(len(models) * n_examples):
                ax[row, col].axvline(x=0, color='black', linewidth=1)

    ax[0, 0].set_title(f'CH1')
    ax[0, 1].set_title(f'CH2')
    ax[0, 2].set_title(f'CH1 Recon.')
    ax[0, 3].set_title(f'CH2 Recon.')