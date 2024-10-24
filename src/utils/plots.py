import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_example(models, test_dataset, n_examples = 2):

    fig, ax = plt.subplots(len(models)*n_examples, 4, figsize=((5)*4, 5*len(models)*n_examples), dpi = 300)

    random_indices = np.random.randint(0, len(test_dataset), size=(len(models), n_examples))

    random_indices = random_indices.flatten()
    imgs = test_dataset.tensors[0][random_indices]
    labels = test_dataset.tensors[1][random_indices]
    dataset = torch.utils.data.TensorDataset(imgs, labels)

    for i, model in enumerate(models):
        model.eval()

        reconstructed = model(dataset.tensors[0])
        
        for j in range(n_examples):

            ax[i*n_examples+j, 0].imshow(imgs[j][0, :, :], cmap='inferno')
            ax[i*n_examples+j, 0].set_xticks([])
            ax[i*n_examples+j, 0].set_yticks([])

            ax[i*n_examples+j, 2].imshow(imgs[j][1, :, :], cmap='inferno')
            ax[i*n_examples+j, 2].set_xticks([])
            ax[i*n_examples+j, 2].set_yticks([])

            ax[i*n_examples+j, 1].imshow(reconstructed[j][0, :, :].detach().numpy(), cmap='inferno')
            ax[i*n_examples+j, 1].set_xticks([])
            ax[i*n_examples+j, 1].set_yticks([])

            ax[i*n_examples+j, 3].imshow(reconstructed[j][1, :, :].detach().numpy(), cmap='inferno')
            ax[i*n_examples+j, 3].set_xticks([])
            ax[i*n_examples+j, 3].set_yticks([])

        # Center the ylabel between the current row and the row below
        ax[i*n_examples+j, 0].set_ylabel(f'Model {model.name}', fontsize=35, rotation=90, labelpad=50, va='center')
        ax[i*n_examples+j, 0].yaxis.set_label_coords(-0.1, +1)

        ax[i*n_examples, 0].set_title(f'CH1', fontsize=30)
        ax[i*n_examples, 1].set_title(f'CH1 Recon.', fontsize=30)
        ax[i*n_examples, 2].set_title(f'CH2', fontsize=30)
        ax[i*n_examples, 3].set_title(f'CH2 Recon.', fontsize=30)

    plt.tight_layout(pad=1.0)