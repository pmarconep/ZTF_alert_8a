from torch.utils.data import TensorDataset

def generate_gaussian_image(size, center, sigma=1.0, noise_level=0.1, brightness=1.0, base_level=0.212):
    """Genera una imagen de tamaño `size` con una distribución gaussiana centrada en `center` 
    y un nivel base de `base_level`."""
    
    # Crear coordenadas
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    x, y = np.meshgrid(x, y)
    
    # Calcular la distancia desde el centro y la función gaussiana
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    
    # Ajustar el brillo
    g *= brightness
    
    # Establecer el valor base mínimo
    g = base_level + (1 - base_level) * g  # Ajustar la gaussiana para que esté en el rango [base_level, 1]
    
    # Agregar ruido uniforme que parte desde el base_level
    noise = np.random.uniform(-noise_level, noise_level, g.shape)
    g += noise
    # Asegurarse de que los valores estén en el rango [base_level, 1]
    g = np.clip(g, base_level, 1)
    g = (g - g.min()) / (g.max() - g.min())
    return g

def generate_gaussian_dataset(num_samples, image_size, border_margin=3, noise_level=0.1, brightness=1.0):
    """Genera un dataset de imágenes gaussianas con centros aleatorios."""
    images = []
    for _ in range(num_samples):
        center = (np.random.randint(border_margin, image_size - border_margin), 
                  np.random.randint(border_margin, image_size - border_margin))
        image = generate_gaussian_image(image_size, center, noise_level=noise_level, brightness=brightness)
        images.append(image)
    images = np.array(images)
    images = images[:, np.newaxis, :, :]  # Añadir una dimensión para el canal
    return torch.tensor(images, dtype=torch.float32)

# Parámetros del dataset
num_samples = 12800
image_size = 21
border_margin = 4
noise_level = 0.1
brightness = 0.7

# Generar el dataset
gaussian_images = generate_gaussian_dataset(num_samples, image_size, border_margin, noise_level, brightness)

# Crear un TensorDataset
gaussian_dataset = TensorDataset(gaussian_images)


# Verificar el tamaño del dataset
print(f"Dataset size: {len(gaussian_dataset)}")
print(f"Image shape: {gaussian_dataset[0][0].shape}")

# Visualizar algunas imágenes
def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        image = dataset[i][0].numpy().squeeze()  # Convertir a numpy y eliminar la dimensión del canal
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()

# Mostrar 10 imágenes del dataset
show_images(gaussian_dataset, num_images=5)