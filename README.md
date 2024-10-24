# Representación temporal de alertas ZTF
## P: alto, Tutor: Sebastián Guzmán
## Profesor: Pablo Estévez

# Descripcion Proyecto
ALeRCE es un broker de alertas astronómicas que recibe y procesa observaciones provenientes del survey ZTF en tiempo real. Dentro de los datos entregados por las alertas se encuentran los stamps o imágenes del lugar donde se produce la alerta. Estos stamps están compuestos por 3 imágenes: La imagen de referencia, la cual corresponde a un promedio de la posición antes de la alerta, imagen de ciencia correspondiente a una foto en el momento en que se detecta la alerta y la imagen de diferencia, siendo una diferencia entre las dos anteriores. Un objeto puede ser detectado más de una vez, lo cual agrega un carácter temporal a las alertas. El objetivo del proyecto es utilizar un AutoEncoder compuesto por una parte convolucional que procesa las imágenes y una parte recurrente que pueda procesar las características recurrentes entregadas por ZTF. Como mínimo se espera que implementen un AutoEncoder convolucional de la primera alerta basándose en el modelo entregado en referencias [1] y exploren la representación generada en el espacio latente. La base de datos requiere ser preprocesada antes de poder ser utilizada y se espera que realicen aumentaciones similares a las del modelo de referencia [2] y las del modelo mejorado [5].

Una vez generado el modelo para la primera alerta, se espera que puedan utilizar la representación para clasificar, donde se espera agreguen features para mejorar la clasificación, comparando resultados con lo obtenido en referencias [2]. Finalmente, se espera que puedan incluir capas recurrentes al modelo para procesar el carácter temporal de las muestras y comparar resultados de representación y clasificación con referencias [2].

## Referencias
- [1] Astorga, N., Huijse, P., Estévez, P. A., Forster, F. (2018, July). Clustering of Astronomical Transient Candidates Using Deep Variational Embedding. In 2018 International Joint Conference on Neural Networks (IJCNN). IEEE.
- [2] Carrasco-Davis, Rodrigo, et al. "Alert Classification for the ALeRCE Broker System: The Real-time Stamp Classifier." arXiv preprint arXiv:2008.03309 (2020).
- [3] Detector de supernovas - Cabrera-Vives, G., Reyes, I., Förster, F., Estévez, P. A., & Maureira, J. C. (2017). Deep-hits: Rotation invariant convolutional neural network for transient detection. The Astrophysical Journal, 836(1), 97.
- [4] Carrasco-Davis, Rodrigo, et al. “Deep Learning for Image Sequence Classification of Astronomical Events”. arXiv preprint arXiv:1807.03869 (2018)
- [5] Neira, D., Estévez, P. A., Forster, F. Temporal Stamp Classifier: Classifying Short Sequences of Astronomical Alerts (2024)
- [7] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification
with deep convolutional neural networks,” in Proceedings of the 25th
International Conference on Neural Information Processing Systems -
Volume 1, ser. NIPS’12. USA: Curran Associates Inc., 2012, pp. 1097–
1105


# Ejecutar Modelos

Para simplificar el uso de los modelos y simplicidad visual al momento de entrenar, el codigo base está escrito en archivos .py, los cuales son llamados y utilizados dentro de los notebooks. 

Como tip general, se puede utilizar el comando `importlib.reload()` para no tener que reiniciar el kernel cada vez que se hace un cambio en las librerias.

    import src.plots as plots
    importlib.reload(plots)

Esto actualiza el cache de la libreria.

## model_v2.ipynb

El trabajo realizado hasta la entrega preliminar se encuentra en el notebook `model_v2.ipynb`. El proceso de entrenar modelos se puede realizar ejecutando el notebook entero. Sin embargo, si se quiere utilizar cualquier modelo ya entrenado, se puede acceder a la carpeta `models/`, donde se pueden encontrar los archivos `.pth` de cada modelo. Para inicializar estos se debe realizar lo siguiente.

    import torch
    import src.model.ae as ae

    #initialize model, latent_dim must match with the pretrained model. Name is just for plotting porpuses.
    example_model = ae.AE(latent_dim=30, n_channels=2, name='example_model')

    example_model.load_state_dic(torch.load('models/model.pth))
    example_model.eval()

Now you can use that model to test it, or even train it more.

## Plotting and metrics

All plotting and metrics functions use a list of models, to plot all models in one figure. If you want to only plot one, put the model into `[]` so is a list of one element.

    metrics.plot_umap_lp([example_model], val_dataset, n_neighbors, min_dist, metric, norm)