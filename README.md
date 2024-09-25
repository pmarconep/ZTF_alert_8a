# Representación temporal de alertas ZTF
## P: alto, Tutor: Sebastián Guzmán

## Integrantes
- Juan Pablo Contreras
- Pascual Marcone

ALeRCE es un broker de alertas astronómicas que recibe y procesa observaciones provenientes del survey ZTF en tiempo real. Dentro de los datos entregados por las alertas se encuentran los stamps o imágenes del lugar donde se produce la alerta. Estos stamps están compuestos por 3 imágenes: La imagen de referencia, la cual corresponde a un promedio de la posición antes de la alerta, imagen de ciencia correspondiente a una foto en el momento en que se detecta la alerta y la imagen de diferencia, siendo una diferencia entre las dos anteriores. Un objeto puede ser detectado más de una vez, lo cual agrega un carácter temporal a las alertas. El objetivo del proyecto es utilizar un AutoEncoder compuesto por una parte convolucional que procesa las imágenes y una parte recurrente que pueda procesar las características recurrentes entregadas por ZTF. Como mínimo se espera que implementen un AutoEncoder convolucional de la primera alerta basándose en el modelo entregado en referencias [1] y exploren la representación generada en el espacio latente. La base de datos requiere ser preprocesada antes de poder ser utilizada y se espera que realicen aumentaciones similares a las del modelo de referencia [2] y las del modelo mejorado [5].

Una vez generado el modelo para la primera alerta, se espera que puedan utilizar la representación para clasificar, donde se espera agreguen features para mejorar la clasificación, comparando resultados con lo obtenido en referencias [2]. Finalmente, se espera que puedan incluir capas recurrentes al modelo para procesar el carácter temporal de las muestras y comparar resultados de representación y clasificación con referencias [2].

## Referencias
- [1] Astorga, N., Huijse, P., Estévez, P. A., Forster, F. (2018, July). Clustering of Astronomical Transient Candidates Using Deep Variational Embedding. In 2018 International Joint Conference on Neural Networks (IJCNN). IEEE.
- [2] Carrasco-Davis, Rodrigo, et al. "Alert Classification for the ALeRCE Broker System: The Real-time Stamp Classifier." arXiv preprint arXiv:2008.03309 (2020).
- [3] Detector de supernovas - Cabrera-Vives, G., Reyes, I., Förster, F., Estévez, P. A., & Maureira, J. C. (2017). Deep-hits: Rotation invariant convolutional neural network for transient detection. The Astrophysical Journal, 836(1), 97.
- [4] Carrasco-Davis, Rodrigo, et al. “Deep Learning for Image Sequence Classification of Astronomical Events”. arXiv preprint arXiv:1807.03869 (2018)
- [5] Neira, D., Estévez, P. A., Forster, F. Temporal Stamp Classifier: Classifying Short Sequences of Astronomical Alerts (2024)
