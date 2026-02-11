
El sistema **Focus Lens** optimiza la inferencia mediante recortes cuadrados que aumentan la **resolución efectiva** y reducen el ruido. Usa estrategias de **zoom** o **pad**, valida dimensiones en múltiplos de 80 y mapea resultados al marco original preservando **detecciones** y **puntos clave**.


> ¿Cómo mejora Focus Lens la precisión y eficiencia del modelo?

Focus Lens mejora la **precisión** y **eficiencia** de los modelos de visión por computadora al concentrar la densidad de píxeles en una región de interés específica dentro de un cuadro, en lugar de procesar toda la imagen por igual.

A continuación se detallan los mecanismos específicos mediante los cuales logra estas mejoras:

### 1. Mejora de la Precisión (Accuracy)

El sistema Focus Lens incrementa la calidad de los resultados de inferencia a través de los siguientes métodos:

- **Mayor resolución efectiva:** Al extraer un recorte (crop) de la región de interés y escalarlo para que llene la resolución de entrada del modelo, los detalles del sujeto se vuelven más nítidos para el algoritmo. Esto es especialmente beneficioso para la detección de **objetos pequeños** dentro del área recortada.
- **Reducción de ruido de fondo:** Al eliminar las distracciones e información irrelevante del fondo antes de la inferencia, se reduce la posibilidad de falsos positivos o errores causados por elementos ajenos al sujeto de estudio.
- **Optimización para arquitecturas específicas:** El tamaño del recorte (`focus_size`) está restringido a múltiplos de 80 para alinearse con las arquitecturas de modelos tipo YOLO, lo que evita errores de dimensionamiento y asegura que el modelo trabaje con dimensiones óptimas para sus capas de reducción de escala (strides).

### 2. Mejora de la Eficiencia

Focus Lens optimiza el uso de recursos computacionales de la siguiente manera:

- **Concentración de la capacidad de inferencia:** Evita el desperdicio de ciclos de procesamiento en áreas del cuadro que no contienen sujetos de interés.
- **Velocidad de procesamiento:** El uso de recortes más pequeños puede permitir una selección de modelos más rápida o reducir la sobrecarga de preprocesamiento.
- **Operaciones vectorizadas:** Aunque el sistema añade pasos de preprocesamiento y postprocesamiento, las transformaciones de coordenadas para cajas delimitadoras, máscaras y puntos clave se realizan de forma **vectorizada mediante NumPy**. Esto hace que la sobrecarga computacional de mapear los resultados de vuelta al cuadro completo sea insignificante en comparación con el tiempo de inferencia del modelo.
- **Estrategias flexibles para cuadros pequeños:** Permite elegir entre escalar la imagen (`zoom`) o añadir bordes negros (`pad`). La estrategia de "pad" permite preservar los datos originales de los píxeles sin el costo computacional o los artefactos de la interpolación.




---

> ¿Qué sucede si el sujeto se mueve fuera del recorte?


Si el sujeto se desplaza fuera del área definida por el recorte del **Focus Lens**, ocurren las siguientes situaciones según los documentos técnicos:

- **Pérdida de detección:** El sistema no detectará ningún objeto que se encuentre completamente fuera de la región recortada. Esto se debe a que el modelo de inferencia solo procesa los píxeles contenidos dentro del recorte para mejorar la eficiencia.
- **Degradación de la calidad:** Si el sujeto se mueve hacia el borde y el recorte corta una parte del cuerpo u objeto, la **calidad de la detección se degrada**. Las fronteras del recorte pueden "fragmentar" al sujeto, lo que dificulta que el modelo lo reconozca correctamente.
- **Limitación de los recortes estáticos:** El sistema es vulnerable a sujetos en movimiento cuando se utilizan **recortes estáticos** (posiciones fijas de `focus_x` y `focus_y`), ya que estos no siguen al objetivo automáticamente.
- **Comportamiento según la configuración:**
    - Si se usa la configuración **centrada** (`None`), el recorte permanecerá fijo en el centro del cuadro general, perdiendo al sujeto si este se desplaza hacia los extremos del marco completo.
    - Si se usa una **posición fija** para monitorear un área específica (como una puerta), cualquier actividad fuera de esas coordenadas exactas será ignorada por el modelo.

En resumen, aunque el **Focus Lens** aumenta la precisión dentro del área de interés, introduce el riesgo de "ceguera" ante cualquier evento o movimiento que ocurra fuera de los límites del recorte.

---

> ¿Cómo se mapean las detecciones de vuelta al cuadro completo?


El mapeo de las detecciones de vuelta al cuadro completo es un proceso de **postprocesamiento** que traduce las coordenadas locales del recorte (crop) a las dimensiones originales del video. Este procedimiento es fundamental para que los resultados de la inferencia se alineen correctamente con la imagen completa y utiliza la información capturada en el objeto **CropInfo**, que registra la posición `(x, y)` y el factor de escala (`scale_factor`) aplicados durante el preprocesamiento.

El proceso se divide principalmente en dos transformaciones matemáticas:

1. **Escalado inverso:** Si se aplicó la estrategia de "zoom" (el cuadro era más pequeño que el tamaño del recorte), las coordenadas detectadas se dividen por el `scale_factor`. Esto revierte el aumento de tamaño realizado para llenar la resolución de entrada del modelo.
2. **Aplicación de desplazamiento (offset):** Se suma la posición de origen del recorte `(x, y)` a las coordenadas desescaladas para trasladarlas desde el espacio local del recorte al espacio global del cuadro completo.

### Manejo según el tipo de detección

El sistema aplica estas reglas de forma distinta según la naturaleza de los datos:

- **Cajas delimitadoras (Bounding Boxes):** La transformación se realiza de forma **vectorizada** utilizando NumPy. Se ajustan las coordenadas de los ejes X (`x1`, `x2`) y Y (`y1`, `y2`) simultáneamente, asegurando que el proceso sea extremadamente rápido.
- **Máscaras de segmentación:** Al ser matrices 2D, el proceso es más complejo. Primero, si hubo escalado, la máscara se redimensiona a sus dimensiones originales sin escala utilizando una interpolación de "vecino más cercano" (`INTER_NEAREST`) para no alterar los valores binarios. Luego, se crea una máscara vacía del tamaño del cuadro completo y se inserta la máscara del recorte en la posición del desplazamiento.
- **Puntos clave (Keypoints):** Se aplica una lógica especial para preservar el significado semántico de los puntos en `(0, 0)`, que convencionalmente indican que un punto es "invisible" o no detectado. Estos puntos se mantienen en el origen, mientras que los puntos visibles se someten al escalado inverso y al desplazamiento de forma vectorizada.

Gracias a que estas operaciones están optimizadas con **NumPy**, la sobrecarga computacional de mapear los resultados de vuelta al cuadro completo es insignificante en comparación con el tiempo que toma la inferencia del modelo.

---


> ¿En qué casos es mejor usar pad en lugar de zoom?


De acuerdo con los documentos técnicos, la elección entre la estrategia **pad** y **zoom** depende de las prioridades del procesamiento de la imagen. Es mejor utilizar **pad** en los siguientes casos:

- **Preservación de los datos originales de los píxeles:** La estrategia **pad** es ideal cuando es crítico mantener la integridad de los píxeles originales sin distorsión alguna. A diferencia de **zoom**, esta opción no escala la imagen, por lo que se mantiene un factor de escala (`scale_factor`) de 1.0.
- **Evitar artefactos de interpolación:** Al no redimensionar la imagen, se evitan los artefactos visuales que pueden introducirse mediante algoritmos de interpolación (como `INTER_LINEAR` usado en zoom).
- **Procesamiento de video de baja resolución:** Se recomienda específicamente para videos de baja resolución donde la claridad de los datos originales es vital para la precisión del modelo.
- **Eficiencia en el uso de memoria:** En procesamientos de video de alta resolución, la estrategia de **zoom** puede consumir una cantidad significativa de memoria al tener que asignar espacio para un cuadro escalado antes del recorte. La estrategia **pad** simplemente añade bordes negros (`cv2.BORDER_CONSTANT`) para alcanzar el tamaño de enfoque (`focus_size`) sin aumentar la escala del contenido original.

En resumen, mientras que **zoom** busca maximizar la cobertura de la región de inferencia aceptando pequeñas distorsiones, **pad** prioriza la fidelidad de la imagen original añadiendo bordes negros para cumplir con los requisitos técnicos del modelo (como ser múltiplo de 80).


---

> De acuerdo con los documentos técnicos, la elección entre la estrategia **pad** y **zoom** depende de las prioridades del procesamiento de la imagen. Es mejor utilizar **pad** en los siguientes casos:

- **Preservación de los datos originales de los píxeles:** La estrategia **pad** es ideal cuando es crítico mantener la integridad de los píxeles originales sin distorsión alguna. A diferencia de **zoom**, esta opción no escala la imagen, por lo que se mantiene un factor de escala (`scale_factor`) de 1.0.
- **Evitar artefactos de interpolación:** Al no redimensionar la imagen, se evitan los artefactos visuales que pueden introducirse mediante algoritmos de interpolación (como `INTER_LINEAR` usado en zoom).
- **Procesamiento de video de baja resolución:** Se recomienda específicamente para videos de baja resolución donde la claridad de los datos originales es vital para la precisión del modelo.
- **Eficiencia en el uso de memoria:** En procesamientos de video de alta resolución, la estrategia de **zoom** puede consumir una cantidad significativa de memoria al tener que asignar espacio para un cuadro escalado antes del recorte. La estrategia **pad** simplemente añade bordes negros (`cv2.BORDER_CONSTANT`) para alcanzar el tamaño de enfoque (`focus_size`) sin aumentar la escala del contenido original.

En resumen, mientras que **zoom** busca maximizar la cobertura de la región de inferencia aceptando pequeñas distorsiones, **pad** prioriza la fidelidad de la imagen original añadiendo bordes negros para cumplir con los requisitos técnicos del modelo (como ser múltiplo de 80).

---

> ¿Cómo afectan los artefactos de interpolación a la precisión?


Los artefactos de interpolación son una consecuencia técnica de la **estrategia de zoom** en el sistema Focus Lens, y su impacto en la precisión se maneja como un compromiso entre la resolución y la fidelidad de los datos.

Según las fuentes, así es como afectan y cómo se gestionan estos artefactos:

### 1. El compromiso entre resolución y artefactos

Cuando un cuadro es más pequeño que el tamaño de enfoque (`focus_size`), la estrategia de **zoom** lo escala utilizando una interpolación lineal (`INTER_LINEAR`).

- **Impacto potencial:** Estos artefactos pueden introducir pequeñas distorsiones o falta de nitidez en los bordes de los objetos al "inventar" píxeles para llenar el espacio escalado.
- **Beneficio compensatorio:** A pesar de los artefactos, el sistema acepta este riesgo porque el aumento en la **resolución efectiva** del sujeto suele compensar la pérdida de precisión, permitiendo una mejor detección de objetos pequeños que de otro modo serían ilegibles para el modelo.

### 2. Riesgos en videos de baja resolución

En casos específicos donde la calidad de la imagen ya es comprometida (como en videos de muy baja resolución), los artefactos de interpolación pueden ser más perjudiciales. Las fuentes indican que:

- Para estos escenarios, es crítico **preservar los datos originales de los píxeles** para mantener la precisión del modelo.
- El sistema ofrece la estrategia de **pad** (relleno) específicamente para evitar la interpolación, asegurando que el modelo trabaje con la información pura de la imagen, aunque ocupe menos espacio en el área de inferencia.

### 3. Preservación de la precisión en máscaras

Un punto crítico donde la interpolación podría arruinar la precisión es en las **máscaras de segmentación**. Para evitarlo:

- Durante el mapeo de vuelta al cuadro completo, el sistema no utiliza interpolación lineal. En su lugar, emplea la interpolación de **vecino más cercano (`INTER_NEAREST`)**.
- Esto garantiza que los valores binarios de la máscara (que determinan si un píxel pertenece al objeto o no) no se alteren ni se "suavicen" en valores intermedios, lo cual destruiría la precisión de la segmentación.

En conclusión, los artefactos de interpolación se consideran una limitación aceptable en favor de una mayor densidad de píxeles, a menos que se trabaje con datos donde la fidelidad original sea más importante que el tamaño del objeto, en cuyo caso se debe optar por la estrategia de **pad**.


----

>¿Cómo se manejan las máscaras que quedan fuera de bordes?

El manejo de las máscaras que podrían quedar fuera de los bordes se gestiona en dos etapas principales: una **preventiva** durante la creación del recorte y otra de **ajuste** durante el mapeo de los resultados al cuadro completo.

A continuación se detalla el proceso según las fuentes:

### 1. Restricción de posición (Clamping)

Antes de que se realice la inferencia, el sistema aplica una técnica llamada **clamping** (sujeción) a las coordenadas del recorte.

- Incluso si el usuario o el sistema intentan definir una posición que se saldría de los límites del cuadro (por ejemplo, una coordenada X mayor al ancho de la imagen), el sistema ajusta automáticamente la posición al **valor máximo válido**.
- Esto garantiza que el recorte siempre esté contenido dentro de los límites físicos del cuadro, evitando de antemano que se generen máscaras en coordenadas inexistentes.

### 2. Inicialización de la máscara completa

Cuando llega el momento de mapear las detecciones de vuelta al cuadro completo, el sistema sigue estos pasos:

- Se crea una **máscara del tamaño total del cuadro**, inicializada completamente en `False` (o ceros).
- Esto asegura que cualquier área fuera de la región detectada se considere automáticamente como "sin detección".

### 3. Colocación y manejo de casos fronterizos

Una vez que la máscara del recorte ha sido redimensionada (en caso de que se haya usado la estrategia de _zoom_), se inserta en la máscara de tamaño completo:

- **Colocación por desplazamiento:** La máscara procesada se coloca en la posición exacta del desplazamiento (_offset_) calculada previamente.
- **Manejo de fronteras:** El sistema está diseñado específicamente para **manejar casos fronterizos** donde el recorte se extiende hasta el borde del cuadro. Dado que la posición inicial fue restringida mediante _clamping_, la inserción se realiza de forma que la máscara se alinee perfectamente con los límites del cuadro original sin intentar escribir datos fuera de la matriz de la imagen.

En resumen, gracias a que el origen del recorte se "sujeta" a límites válidos al inicio del proceso, las máscaras resultantes siempre tienen un espacio correspondiente dentro del cuadro completo donde pueden ser insertadas de manera segura.

