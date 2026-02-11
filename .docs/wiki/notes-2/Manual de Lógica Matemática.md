# Manual de Lógica Matemática: Mapeo de Coordenadas en el Sistema Focus Lens

Este manual ha sido diseñado para guiarte a través de los fundamentos matemáticos y arquitectónicos del sistema **Focus Lens**. Como ingenieros, no solo buscamos que el código funcione, sino comprender la armonía entre el preprocesamiento de imágenes y la geometría de las detecciones.

--------------------------------------------------------------------------------

### 1. Introducción al Concepto de Focus Lens

El sistema **Focus Lens** es una técnica de optimización que concentra la "densidad de píxeles" en una región de interés (ROI). En lugar de obligar al modelo a buscar una aguja en un pajar (un objeto pequeño en un cuadro de 4K), recortamos el "pajar" para que el objeto ocupe la mayor parte de la resolución de entrada del modelo.

**Beneficios Clave para el Aprendizaje:**

- **Resolución Efectiva:** Al ampliar una zona pequeña para que llene el tamaño de entrada del modelo (ej. 640px), permitimos que las capas convolucionales extraigan características mucho más ricas y detalladas.
- **Reducción de Ruido:** Eliminamos quirúrgicamente el fondo irrelevante, lo que reduce drásticamente los falsos positivos y la carga cognitiva del modelo.
- **Eficiencia Computacional:** Al enfocar el modelo en un área específica, podemos utilizar arquitecturas más precisas sin el castigo de procesar píxeles vacíos o redundantes.

**Transición Pedagógica:** Una vez que el modelo realiza una detección dentro de este "universo de recorte", las coordenadas resultantes son locales. El verdadero reto matemático —y el corazón de este manual— es el **mapeo inverso**: cómo traducir esas coordenadas locales de vuelta a las coordenadas globales de la imagen original.

--------------------------------------------------------------------------------

### 2. Configuración: Las Reglas del Juego (FocusLensConfig)

Antes de mover un solo píxel, debemos definir las reglas de nuestra transformación mediante el objeto `FocusLensConfig`.

|   |   |   |   |
|---|---|---|---|
|Parámetro|Tipo|Valor por Defecto|Regla de Validación Lógica|
|`focus_size`|`int`|(Requerido)|Debe ser positivo y **múltiplo de 80**.|
|`focus_x`|`int?`|`None`|No negativo. Si es `None`, activa `is_centered`.|
|`focus_y`|`int?`|`None`|No negativo. Si es `None`, activa `is_centered`.|
|`strategy`|`str`|`"zoom"`|Debe ser `"zoom"` (escala) o `"pad"` (relleno).|

**La Regla del Múltiplo de 80:** ¿Por qué 80? Las redes neuronales como YOLO utilizan capas de reducción progresiva llamadas _strides_ (normalmente hasta 32 o 80). Si el tamaño de la imagen no es divisible por estos factores, los "engranajes" matemáticos del modelo no encajarán, provocando errores de alineación en las capas finales.

**El Concepto de Clamping (Limitación):** El sistema aplica una regla de seguridad: ninguna coordenada de recorte puede estar fuera de la imagen. Si intentas recortar en la posición 2000 de una imagen de 1920px, el sistema "sujeta" (_clamping_) el recorte al borde máximo permitido.

--------------------------------------------------------------------------------

### 3. La Anatomía del Recorte (CropInfo)

Al ejecutar el recorte, el sistema genera `CropInfo`, que actúa como el "mapa del tesoro" para el viaje de regreso.

**Metadatos de Transformación (CropInfo):**

- **x, y:** El origen del recorte en el espacio de la imagen (tras el posible escalado).
- **width, height:** Dimensiones finales del recorte (siempre equivalentes a `focus_size`).
- **scale_factor:** El multiplicador crítico que define el cambio de escala.

**La Clave del Scale Factor:** En la estrategia `"zoom"`, si la imagen es más pequeña que el `focus_size`, debemos calcular cuánto agrandarla: scale = \frac{focus\_size}{\min(frame\_width, frame\_height)} Este factor es el puente entre el **Espacio Local** (el recorte) y el **Espacio Global** (la imagen original).

--------------------------------------------------------------------------------

### 4. El Corazón Matemático: La Lógica del Remapeo

Para llevar una detección desde el recorte al cuadro completo (_Full Frame_), aplicamos una transformación lineal inversa en dos pasos:

#### Paso 1: Inversión de Escala (Normalización)

Si aplicamos un zoom, las coordenadas detectadas están en un mundo "agrandado". Debemos dividirlas por el factor de escala para devolverlas a la magnitud original.

#### Paso 2: Aplicación del Desplazamiento (Offset Desescalado)

Sumamos la posición del recorte (`crop_x`), pero atención: como `crop_x` se midió en el espacio escalado, también debe ser desescalado.

**Fórmulas Comparativas:**

|   |   |   |
|---|---|---|
|Escenario|Estrategia|Fórmula de Remapeo Final|
|**Sin Escala**|`pad`|CoordenadaCrop + Offset|
|**Con Escala**|`zoom`|(CoordenadaCrop / Scale) + (Offset / Scale)|

_Nota: Esta lógica asegura que, incluso con zooms agresivos, la detección aterrice exactamente sobre el objeto real en la imagen original._

--------------------------------------------------------------------------------

### 5. Aplicación Práctica por Tipo de Detección

Diferentes tipos de datos requieren diferentes manipulaciones vectoriales para mantener la precisión:

- **Cajas de Delimitación (Bounding Boxes):** Se aplica la fórmula de forma **vectorizada**. Usando NumPy, actualizamos los ejes `[x1, x2]` y `[y1, y2]` simultáneamente (slicing `[:, [0, 2]]`), lo que garantiza un rendimiento óptimo en tiempo real.
- **Máscaras de Segmentación:** Al ser matrices, el proceso es más visual:
    1. Calculamos las dimensiones originales del recorte: orig\_w = focus\_size / scale.
    2. Redimensionamos la máscara detectada a esas dimensiones usando `INTER_NEAREST` (fundamental para no crear píxeles borrosos; solo queremos 0 o 1).
    3. Colocamos la máscara resultante sobre un lienzo negro del tamaño completo de la imagen.
- **Puntos Clave (Keypoints):** Se transforman de forma similar a las cajas, pero con una regla semántica inquebrantable.

:::warning **Insight Crítico: El Significado del (0,0)** En los modelos de estimación de pose, una coordenada **(0,0)** es una convención que indica que el punto es "invisible" o no detectado. **Nunca** apliques el desplazamiento a estos puntos. Si lo haces, los puntos invisibles aparecerán erróneamente en la esquina del recorte, rompiendo la lógica del modelo. :::

--------------------------------------------------------------------------------

### 6. Consideraciones de Rendimiento y Errores Comunes

Como especialistas, debemos anticipar los cuellos de botella:

1. **Presión de Memoria en Zoom:** La estrategia `zoom` crea una copia escalada de la imagen antes de recortar. En zooms agresivos, esto puede duplicar momentáneamente el uso de memoria RAM por cada hilo de ejecución.
2. **Error de Interpolación:** Usar `INTER_LINEAR` para el zoom inicial es correcto para detectar, pero para las **máscaras** se debe usar obligatoriamente `INTER_NEAREST` para preservar la integridad binaria de la segmentación.
3. **Desalineación de Máscaras:** El error más común es intentar colocar la máscara en la imagen original sin desescalar primero sus dimensiones internas. Esto resulta en máscaras que "flotan" fuera del objeto.

--------------------------------------------------------------------------------

### 7. Resumen Visual del Flujo de Ejecución (Pipeline Flow)

Utiliza esta lista de verificación para asegurar que tu implementación de Focus Lens sea matemáticamente sólida:

1. **Validar Configuración:** Confirmar que `focus_size` sea múltiplo de 80.
2. **Calcular Escala:** Si la imagen es pequeña y usas `zoom`, define el `scale_factor`.
3. **Aplicar Clamping:** Asegurar que el origen del recorte (x, y) no exceda los límites del frame.
4. **Ejecutar Inferencia:** Procesar el recorte en el espacio local.
5. **Desescalar Coordenadas:** Dividir las detecciones por el `scale_factor`.
6. **Trasladar (Offset):** Sumar la posición del recorte desescalada.
7. **Preservar Semántica:** Filtrar puntos (0,0) en keypoints antes de desplazar.
8. **Proyectar:** Dibujar resultados en el lienzo global de la imagen original.

Dominar esta lógica matemática es lo que separa a un programador de un ingeniero de visión artificial capaz de optimizar sistemas en entornos de alta exigencia.