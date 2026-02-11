# Guía Visual de Entidades: Los Bloques de Construcción de Bakery

## 1. Introducción: ¿Qué son las Entidades de Dominio?

En el fascinante mundo de la Visión por Computadora, las **entidades** son "paquetes de información" estructurados que el sistema utiliza para interpretar el mundo. Imagina que Bakery es un artista digital: para transformar un video crudo en una escena con la estética de **"Disney/Roger Rabbit"**, no puede simplemente procesar píxeles aislados. Necesita organizar lo que ve en conceptos con significado.

El objetivo final es crear una composición artística de **8 capas**, donde los personajes resaltan en color sobre un fondo que se oscurece automáticamente al **60% de oscuridad** en blanco y negro. Para lograr este nivel de detalle, Bakery utiliza entidades por tres razones fundamentales:

- **Organización:** Permite que la detección, la pose y el renderizado compartan una "gramática" común.
- **Inmutabilidad:** Una vez creada una entidad, no cambia. Esto garantiza que los datos sean predecibles y evita errores "fantasma" durante el procesamiento.
- **Claridad:** Para un mentor o un desarrollador, es mucho más natural trabajar con un objeto llamado `Skeleton` que con una lista cruda de números decimales.

_Todo este viaje de transformación comienza con la captura del lienzo original: el Frame._

--------------------------------------------------------------------------------

## 2. El Lienzo Digital: Entidades de Frame (Cuadros)

El punto de partida es la entidad `Frame`. A diferencia de una simple imagen, un `Frame` es una captura con "identidad propia", vinculada a un momento específico del video. Junto a él, encontramos el `CropInfo`, que actúa como una "lupa inteligente" para optimizar el trabajo de la Inteligencia Artificial.

### Comparativa: El Todo frente a la Parte

|   |   |   |
|---|---|---|
|Entidad|Propósito|Atributos Clave|
|**Frame**|Representa la imagen completa del video.|`data` (píxeles), `frame_id`, `width`, `height`.|
|**CropInfo**|Metadatos de un recorte o "zoom" táctico.|`frame_id`, `x`, `y` (origen), `width`, `height`, `scale_factor`.|

### El "Focus Lens": Zoom vs. Pad

El `CropInfo` es el alma de la función **Focus Lens**. Esta herramienta permite que el sistema se concentre solo en lo importante (por ejemplo, una persona en el centro) procesándolo en una resolución menor para ganar velocidad. Para manejar esto, Bakery usa dos estrategias:

1. **Zoom:** Escala la imagen para llenar el área de enfoque, evitando bordes negros.
2. **Pad:** Mantiene la escala original y añade bordes negros, evitando distorsiones por interpolación.

Gracias al atributo `scale_factor` (factor de escala), el sistema puede realizar un mapeo de coordenadas: traduce los resultados del recorte pequeño de vuelta al lienzo grande con precisión matemática.

_Con el lienzo definido, es momento de que el sistema identifique qué objetos habitan en él._

--------------------------------------------------------------------------------

## 3. Detectando el Mundo: Entidades de Detection (Segmentación)

Bakery necesita distinguir entre el fondo y los protagonistas. Aquí es donde entran las entidades de detección, que nos dicen "dónde está" algo y "qué forma" tiene.

**BoundingBox (Caja delimitadora):** El rectángulo (formato xyxy) que encierra al objeto. Define su ubicación general y su `class_id` (qué tipo de objeto es).

**Mask (Máscara):** La silueta precisa píxel por píxel. Es el "recorte de tijera" que permite aplicar color al personaje mientras el fondo se queda en blanco y negro.

**Segmentation:** El contenedor maestro que agrupa todas las cajas y máscaras de un mismo cuadro (`frame_id`).

### Limpieza Inteligente: El Proceso NMS e IoU

A veces, la IA detecta el mismo objeto varias veces. Para limpiar esto, Bakery usa el **NMS (Non-Maximum Suppression)** bajo la regla del **IoU (Intersection over Union)**:

1. **La Regla del Solape (IoU):** El sistema mide qué tanto se enciman dos cajas. Si el IoU (el porcentaje de superposición) es mayor a 0.5, se consideran duplicados.
2. **La Analogía del Mentor:** Imagina a varias personas apuntando con el dedo al mismo objeto. No necesitamos diez dedos señalando; solo nos quedamos con "el dedo que apunta con más firmeza" (la detección con mayor puntaje de confianza).
3. **Resultado:** El sistema elimina las señales más débiles, dejando una detección única y limpia por objeto.

_Cuando el objeto detectado es un humano, el sistema busca un nivel de detalle superior: su postura._

--------------------------------------------------------------------------------

## 4. El Lenguaje del Cuerpo: Entidades de Pose (Esqueletos)

Para capturar el movimiento con fluidez cinematográfica, Bakery traduce el cuerpo humano a un lenguaje de puntos y conexiones.

- **KeyPoint:** Un punto anatómico específico (nariz, codo, tobillo) con su ubicación (x, y) y un **score de confianza** (de 0.0 a 1.0).
- **Skeleton:** La colección de puntos de una sola persona.
- **PoseEstimation:** El conjunto de todos los esqueletos detectados en el cuadro.

Bakery utiliza el formato **COCO**, el estándar universal de la industria que identifica **17 puntos clave**. Usar COCO es como hablar el "idioma oficial" de la visión artificial, asegurando que el sistema reconozca siempre la misma estructura humana.

### Pasos para construir el esqueleto:

1. **Localizar:** Identificar cada uno de los 17 puntos (ej. hombro izquierdo).
2. **Filtrar:** Si la confianza es baja (ruido), el punto se ignora para evitar esqueletos "saltarines".
3. **Conectar:** Agrupar los puntos válidos en la entidad `Skeleton`.
4. **Consolidar:** Guardar todos los esqueletos del cuadro en una `PoseEstimation`.

--------------------------------------------------------------------------------

## 5. El Cerebro del Sistema: Entidades de Configuration

Nada de lo anterior funcionaría sin reglas claras. Las entidades de configuración dictan cómo se comporta la IA y cómo se gestionan los recursos de hardware (CPU/GPU).

### Configuraciones Críticas

|   |   |   |
|---|---|---|
|Configuración|Parámetro Clave|Impacto del Valor|
|**ModelConfig**|`confidence`|**Bajo:** Detecta todo, incluso errores. **Alto:** Solo detecta lo que es muy obvio.|
|**PipelineConfig**|`seg_interval`|Controla cada cuántos cuadros se busca "qué objetos hay".|
|**RenderConfig**|`bw_darkness`|Define el nivel de oscuridad del fondo (Estándar: 0.6 o 60%).|

### Smart Scheduling (Programación Inteligente)

La segmentación es una tarea pesada. Gracias al parámetro `seg_interval`, Bakery solo busca objetos nuevos cada **N** cuadros, mientras que la pose se calcula en cada cuadro para mantener la fluidez del movimiento.

**Ejemplo con** `**seg_interval = 5**`**:**

- **Cuadro 1:** [Segmentación + Pose] -> _Esfuerzo máximo de la IA._
- **Cuadros 2-4:** [Pose] -> _Ahorra energía usando la máscara del Cuadro 1._
- **Cuadro 5:** [Segmentación + Pose] -> _Actualiza la información de los objetos._

--------------------------------------------------------------------------------

## 6. Resumen: Del Dato a la Estética "Roger Rabbit"

Finalmente, el `DisneyAnnotator` toma todas estas entidades y las organiza en su sistema de **8 capas** de renderizado:

- **Capa 1:** El fondo original transformado a blanco y negro con un **60% de oscuridad**.
- **Capas 2 a 8:** Superposición de segmentaciones a color, máscaras precisas, esqueletos de 17 puntos y el efecto de "spotlight" o reflector que da profundidad a la escena.