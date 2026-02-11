# Guía Maestra de Metadatos: El Corazón de la Inferencia en Bakery-Luna

## 1. Introducción al Concepto de Metadatos en Inferencia

En el ecosistema de `bakery-luna`, los metadatos no son simplemente "datos sobre datos"; son el **manual de instrucciones esencial** que el sistema debe leer y comprender antes de intentar realizar cualquier tarea de visión computacional. Sin estos metadatos, el motor de inferencia no sabría qué tamaño de imagen esperar, cuántas detecciones procesar o si el modelo está diseñado para segmentar objetos o detectar posturas humanas.

La clase `ModelRepository` actúa como el arquitecto y bibliotecario inteligente del sistema: su trabajo es descubrir, validar y extraer esta información automáticamente para que el motor de OpenVINO pueda compilar la red de forma óptima.

**Definición de Alto Nivel:** Los metadatos de inferencia son el conjunto de parámetros técnicos (dimensiones, precisión y estructura de salida) extraídos de un modelo de OpenVINO que permiten configurar el flujo de trabajo (pipeline) de Inteligencia Artificial de manera automática, asegurando la compatibilidad entre el modelo y el motor de ejecución.

Las tres funciones principales de los metadatos son:

- **Descubrimiento:** Localizar archivos `.xml` válidos en directorios locales mediante un escaneo recursivo.
- **Validación:** Asegurar que el modelo cumple con las reglas estrictas de la arquitectura YOLO (rechazando formatos incompatibles).
- **Configuración:** Consolidar la información en objetos `ModelConfig` que dictan cómo se procesarán los datos en el mundo real.

_Para entender cómo se procesan estos datos, primero debemos mirar la "calidad" del pensamiento del modelo: su precisión._

--------------------------------------------------------------------------------

## 2. Precisión Técnica: FP16 vs. FP32

La **precisión** se refiere al formato numérico utilizado para representar los pesos y cálculos del modelo. En `bakery-luna`, el sistema detecta la precisión analizando la ruta del archivo (path). Esta detección es **insensible a mayúsculas** (por ejemplo, detectará tanto `fp16` como `FP16`).

¿Por qué importa? El uso de **FP16** (Half Precision) suele ofrecer un mejor rendimiento en hardware compatible, mientras que **FP32** (Single Precision) es el estándar de fidelidad completa. Si el sistema no encuentra un marcador específico en la ruta, optará por **FP32** como valor seguro por defecto.

|   |   |   |
|---|---|---|
|Tipo de Precisión|Marcador en la Ruta (Ejemplo)|Comportamiento del Sistema|
|**FP16**|`exports/fp16/yolo-seg.xml`|Detectado por el marcador `fp16` o `FP16`.|
|**FP32**|`models_fp32/pose_model.xml`|Detectado por el marcador `fp32` o `FP32`.|
|**FP32 (Default)**|`models/yolo-seg.xml`|Valor por defecto si no hay marcadores en la ruta.|

_La precisión define la calidad de los datos, pero la "forma" de estos datos es dictada por las dimensiones de entrada._

--------------------------------------------------------------------------------

## 3. Formas de Entrada (Input Shapes) y Resolución

El concepto de `input_shape` describe el formato exacto del "paquete" de píxeles que el modelo espera recibir. Este se extrae directamente de la capa de entrada del modelo y se representa como una tupla:

```python
(1, 3, 640, 640)
```

1. **Batch (1):** Cantidad de imágenes procesadas simultáneamente. El sistema está optimizado para inferencia en tiempo real de una sola imagen.
2. **Channels (3):** Indica los canales de color. El valor `3` corresponde al estándar RGB (Rojo, Verde, Azul).
3. **Height (640):** La altura de la imagen en píxeles.
4. **Width (640):** El ancho de la imagen en píxeles.

La **Resolución** es el valor escalar que el sistema extrae de estas dimensiones espaciales.

**Optimización de Arquitectura:** La resolución es crítica para la **optimización del caché de preprocesamiento**. Si posees dos modelos (ej. Segmentación y Pose) que comparten la misma resolución, el sistema puede reutilizar los tensores ya procesados, evitando cálculos redundantes y acelerando drásticamente el flujo de ejecución.

_Una vez que el modelo sabe qué tamaño de imagen recibirá, debe estructurar lo que "ve" en sus salidas._

--------------------------------------------------------------------------------

## 4. Arquitectura de Salida: Segmentación vs. Pose

`bakery-luna` valida la arquitectura del modelo analizando el conteo y la forma de sus tensores de salida (outputs).

### Modelo de Segmentación (Instance Segmentation)

Diseñado para identificar objetos y sus contornos.

- **Conteo de Salidas:** Requiere exactamente **2 salidas**.
- **Estructura del Tensor:**
    - _Output 0 (Boxes):_ `[batch, 84, N]`. Aquí, el valor **84** se compone de 4 coordenadas para la caja y **80 puntuaciones de clase**.
    - _Output 1 (Masks):_ `[batch, 32, H, W]`. Contiene los prototipos de máscaras para la segmentación.

### Modelo de Pose (Pose Estimation)

Diseñado para identificar puntos clave (keypoints) en una figura humana.

- **Conteo de Salidas:** Requiere exactamente **1 salida** combinada.
- **Estructura del Tensor:**
    - _Output 0 (Combinado):_ `[batch, 56, N]`. El valor **56** representa 4 coordenadas de caja, 1 de confianza y **51 valores de puntos clave** (17 puntos clave x 3 valores: x, y, visibilidad).

_Saber qué es el modelo es vital, pero el sistema necesita encontrar estos archivos automáticamente para funcionar._

--------------------------------------------------------------------------------

## 5. Identificación y Validación de Modelos

Para que el `ModelRepository` acepte un modelo, este debe superar un proceso de validación técnica. Si el modelo tiene **3 o más salidas**, el sistema lo rechazará automáticamente por considerarlo una **arquitectura No-YOLO**.

**Checklist de Validación para Desarrolladores:**

- [ ] **Extensión y Pareja de Archivos:** Debe ser un archivo `.xml` y existir un archivo `.bin` con el mismo nombre (el `.xml` define la **topología** o "cerebro", mientras que el `.bin` contiene los **pesos** o "conocimiento").
- [ ] **Patrón de Nombre:** El nombre del archivo o su ruta debe incluir uno de estos patrones (case-insensitive):
    - _Segmentación:_ `seg_`, `-seg`, `_seg`, `seg.` o `yolo-seg`.
    - _Pose:_ `pose_`, `-pose`, `_pose`, `pose.` o `yolo-pose`.
- [ ] **Capa de Entrada Estándar:** La capa de entrada debe llamarse `images` y utilizar un tipo de dato `float32`.
- [ ] **Conteo de Salidas:** Exactamente 1 para Pose o 2 para Segmentación.

_Con estos conceptos claros, estás listo para que el sistema realice la_ _**compilación del motor (Engine Compilation)**_ _y comience la inferencia._

--------------------------------------------------------------------------------

## 6. Resumen de Flujo: De Archivo a Configuración (ModelConfig)

Toda la información técnica se consolida en el objeto `ModelConfig`. Este objeto es el que utiliza el sistema durante la fase de "Configuration Update" para preparar la ejecución final.

|   |   |   |
|---|---|---|
|Propiedad de Metadatos|Origen (Fuente)|Uso en el Sistema|
|**Model Path (.xml)**|Escaneo de disco|Define la topología para `Core.read_model()`.|
|**Model Type**|Patrones de nombre|Determina si se usa post-procesamiento de Pose o Seg.|
|**Resolution**|Dimensiones de entrada|Optimiza el caché de tensores y preprocesamiento.|
|**Precision**|Marcador en la ruta|Informa sobre el rendimiento y capacidad del hardware.|
|**Confidence**|**Argumentos CLI**|Filtra las detecciones durante la ejecución.|

Comprender estos metadatos es lo que separa a un usuario de herramientas de un verdadero arquitecto de IA. Al dominar la relación entre la estructura del archivo y el motor de OpenVINO, tienes el poder de construir pipelines robustos, eficientes y profesionalmente escalables. ¡El éxito de la inferencia reside en la precisión de sus cimientos!