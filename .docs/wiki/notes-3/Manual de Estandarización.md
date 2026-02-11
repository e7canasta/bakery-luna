# Manual de Estandarización: Integración de Modelos YOLO en Bakery-Luna

## 1. Marco Operativo y Alcance del Sistema

Dentro del ecosistema `bakery-luna`, el subsistema de gestión de modelos (`ModelManagement`) no es opcional; es el primer componente inicializado en el pipeline de ejecución (según `run_luna.py`). Este subsistema actúa como el guardián de integridad del sistema, operando como el núcleo de descubrimiento, validación y configuración. La estandarización es la única garantía de estabilidad operativa: sin un cumplimiento riguroso de estas normas, el pipeline de inferencia no puede compilar los motores de ejecución, lo que detendría el procesamiento de flujos de video en tiempo real.

Basándose en la arquitectura de la clase `ModelRepository`, el sistema asume las siguientes responsabilidades críticas:

- **Descubrimiento Automatizado:** Rastreo recursivo e idempotente del sistema de archivos para identificar archivos `.xml` válidos.
- **Validación de Formato:** Verificación estricta de que el modelo cumpla con las especificaciones de salida de YOLO v8 (segmentación o pose).
- **Extracción de Metadatos:** Obtención de dimensiones de entrada, formas de salida y precisión para la configuración dinámica del hardware.

**Impacto Operativo:** La validación automatizada elimina la posibilidad de fallos en tiempo de ejecución (`runtime errors`) por incompatibilidad de tensores. Este enfoque de "fallo temprano" asegura que solo los modelos conformes entren al pipeline, optimizando la labor del ingeniero de ML al garantizar que el sistema es resiliente y predecible. La conformidad comienza con la estructura física de los archivos en el almacenamiento.

--------------------------------------------------------------------------------

## 2. Arquitectura de Archivos y Formatos Requeridos

El sistema exige exclusivamente el formato **OpenVINO Intermediate Representation (IR)**. Es mandatorio mantener la paridad absoluta entre la topología y los pesos del modelo para que el motor de inferencia pueda instanciar el grafo de cómputo.

### Requisitos de Archivos

|   |   |   |
|---|---|---|
|Tipo de Archivo|Propósito|Regla de Nomenclatura|
|**.xml**|Descripción de la topología de la red y el grafo de cómputo.|Debe compartir el nombre base exacto con el archivo `.bin`.|
|**.bin**|Pesos binarios y parámetros entrenados del modelo.|Debe compartir el nombre base exacto con el archivo `.xml`.|

### Configuración de la Capa de Entrada (XML)

El archivo XML debe definir una capa de entrada que cumpla con los siguientes requisitos técnicos innegociables:

- **Nombre de la Capa:** Debe ser estrictamente `images`.
- **Forma (Shape):** Formato `[batch, channels, height, width]`. Ejemplo: `[1, 3, 640, 640]`.
- **Tipo de Dato:** Debe ser `float32`. **Advertencia:** El uso de `int8` o `float16` dentro de la definición del XML (incluso si los pesos están cuantizados) puede provocar errores de alineación en el preprocesador actual.

**Impacto Operativo:** Esta estructura permite que el motor OpenVINO compile los modelos de forma autónoma. La paridad de nombres y la definición precisa de la capa `images` eliminan la necesidad de intervención manual o archivos de configuración externos. Cualquier desviación en el nombre de la entrada impedirá que el pipeline de preprocesamiento mapee los tensores, resultando en el rechazo del modelo.

--------------------------------------------------------------------------------

## 3. Especificaciones Técnicas de Salida por Tipo de Modelo

Para asegurar la compatibilidad con los adaptadores de `bakery-luna`, los modelos YOLO v8 deben adherirse a estructuras de salida rígidas. El sistema rechazará cualquier modelo cuya arquitectura de tensores no coincida con los estándares de segmentación o pose.

### Modelo de Segmentación (YOLO Segmentation)

Debe producir **exactamente 2 salidas**:

- **Output 0 (Detecciones):** Tensor con forma `[batch, 84, N]`. Los 84 canales se desglosan en: 4 para coordenadas bbox, 1 de confianza y 79/80 para puntuaciones de clase. Para una resolución de 640x640, el valor de `N` es típicamente **1792**.
- **Output 1 (Máscaras):** Tensor con forma `[batch, 32, 80, 80]`, que contiene los prototipos de máscara.

### Modelo de Pose (YOLO Pose)

Debe producir **exactamente 1 salida**:

- **Output 0 (Combinada):** Tensor con forma `[batch, 56, N]`. Para una resolución de 640x640, `N` es típicamente **21942**.
- **Mapa de Canales (56 canales):**
    - **Canales 0-3:** Coordenadas de la caja delimitadora (x, y, w, h).
    - **Canal 4:** Confianza del objeto.
    - **Canales 5-55:** 17 puntos clave (keypoints) con 3 valores cada uno (x, y, visibilidad).

**Impacto Operativo:** Estas estructuras son el "estándar de oro" para el post-procesamiento. El sistema utiliza el conteo y la forma de las salidas para asignar los algoritmos de decodificación correctos. Cualquier variación (ej. un modelo con 3 salidas) provocará el rechazo inmediato para evitar corrupciones de memoria.

--------------------------------------------------------------------------------

## 4. Convenciones de Nomenclatura y Detección de Atributos

`bakery-luna` utiliza una estrategia de **"convención sobre configuración"**. La clasificación de los modelos es automática y se basa en patrones de nombres **case-insensitive** (insensibles a mayúsculas/minúsculas).

### Reglas de Clasificación (ModelType)

|   |   |   |
|---|---|---|
|Tipo de Modelo|Patrones Requeridos (Case-Insensitive)|Ejemplo de Nombre|
|**Segmentación**|`seg_`, `-seg`, `_seg`, `seg.`, `yolo-seg`|`V8-Base-Seg.xml`|
|**Pose**|`pose_`, `-pose`, `_pose`, `pose.`, `yolo-pose`|`Human_Pose_v8.xml`|

### Detección de Precisión (FP16 vs FP32)

La precisión se infiere de la ruta del directorio o del nombre del archivo:

- **FP16:** Si la ruta o el nombre contienen la cadena `fp16`.
- **FP32:** Valor predeterminado si no se detecta la marca de FP16.

**Impacto Operativo:** El cumplimiento de estas convenciones permite un despliegue "plug-and-play". Un ingeniero puede integrar nuevos modelos simplemente respetando la nomenclatura, eliminando la necesidad de editar archivos de configuración pesados y reduciendo el error humano en entornos de producción.

--------------------------------------------------------------------------------

## 5. Flujo de Descubrimiento, Validación y Manejo de Errores

El `ModelRepository` ejecuta un proceso sistemático para transformar archivos en disco en objetos `ModelConfig` operativos.

### Algoritmo de Descubrimiento

1. **Escaneo Recursivo:** Localización de todos los archivos `.xml` mediante `Path.rglob`.
2. **Validación de Formato:** Carga mediante OpenVINO y verificación de que el conteo de salidas sea 1 o 2.
3. **Detección de Tipo:** Aplicación de patrones de nombre (seg/pose).
4. **Inspección de Arquitectura:** Verificación de las dimensiones de los tensores de salida (Shapes).
5. **Extracción de Metadatos:** Generación del objeto `ModelConfig`.

### Escenarios de Error y Comportamiento

|   |   |
|---|---|
|Escenario de Error|Comportamiento del Sistema|
|**Conteo de salidas erróneo**|Modelo rechazado; se omite del repositorio.|
|**Non-YOLO Architecture**|Si las formas de los tensores son inválidas, el modelo se descarta.|
|**Archivo .bin faltante**|Fallo de carga en OpenVINO; el modelo se marca como inválido.|
|**Tipo desconocido**|Si no hay patrones `seg` o `pose`, el modelo es ignorado.|
|**Sin modelos válidos**|`discover_models()` devuelve una lista vacía; el pipeline de Luna se detiene.|

**Impacto Operativo:** La resiliencia del sistema permite omitir modelos experimentales corruptos sin detener la ejecución global, siempre que existan modelos válidos para las tareas requeridas. Esto asegura la continuidad operativa en entornos de despliegue continuo.

--------------------------------------------------------------------------------

## 6. Entidad ModelConfig y Metadatos de Inferencia

El objeto `ModelConfig` es el contrato final que recibe el `InferenceEngine` para la compilación del modelo en OpenVINO.

### Propiedades Críticas

- **Resolution:** Valor entero extraído de las dimensiones espaciales de la entrada `images`. **Nota Crítica:** El sistema asume una **resolución cuadrada** (ej. 640x640) para este cálculo.
- **ModelType & Precision:** Determinan la ruta de post-procesamiento y el hardware optimizado.
- **Shapes:** Metadatos de tensores utilizados para validar la compatibilidad antes de la inferencia.

**Impacto Operativo:** La propiedad `resolution` es vital para la optimización del `DualModelPipeline`. Si el modelo de segmentación y el de pose comparten la misma resolución, el sistema activa la caché de preprocesamiento. Esto permite procesar la imagen de entrada **una sola vez** para ambos modelos, reduciendo el overhead de CPU en un **50%** y maximizando los FPS.

--------------------------------------------------------------------------------

## 7. Lista de Verificación para el Ingeniero de ML (Checklist)

Antes de mover un modelo a producción, asegúrese de cumplir con cada punto:

- [ ] **Par de archivos:** ¿Existen los archivos `.xml` y `.bin` con nombres base idénticos?
- [ ] **Entrada 'images':** ¿La capa de entrada se llama exactamente `images`?
- [ ] **Resolución Cuadrada:** ¿El modelo fue exportado con dimensiones cuadradas (ej. 640x640)?
- [ ] **Tipo de dato:** ¿La entrada está definida como `float32`?
- [ ] **Nombres de salida:** ¿Las salidas tienen los friendly names `output0` (y `output1` para seg)?
- [ ] **Estructura YOLO:**
    - ¿Segmentación: 2 salidas con formas `[1, 84, N]` y `[1, 32, 80, 80]`?
    - ¿Pose: 1 salida con forma `[1, 56, N]`?
- [ ] **Patrón de nombre:** ¿El nombre incluye `seg` o `pose` (ej. `yolo8n-seg.xml`)?
- [ ] **Marca FP16:** Si el modelo es FP16, ¿el nombre o la ruta contienen la cadena `fp16`?

El cumplimiento de esta lista garantiza una integración exitosa en el pipeline `Luna` y la máxima eficiencia en el uso de recursos de hardware.