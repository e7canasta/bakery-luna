# Especificación Técnica: Arquitectura Modular y Patrones de Diseño del Framework Bakery

## 1. Introducción y Fundamentos de Diseño

El framework **Bakery** constituye una arquitectura de referencia para la inferencia de visión artificial de alto rendimiento, diseñada específicamente para mitigar el acoplamiento técnico en sistemas que dependen de aceleración por hardware. En el desarrollo de sistemas de tiempo real, la dependencia directa de SDKs de proveedores (como Intel OpenVINO) suele generar una deuda técnica que asfixia la evolución del software. Bakery resuelve este desafío mediante una **arquitectura de cinco capas**, segregando rígidamente la lógica de dominio de los detalles de implementación del hardware.

La implementación **Bakery-Luna** ejemplifica esta filosofía al orquestar tareas concurrentes de segmentación de instancias y estimación de pose humana. El sistema no solo busca la precisión analítica, sino que persigue una estética visual inspirada en la técnica _Disney/Roger Rabbit_, donde los elementos detectados en color interactúan con un entorno estilizado. Esta separación de responsabilidades garantiza que la orquestación de modelos sea agnóstica al motor de ejecución, permitiendo una mantenibilidad a largo plazo en un ecosistema de hardware en constante cambio. La robustez del sistema reside en su base conceptual, lo que nos lleva a explorar la Capa de Core.

--------------------------------------------------------------------------------

## 2. Capa de Core: El Modelo de Dominio Inmutable

La **Capa de Core** es el cimiento estratégico del framework. Diseñada bajo principios de **Domain-Driven Design (DDD)**, esta capa es totalmente independiente de dependencias externas y actúa como la "única fuente de verdad" del sistema. Al definir entidades como **Value Objects** inmutables, garantizamos la integridad de los datos durante transformaciones críticas, como el mapeo de coordenadas desde espacios de recorte (_crops_) al cuadro original.

### Evaluación de Entidades y Tipos de Dominio

El sistema utiliza una jerarquía de tipos estricta y validada mediante enumeraciones técnicas (`ModelSize`, `ModelType`, `Device`, `Precision`). La inmutabilidad previene efectos secundarios en el flujo de datos concurrente.

|   |   |   |   |
|---|---|---|---|
|Categoría|Entidades Clave|Rol Arquitectónico (Aggregate Root)|Evaluación de Diseño|
|**Frame**|`Frame`, `CropInfo`|**Aggregate Root**|Encapsula la matriz NumPy y metadatos. El `frame_id` vincula todo el ciclo de vida.|
|**Detection**|`BoundingBox`, `Mask`, `Segmentation`|Child de `Frame`|Representa resultados de YOLO Seg con coordenadas normalizadas.|
|**Pose**|`KeyPoint`, `Skeleton`, `PoseEstimation`|Child de `Frame`|Estructura los 17 puntos COCO (`COCO_KEYPOINT_NAMES`) para análisis biomecánico.|
|**Config**|`ModelConfig`, `PipelineConfig`, `FocusLensConfig`|N/A|Centraliza parámetros operativos con validación _fail-fast_.|

### Restricciones y Jerarquía de Datos

Para garantizar la estabilidad, ciertas entidades imponen restricciones técnicas rigurosas:

- **FocusLensConfig:** El parámetro `focus_size` debe ser obligatoriamente un **múltiplo de 80**, permitiendo una alineación óptima con los requisitos de los modelos YOLO.
- **Atributos de Frame:** Incluye `data`, `frame_id`, y dimensiones intrínsecas, sirviendo como el punto de origen para cualquier operación de inferencia.
- **Skeletons:** Cumplen con el formato canónico COCO de 17 puntos clave, garantizando compatibilidad con estándares de la industria.

Estas abstracciones definen el lenguaje universal del sistema antes de pasar a la capa que interactúa directamente con los recursos de computación.

--------------------------------------------------------------------------------

## 3. Capa de Adapters: Abstracción de Hardware y Patrón Adapter

La **Capa de Adapters** actúa como el aislante tecnológico que protege al Core de las especificidades de **Intel OpenVINO**. Implementando el patrón **Adapter**, esta capa traduce los contratos del dominio en llamadas a la API de bajo nivel, permitiendo que Bakery sea agnóstico respecto al motor de inferencia.

### Componentes de Interacción y Contratos Técnicos

1. **ModelRepository:** Gestiona el descubrimiento dinámico mediante un escaneo recursivo de archivos `.xml`. Utiliza convenciones de nombres (ej. patrones que contengan "seg" o "pose") para identificar modelos. Su función crítica es extraer metadatos (resolución, precisión FP16/FP32) sin compilar el modelo, evitando la penalización de **1 a 3 segundos** de carga inicial durante el descubrimiento.
2. **InferenceEngine:** Un _wrapper_ sobre el runtime de OpenVINO que produce **tensores crudos** a partir de entidades `Frame`. Estos tensores se validan estrictamente contra las firmas del modelo:
    - **Segmentación:** Requiere 2 salidas; la salida 0 de cajas debe tener forma `[1, 84, N]`.
    - **Pose:** Requiere 1 salida con forma `[1, 56, N]`.

Este desacoplamiento permite que, ante un cambio hacia TensorRT u ONNX Runtime, solo se deba sustituir el adaptador, manteniendo el resto del sistema intacto. Una vez que el hardware está adaptado, se requiere una lógica superior para orquestar la ejecución.

--------------------------------------------------------------------------------

## 4. Capa de Pipeline: Orquestación y Optimización de Flujos

El **DualModelPipeline** es el cerebro operativo encargado de transformar tensores crudos en **Entidades de Dominio** mediante una orquestación concurrente eficiente. Su prioridad es maximizar el rendimiento mediante el uso inteligente de recursos.

### Pilares de Optimización Crítica

El pipeline implementa tres estrategias que definen su superioridad técnica:

- **Smart Segmentation Scheduling:** La segmentación es costosa y los objetos suelen tener estabilidad temporal. El parámetro `seg_interval` (por defecto 5) permite ejecutar la segmentación cada _N_ cuadros, reduciendo la carga computacional en un **factor de 5x**, mientras la pose se estima en cada cuadro para capturar movimientos rápidos.
- **Preprocessing Cache:** Si los modelos de segmentación y pose comparten la misma resolución de entrada (ej. 640x640), el pipeline reutiliza los tensores preprocesados. Esta eliminación de operaciones redundantes de _letterboxing_ y normalización resulta en una **ganancia de rendimiento de ~30%**.
- **Focus Lens Integration:** Permite procesar regiones de interés (ROI) mediante recortes. Al reducir la resolución efectiva procesada, se incrementa la precisión y la velocidad, mapeando automáticamente los resultados de vuelta a las coordenadas globales del cuadro original.

Los resultados procesados por el pipeline deben ser transformados en representaciones visuales, lo que introduce la capa de presentación.

--------------------------------------------------------------------------------

## 5. Capa de Presentation: Visualización y Estética Disney/Roger Rabbit

La **Capa de Presentation** separa la lógica de renderizado de la inferencia, permitiendo que la estética evolucione sin riesgo de regresiones en la lógica de detección. El componente central es el **DisneyAnnotator**, un motor de visualización avanzado.

### El Pipeline de Renderizado de 8 Capas

Para lograr la estética de interacción entre realidad y animación, el sistema utiliza un flujo de 8 capas:

1. **Fondo (Layer 1):** Imagen original convertida a blanco y negro con un **60% de oscuridad** para eliminar ruido visual.
2. **Overlays de Color (Layers 2-8):** Aplicación selectiva de color sobre el fondo oscuro:
    - Segmentaciones de instancias y máscaras binarias (Layer 2-3).
    - Bounding Boxes y esqueletos de pose de 17 puntos (Layer 4-5).
    - Efecto de brillo en la lente de enfoque (Layer 6).
    - Efecto de spotlight sobre objetos detectados (Layer 7).
    - Metadatos y métricas de rendimiento (Layer 8).

Esta separación mejora la interpretabilidad de los datos, permitiendo al operador humano validar resultados de IA de forma instantánea. Para que estas capas funcionen armónicamente, se requiere de herramientas matemáticas ubicadas en la capa de utilidades.

--------------------------------------------------------------------------------

## 6. Capa de Utils: Funciones Transversales y Geometría

La **Capa de Utils** es el soporte técnico transversal. Proporciona los algoritmos estadísticos y geométricos necesarios para la precisión del sistema, sin contaminar la lógica de negocio.

### Operaciones Geométricas y Métricas

- **Geometría:** Incluye funciones vitales como `xywh2xyxy` para la normalización de formatos YOLO y el algoritmo **NMS (Non-Maximum Suppression)**. Cabe destacar que `nms` depende intrínsecamente de la métrica `bbox_iou` (Intersection over Union) para eliminar redundancias espaciales.
- **PerformanceMetrics:** Monitorea la eficiencia del pipeline, calculando ratios como `total_frames / seg_runs`.
- **FPSCounter:** Utiliza un **buffer circular** de 30 cuadros para proporcionar una medición suavizada y estable de los cuadros por segundo, evitando fluctuaciones ruidosas en el reporte de rendimiento.

Estas utilidades actúan bajo el principio de inversión de dependencias que cohesiona todo el sistema.

--------------------------------------------------------------------------------

## 7. Inversión de Dependencias y Extensibilidad Futura

La arquitectura Bakery se fundamenta en el **Principio de Inversión de Dependencias (DIP)**. Los módulos de alto nivel (Pipeline, Annotator) nunca dependen de las implementaciones de bajo nivel (OpenVINO); ambos dependen de las abstracciones definidas en el Core.

### Estrategia de Extensibilidad

Esta estructura es el garante de la escalabilidad. El `DualModelPipeline` es agnóstico a _cómo_ se generó el tensor (sea mediante OpenVINO, TensorRT u ONNX Runtime). Para integrar un nuevo motor:

1. Se crea un nuevo **Adapter** que cumpla con el contrato de `InferenceEngine`.
2. Se asegura que el nuevo adaptador retorne las entidades del Core (`Segmentation`, `PoseEstimation`).
3. El impacto en el resto del sistema es **cero**, ya que el Pipeline y el Annotator consumen las mismas entidades inmutables.

### Conclusión de Mantenibilidad

La separación física en directorios y la exposición selectiva de la API mediante archivos `__init__.py` protegen al sistema contra la deuda técnica. Bakery no es solo un framework de visión; es una infraestructura diseñada para durar, donde la elegancia del código y el rendimiento de hardware convergen en una solución profesional y escalable.