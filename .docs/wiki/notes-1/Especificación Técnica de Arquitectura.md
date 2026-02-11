# Especificación Técnica de Arquitectura: Framework Bakery

## 1. Introducción y Propósito del Sistema

El framework **Bakery** constituye una arquitectura modular de visión computacional diseñada para la orquestación de flujos de trabajo de alto rendimiento. En entornos de producción, la mera ejecución de modelos de Inteligencia Artificial es insuficiente; Bakery aborda la necesidad estratégica de integrar **inferencia dual** (segmentación de instancias y estimación de pose) con un motor de renderizado cinematográfico de grado profesional.

La implementación de referencia, **bakery-luna**, demuestra la viabilidad de este sistema mediante el uso de modelos YOLO especializados. La arquitectura de Bakery no es solo una elección organizativa, sino un requisito técnico para habilitar la ejecución determinista en hardware específico, optimizando el uso de **Intel iGPU con precisión FP16** mediante el runtime de **OpenVINO**. Este diseño desacoplado garantiza que el overhead de post-procesamiento estético no genere cuellos de botella en el bucle de ejecución primario, permitiendo una escalabilidad lineal y una mantenibilidad de largo plazo en sistemas de edge computing.

## 2. Arquitectura de Cinco Capas: Marco Estructural

Bakery se organiza bajo una arquitectura de capas estricta donde las dependencias fluyen exclusivamente hacia el núcleo (Core). Esta separación de responsabilidades minimiza el acoplamiento y permite la evolución independiente de la lógica de negocio frente a las implementaciones de hardware.

|   |   |   |   |
|---|---|---|---|
|Capa|Ubicación en Código|Responsabilidad Principal|Entidades y Salidas|
|**Core**|`bakery/core/entities/`|Definición del dominio y contratos de datos inmutables (DIP Contract).|`Frame`, `CropInfo`, `Segmentation`, `PoseEstimation`|
|**Adapters**|`bakery/adapters/openvino/`|Aislamiento de hardware y comunicación con el runtime externo.|`InferenceEngine`, Tensors raw (`[1, 84, N]`, etc.)|
|**Pipeline**|`bakery/pipeline/`|Orquestación del flujo, gestión de caché y lógica de ejecución dual.|`DualModelPipeline`, Entidades procesadas|
|**Presentation**|`bakery/annotators/`|Renderizado estilizado y experiencia visual (Decoupled UI).|`DisneyAnnotator`, `RenderConfig`|
|**Utils**|`bakery/utils/`|Funciones puras transversales, geometría y métricas.|`xywh2xyxy`, `nms`, `FPSCounter`|

Esta estructura asegura que el sistema sea altamente testable; la lógica del pipeline puede validarse sin acceso físico al hardware de Intel, utilizando mocks de los adaptadores que cumplan con el contrato definido en la capa Core.

## 3. Capa Core: Entidades de Dominio e Inmutabilidad (DDD)

La capa Core actúa como el lenguaje común y el contrato de integridad de todo el sistema. Bajo los principios de **Domain-Driven Design (DDD)**, Bakery trata al objeto `Frame` como el **Aggregate Root** de todos los resultados de inferencia, vinculando temporal y espacialmente cada detección a un punto específico del flujo de datos.

### Categorías de Entidades e Inmutabilidad

El uso de **frozen dataclasses** garantiza que las entidades sean objetos de valor inmutables, eliminando efectos secundarios durante la propagación de datos en pipelines de alta velocidad.

- **Frame (Aggregate Root):** Encapsula el array NumPy, `frame_id`, y dimensiones. Es la raíz a la que se anclan los resultados.
- **CropInfo:** Entidad crítica para el sistema Focus Lens. Almacena el origen de coordenadas (x, y), dimensiones del recorte y el scale\_factor. Sin esta entidad, la reconstrucción de coordenadas hacia el espacio del frame original sería matemáticamente imposible.
- **Detection Entities:**
    - `BoundingBox`: Coordenadas x1, y1, x2, y2 en formato pixel-space.
    - `Mask`: Datos de segmentación binaria o probabilística.
    - `Segmentation`: Agregado inmutable de cajas y máscaras.
- **Pose Entities:**
    - `KeyPoint`: Coordenadas y confianza individual.
    - `Skeleton`: Estructura canónica de 17 puntos basada en el estándar **COCO_KEYPOINT_NAMES**.
    - `PoseEstimation`: Agregado de todos los esqueletos detectados.

## 4. Capa de Adaptadores: Aislamiento de Hardware Intel OpenVINO

Bakery implementa el patrón **Adapter** para interactuar con el runtime de OpenVINO. Esto permite que el resto del framework permanezca agnóstico a las particularidades de la API de Intel, facilitando una futura migración a TensorRT u ONNX Runtime con un impacto nulo en el Pipeline.

### Validación y Repositorio de Modelos

El `ModelRepository` realiza un descubrimiento dinámico de archivos `.xml` y ejecuta una validación arquitectónica estricta antes de la carga:

- **Segmentation:** Requiere exactamente **2 salidas**. La primera para cajas/clases ([1, 84, N]) y la segunda para prototipos de máscaras ([1, 32, H, W]).
- **Pose:** Requiere **1 salida** con la estructura de tensores [1, 56, N].

El `InferenceEngine` gestiona el ciclo de vida de la compilación del modelo para el dispositivo seleccionado. Al centralizar aquí la dependencia de OpenVINO, Bakery asegura que la lógica de post-procesamiento (conversión de tensores a entidades del Core) sea la única "traducción" necesaria para mantener la integridad del dominio.

## 5. Capa de Pipeline: Orquestación e Inferencia Dual

El `DualModelPipeline` actúa como el motor de ejecución, coordinando la ejecución de los modelos de segmentación y pose. Su diseño se centra en maximizar el throughput mediante tres estrategias de optimización:

1. **Smart Segmentation Scheduling:** Dado que la segmentación es computacionalmente densa y la coherencia espacial de los objetos es alta, el sistema utiliza `seg_interval` (ej. valor de 5). Esto reduce las llamadas de inferencia de segmentación en un **40%**, reutilizando resultados previos mientras la pose se ejecuta en cada frame.
2. **Preprocessing Cache:** Cuando los modelos comparten resolución de entrada (ej. ambos 640x640), el pipeline reutiliza el tensor preprocesado. Esto reduce el costo de normalización y letterboxing en un **50%**.
3. **Focus Lens Integration:** Implementa inferencia en regiones de interés mediante estrategias de **"zoom"** (escalado para llenar la resolución del modelo) o **"pad"** (letterboxing para preservar la escala original).

**Impacto en Rendimiento:** La combinación de estas optimizaciones resulta en una mejora de hasta el **60% en la velocidad de ejecución** total en hardware Intel iGPU de consumo.

## 6. Capa de Presentación: Estética Cinematográfica

El `DisneyAnnotator` implementa un sistema de renderizado multicapa que desacopla la estética visual de la lógica de inferencia. El objetivo técnico es transformar datos matemáticos brutos en una representación visual coherente inspirada en la técnica de _Roger Rabbit_.

### Sistema de Renderizado de 8 Capas

El pipeline de renderizado sigue una secuencia estricta para garantizar profundidad y claridad:

1. **Capa 1 (Fondo):** Imagen original desaturada y oscurecida (60% darkness).
2. **Capas 2-4 (Segmentación):** Máscaras de instancia, bordes de máscaras y bounding boxes en full color.
3. **Capas 5-6 (Pose):** Conexiones del esqueleto (limbs) y puntos clave (keypoints) con colores diferenciados.
4. **Capa 7 (Spotlight):** Aplicación de una máscara de brillo en la región definida por Focus Lens o detecciones activas.
5. **Capa 8 (UI/Metrics):** Superposición de metadatos de rendimiento (FPS, eficiencia de caché).

Esta separación permite que el renderizado se desactive totalmente para procesamiento por lotes (batch) o se modifique el estilo estético sin alterar los pesos de los modelos ni la lógica de inferencia.

## 7. Capa de Utilidades y Gestión de Dependencias Inversas

La capa de **Utils** provee el soporte matemático transversal a través de funciones puras. Estas funciones son stateless, facilitando la testabilidad unitaria (con una suite actual de **126 pruebas documentadas**).

### Componentes de Soporte

- **Geometría:** Implementaciones de `xywh2xyxy`, cálculo de `bbox_iou` y `nms` (Non-Maximum Suppression) para la eliminación de redundancias.
- **Focus Lens Mapping:** Funciones que utilizan `CropInfo` para mapear coordenadas desde el espacio recortado (x, y local) de vuelta al aggregate root (`Frame`).
- **Métricas:** El `FPSCounter` emplea una ventana de suavizado para reportar métricas de rendimiento estables.

Bakery cumple estrictamente con el **Dependency Inversion Principle (DIP)**: el Core define el "Contrato" de datos. Los módulos de alto nivel (Pipeline) dependen de estas abstracciones, mientras que los detalles (Adapters) se inyectan en tiempo de ejecución, permitiendo que el framework sea verdaderamente agnóstico al hardware.

## 8. Conclusión: Filosofía de Diseño y Estado de Producción

Bajo la filosofía **"Complejidad por diseño, no por accidente"**, Bakery evita el sobre-ingeniería común en frameworks de investigación como Detectron2, enfocándose en la eficiencia de inferencia en el edge.

Actualmente en su **Fase 1.6 Luna**, el framework es un sistema listo para producción, optimizado para procesar video en tiempo real en hardware Intel de consumo con una sofisticación visual inexistente en soluciones como MediaPipe. La arquitectura garantiza que, a medida que los modelos de visión evolucionen, Bakery podrá integrarlos manteniendo su integridad estructural y su diferenciada estética cinematográfica.