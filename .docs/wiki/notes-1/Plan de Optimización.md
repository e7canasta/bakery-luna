# Plan de Optimización de Rendimiento: Framework Bakery-Luna

## 1. Contexto Estratégico y Arquitectura del Sistema

En el despliegue de sistemas de visión computacional de alta exigencia, la ejecución simultánea de modelos de segmentación de instancias y estimación de pose representa un cuello de botella crítico para el procesamiento en tiempo real. El framework **bakery-luna** mitiga este impacto mediante una arquitectura de cinco capas (Core, Adapters, Pipeline, Presentation y Utils) que desacopla la lógica de negocio de la gestión de recursos de bajo nivel. Esta separación de preocupaciones permite que el sistema mantenga una alta modularidad sin sacrificar la eficiencia, facilitando la implementación de estrategias de optimización que se integran directamente en el flujo de inferencia.

### Análisis de la Arquitectura de Capas

El rendimiento del sistema está blindado por la interacción entre las capas de **Adapters** y **Pipeline**. La capa de _Adapters_ encapsula las dependencias de Intel OpenVINO mediante el patrón _Adapter_, aislando el `InferenceEngine` y el `ModelRepository`. Un aspecto estratégico fundamental es que el `InferenceEngine` compila los modelos para **CPU por defecto**; cualquier cambio hacia dispositivos como GPU o VPU requiere una re-compilación deliberada para optimizar los kernels de ejecución. Por su parte, la capa de _Pipeline_ aplica el principio de **Dependency Inversion**, permitiendo que el orquestador dependa de entidades abstractas (`Frame`, `Segmentation`, `PoseEstimation`) y no de implementaciones de hardware específicas, asegurando que las optimizaciones de velocidad no comprometan la estabilidad del núcleo.

### Propósito del Dual-Model Inference

La carga computacional de ejecutar simultáneamente YOLO-seg y YOLO-pose es asimétrica. Mientras que la segmentación requiere el procesamiento de máscaras y prototipos de alta resolución (típicamente en dos capas de salida), la estimación de pose se centra en 17 puntos clave (COCO format) en una sola capa de salida. Sin una orquestación inteligente, la redundancia en el preprocesamiento y la ejecución continua de ambos modelos degradarían los FPS a niveles no aptos para aplicaciones industriales o interactivas.

## 2. Optimización mediante Smart Segmentation Scheduling

La estabilidad temporal es la premisa básica de la optimización en video: en una secuencia de cuadros, la morfología y posición de los objetos segmentados cambian de forma gradual. El sistema capitaliza esta redundancia para reducir los ciclos de inferencia sin afectar la percepción visual del usuario final.

### Mecánica del Parámetro `seg_interval`

A través del parámetro `seg_interval`, el `DualModelPipeline` implementa una programación asincrónica de tareas. Mientras que la estimación de pose se ejecuta en cada cuadro para capturar movimientos humanos rápidos con precisión quirúrgica, la segmentación se invoca únicamente cada N cuadros. Durante los intervalos de reposo, el pipeline suspende la llamada al `seg_engine`, liberando el bus de datos y los recursos de la VPU/GPU.

### Impacto en la Eficiencia Computacional

El ajuste del `seg_interval` produce una reducción lineal inmediata en la carga del procesador.

- **Escenario** `**seg_interval = 1**`**:** Ejecución continua; consumo máximo de recursos y latencia base.
- **Escenario** `**seg_interval = 5**`**:** Reducción del **80%** en las ejecuciones de segmentación. El ahorro de ciclos de cómputo permite redistribuir la potencia hacia el renderizado de la estética Disney o hacia una mayor tasa de cuadros por segundo globales.

### Gestión de Resultados Cachados

Para mantener la continuidad visual, el pipeline gestiona el estado a través de `_cached_segmentation`. En los cuadros donde no se realiza inferencia, el sistema sirve el último resultado válido almacenado. Esto garantiza que las máscaras y bounding boxes se mantengan visibles mientras el motor de inferencia se concentra exclusivamente en la actualización de los esqueletos de pose.

## 3. Eficiencia de Memoria: Preprocessing Cache

El preprocesamiento de imágenes (letterboxing, normalización y conversión de espacio de color) suele ejecutarse de forma redundante cuando múltiples modelos analizan el mismo frame. En `bakery-luna`, evitar esta duplicidad es crítico para minimizar la latencia _glass-to-glass_.

### Sincronización de Resoluciones y Reutilización de Tensores

El `PreprocessCache` se activa automáticamente cuando el modelo de segmentación y el de pose comparten la misma resolución de entrada (ej. 640x640). En este escenario, el primer modelo realiza las operaciones de transformación y almacena el resultado en un objeto `ov.Tensor`. Este tensor se mantiene en **formato NCHW** y **precisión FP32**, permitiendo que el segundo modelo lo consuma directamente de la memoria sin repetir un solo ciclo de cálculo.

### Comparativa de Operaciones de Preprocesamiento

|   |   |   |
|---|---|---|
|Operación de Pipeline|Flujo sin Caché (Redundante)|Flujo con Caché (Optimizado)|
|**Resize / Letterboxing**|Ejecución 2x|**Ejecución 1x**|
|**Conversión de Color**|Ejecución 2x|**Ejecución 1x**|
|**Normalización (Media/Desv)**|Ejecución 2x|**Ejecución 1x**|
|**Formateo NCHW (FP32)**|Ejecución 2x|**Ejecución 1x**|
|**Asignación de Memoria**|Tensors independientes|**Memoria compartida**|

Esta eliminación del overhead de OpenVINO reduce drásticamente la latencia de entrada, permitiendo una ejecución más fluida incluso en hardware con ancho de banda de memoria limitado.

## 4. Optimización de Inferencia Selectiva con Focus Lens

El sistema de **Focus Lens** aplica el concepto de Región de Interés (ROI) para concentrar el presupuesto computacional en las áreas más relevantes del cuadro, permitiendo aumentar la precisión en objetos distantes o reducir la resolución efectiva para ganar velocidad.

### Estrategias de Procesamiento y Restricciones Técnicas

El `FocusLensConfig` define dos estrategias principales gestionadas por la utilidad `apply_focus_lens`:

- **Estrategia** `**zoom**`**:** Recorta la ROI y la escala a la resolución del modelo, maximizando el detalle.
- **Estrategia** `**pad**`**:** Mantiene la escala original y rellena los bordes, evitando artefactos de interpolación. **Nota Crítica de Arquitectura:** Por diseño del sistema de validación, el parámetro `focus_size` **debe ser obligatoriamente un múltiplo de 80** para asegurar la compatibilidad con las operaciones de stride de los modelos YOLO.

### Dinámica de Coordenadas y Mapping

Dado que la inferencia ocurre sobre un recorte, los resultados deben transformarse para alinearse con el cuadro completo. El sistema utiliza el objeto `CropInfo` para ejecutar una **inversión de escala y traslación**. Las utilidades `map_detections_to_full_frame` y `map_keypoints_to_full_frame` realizan el cálculo matemático necesario para proyectar bounding boxes, máscaras y los 17 puntos clave del esqueleto de vuelta al espacio de coordenadas original del video.

## 5. Métricas de Rendimiento y Evaluación de Resultados

La optimización en tiempo real requiere una observabilidad total. `bakery-luna` integra la clase `PerformanceMetrics` para auditar la eficacia de las estrategias implementadas a través de KPIs granulares.

### Análisis de KPIs de Eficiencia

- **Seg Efficiency Ratio:** Calculado como `total_frames / seg_runs`. Valida que el _Smart Scheduling_ está operando según el intervalo configurado.
- **Average FPS:** El indicador definitivo de la capacidad de respuesta del sistema bajo carga de trabajo sostenida.
- **Inference Consistency:** Se deriva de la lógica de ventana suavizada (smoothing window) del `FPSCounter`. Utiliza un buffer circular (típicamente de 30 cuadros) para medir la estabilidad de los tiempos de procesamiento, filtrando picos aislados de latencia para ofrecer una métrica de fluidez real.

### Perfiles de Configuración Estratégica

|   |   |   |   |
|---|---|---|---|
|Perfil|`seg_interval`|`focus_size`|Optimización Clave|
|**Máximo Rendimiento**|10|480|Agresivo frame-skipping y baja resolución.|
|**Equilibrado**|5|640|Estándar operativo para la mayoría de hardware Intel.|
|**Alta Precisión**|1|Desactivado|Inferencia full-frame sin caching de segmentación.|

## 6. Conclusiones y Hoja de Ruta de Implementación

La sinergia entre el **Smart Scheduling**, el **Preprocessing Caching** y el **Focus Lens** convierte a `bakery-luna` en una arquitectura de alto rendimiento capaz de procesar flujos complejos con una latencia mínima. Este enfoque estratégico permite que el hardware Intel alcance su máximo potencial, equilibrando la carga entre CPU y aceleradores.

### Mandato de Implementación

Para aplicar este plan de optimización en nuevos entornos, siga estos pasos críticos:

1. **Validación de Modelos en** `**ModelRepository**`**:** Verifique que los modelos YOLO posean la estructura de salida requerida (2 salidas para segmentación, 1 salida para pose).
2. **Alineación de Tensores:** Configure ambos modelos con la misma resolución nativa para forzar el uso del `PreprocessCache` (NCHW/FP32).
3. **Calibración de Focus Lens:** Seleccione un `focus_size` que sea múltiplo de 80 según el área crítica de la escena.
4. **Ajuste de Intervalos:** Establezca un `seg_interval` inicial de 5 y monitorice el `Seg Efficiency Ratio`.
5. **Auditoría de Desempeño:** Valide la consistencia de la inferencia mediante el `FPSCounter` para asegurar que el sistema mantiene la fluidez requerida por la estética de renderizado final.