# Protocolo de Integración: Subsistema de Gestión de Modelos (Arquitectura Luna)

## 1. Fundamentos y Alcance del Repositorio de Modelos

En la Arquitectura Luna, la clase `ModelRepository` actúa como la **primera línea de defensa** del sistema. No es un simple cargador de archivos; es el componente crítico de abstracción que impone un contrato de integridad de datos antes de que cualquier activo alcance el motor de ejecución. Este subsistema funciona como una puerta de validación estricta diseñada para prevenir corrupciones de memoria y fallos catastróficos en el backend de C++ de OpenVINO, asegurando que solo los modelos que cumplen con las especificaciones técnicas de Luna sean admitidos en el pipeline.

### Responsabilidades Estratégicas

El repositorio centraliza la lógica de descubrimiento y saneamiento, desglosando sus funciones en cuatro pilares operativos:

|   |   |   |
|---|---|---|
|Responsabilidad|Método Principal|Objetivo Estratégico|
|**Descubrimiento**|`discover_models()`|Escaneo recursivo para identificar activos `.xml` y sus correspondientes `.bin`.|
|**Validación**|`validate_model()`|Aplicación del contrato técnico YOLO (conteo de salidas y topología de tensores).|
|**Extracción**|`extract_metadata()`|Interrogación de la estructura del modelo para derivar formas, resolución y precisión.|
|**Búsqueda**|`get_model_by_type()`|Recuperación selectiva por tipo; devuelve el **primer** match o `None` si no existe.|

### Valor Arquitectónico ("So What?")

La centralización de estas tareas elimina la deuda técnica derivada de configuraciones manuales y rutas estáticas. Al validar la estructura de salida y la integridad de los archivos en la fase de inicialización, Luna garantiza la estabilidad del motor de inferencia en tiempo de ejecución, permitiendo que los desarrolladores se enfoquen en la lógica de negocio sin preocuparse por la compatibilidad de archivos individuales.

Este flujo operativo comienza con el rastreo físico del almacenamiento, transformando una estructura de archivos desorganizada en un catálogo de activos verificados.

--------------------------------------------------------------------------------

## 2. Procedimiento de Descubrimiento Recursivo y Organización de Directorios

La escalabilidad de un sistema de visión depende de su capacidad para adaptarse a topologías de almacenamiento dinámicas. El `ModelRepository` implementa un algoritmo de descubrimiento basado en `Path.rglob("*.xml")`, lo que permite una flexibilidad total en la organización de los activos, ya sea para despliegues locales simples o estructuras complejas de versionado.

### Topologías Soportadas

El sistema identifica automáticamente modelos en tres configuraciones principales:

1. **Estructura Plana:** Todos los archivos `.xml` y `.bin` residen en la raíz del directorio.
2. **Jerarquía por Precisión:** Modelos organizados en subcarpetas (ej. `fp16/`, `fp32/`).
3. **Jerarquía por Tarea:** Organización por función del modelo (ej. `segmentation/`, `pose/`).

### Optimización de Hardware y Despliegue ("So What?")

El soporte para escaneo recursivo y la detección de rutas permite que el sistema Luna apunte dinámicamente a diferentes objetivos de hardware. Por ejemplo, al desplegar versiones **FP16** y **FP32** simultáneamente, el sistema puede priorizar modelos de media precisión para GPUs o NPUs y modelos de precisión completa para CPUs sin intervención manual en el código. Esto facilita las pruebas A/B de rendimiento y la optimización de recursos según el entorno de ejecución.

Una vez localizado un activo, el protocolo exige una verificación exhaustiva de su estructura interna para garantizar la compatibilidad con los adaptadores de Luna.

--------------------------------------------------------------------------------

## 3. Criterios de Validación de Modelos y Especificaciones YOLO

La validación es el mecanismo de control de calidad que garantiza que el flujo de post-procesamiento reciba tensores con la dimensionalidad esperada. Luna impone un contrato estricto basado en las arquitecturas YOLO v8.

### Especificaciones Técnicas de Salida

Para que un modelo sea aceptado, sus capas de salida (que deben incluir el atributo `friendly_name` como `output0` u `output1`) deben seguir estas reglas:

|   |   |   |
|---|---|---|
|Tipo de Modelo|Salidas|Estructura del Tensor y Composición de Canales|
|**Pose**|1|`[batch, 56, anchors]`: 4 coords (bbox) + 1 conf + 51 puntos clave (17x3).|
|**Segmentation**|2|**Out 0:** `[batch, 84, anchors]` (4 coords + 80 class scores). <br> **Out 1:** `[batch, 32, H, W]` (32 prototipos de máscara).|

### Protocolo de Rechazo y Exclusión

El sistema descartará automáticamente cualquier activo que presente:

- Extensión distinta a `.xml` o ausencia del archivo `.bin` con el mismo nombre base.
- Modelos con 0 salidas (exportación fallida) o 3+ salidas (arquitectura no soportada).
- Incapacidad de carga por parte del motor OpenVINO (archivo corrupto).

### Estabilidad del Post-procesamiento ("So What?")

Este nivel de rigor asegura que los algoritmos de decodificación de cajas y máscaras operen sobre estructuras predecibles. Al validar que el modelo de segmentación tiene exactamente 84 canales en su primera salida (4 de coordenadas y 80 de clases), se previenen errores de acceso a memoria y se asegura la integridad de la clasificación en el pipeline de producción.

Tras confirmar la validez técnica, el sistema procede a la auto-configuración dinámica mediante la extracción de metadatos.

--------------------------------------------------------------------------------

## 4. Extracción de Metadatos y Clasificación Heurística

El sistema Luna se auto-configura interrogando directamente la topología del modelo. La extracción automatizada elimina la necesidad de archivos de configuración externos propensos a errores humanos.

### Metadatos Críticos y Detección de Tipo

El método `extract_metadata()` recupera el `input_shape` (requerido: **float32** en formato `[batch, channels, height, width]`), los `output_shapes`, la resolución espacial y la precisión. La clasificación se rige por un contrato de nomenclatura (case-insensitive):

- **Segmentation:** Patrones `seg_`, `-seg`, `_seg`, `seg.`, o `yolo-seg`.
- **Pose:** Patrones `pose_`, `-pose`, `_pose`, `pose.`, o `yolo-pose`.
- **Precisión:** Se detecta `FP16` si la cadena "fp16" existe en la ruta; de lo contrario, se asume `FP32`.

### Eficiencia en el Pipeline Dual ("So What?")

La resolución extraída (típicamente 640x640) es la clave para la optimización del caché de pre-procesamiento. Dado que tanto el modelo de Pose como el de Segmentación suelen derivar de la misma base, Luna detecta si comparten resolución para ejecutar una única operación de redimensionamiento y normalización. Este "pipeline dual" reduce significativamente la carga computacional y mejora la latencia general.

Estos metadatos se consolidan en la entidad `ModelConfig`, que sirve como el contrato final para la inicialización del motor de inferencia.

--------------------------------------------------------------------------------

## 5. Integración en el Pipeline de Inferencia Dual y Flujo de Trabajo

La inicialización del sistema es un proceso orquestado donde el `ModelRepository` suministra la configuración necesaria para la compilación de modelos en el hardware de destino.

### Flujo de Ejecución (Basado en `run_luna.py`)

1. **Fase de Descubrimiento:** El repositorio escanea el directorio raíz y genera objetos `ModelConfig`.
2. **Recuperación Selectiva:** El sistema solicita un modelo de cada tipo mediante `get_model_by_type()`. Si un tipo falta, el sistema aborta la ejecución para mantener la integridad del pipeline dual.
3. **Sincronización de Parámetros:** Se inyectan umbrales de confianza y resolución detectada.
4. **Compilación del Engine:** Los objetos `ModelConfig` se entregan al `InferenceEngine` para la optimización en el hardware de destino (CPU/GPU/NPU).
5. **Instanciación del Pipeline:** Los motores compilados se integran en el `DualModelPipeline`.

### El Contrato `ModelConfig` ("So What?")

La entidad `ModelConfig` no es solo un contenedor; es el manual de instrucciones para el hardware. Al incluir la precisión detectada, permite que OpenVINO active rutas de ejecución optimizadas para tipos de datos específicos, maximizando el rendimiento del silicio disponible y garantizando que el pipeline de inferencia sea eficiente y estable.

--------------------------------------------------------------------------------

## 6. Diagnóstico, Manejo de Errores y Verificación de Integración

Para asegurar la fiabilidad en entornos de producción, el subsistema implementa una estrategia de manejo de errores que prioriza la continuidad del descubrimiento sobre la interrupción del sistema.

### Escenarios de Error y Estrategias

|   |   |   |
|---|---|---|
|Escenario de Error|Detección|Estrategia de Manejo|
|**Directorio Vacío**|`discover_models()`|Retorna lista vacía; el sistema principal emite error y sale.|
|**Nombre No Estandarizado**|Heurística de patrones|El modelo se omite (no se puede determinar el tipo).|
|**Arquitectura No-YOLO**|Inspección de salidas|Rechazo preventivo si hay 0 o 3+ salidas.|
|**Fallo de Carga .xml/.bin**|`ov.Core.read_model()`|Registro del fallo y continuación con el siguiente activo.|

### Verificación mediante Mocks (`opset10`)

La fiabilidad del repositorio se valida mediante pruebas BDD que utilizan modelos sintéticos creados con `opset10` de OpenVINO. Esto permite verificar la lógica de descubrimiento y validación en entornos de CI/CD sin necesidad de distribuir archivos binarios de modelos reales, asegurando que el contrato de integración se mantenga intacto tras cada cambio en el código.

### Checklist de Requisitos Clave para Integradores

Para que un modelo sea aceptado con éxito por la Arquitectura Luna, debe cumplir:

- [ ] **Emparejamiento:** Archivos `.xml` y `.bin` con nombres de base idénticos.
- [ ] **Capa de Entrada:** Nombre exacto `images`, tipo `float32`, tensor 4D `[1, 3, H, W]`.
- [ ] **Salidas de Segmentación:** Exactamente 2 salidas con `friendly_name` (`output0`, `output1`).
- [ ] **Salidas de Pose:** Exactamente 1 salida con `friendly_name` (`output0`).
- [ ] **Nomenclatura:** Inclusión de patrones de tipo (`seg`, `pose`) y precisión (`fp16` si aplica) en el nombre o ruta.
- [ ] **Formato YOLO:** Salida 0 de segmentación con 84 canales (4 coord + 80 clases) y salida de pose con 56 canales.

El cumplimiento estricto de este protocolo garantiza una transición sin fricciones desde el entrenamiento hasta la ejecución optimizada en el ecosistema Luna.