# Guía Comparativa: Modelos de Segmentación vs. Pose en bakery-luna

## 1. Introducción: El ADN de los Modelos de Visión

En la arquitectura de **bakery-luna**, el componente `ModelRepository` actúa como la capa de abstracción fundamental para la gestión de modelos. Su misión es orquestar el descubrimiento, validación y extracción de metadatos de archivos OpenVINO sin que el resto del sistema deba preocuparse por la ubicación física de los archivos.

Para un ingeniero de software, entender la diferencia entre los modelos de **Segmentación** y de **Pose** es crítico para asegurar la integridad de la inferencia. El sistema implementa una **identificación automática** basada en la estructura de los tensores de salida; este mecanismo no solo previene errores de ejecución (_runtime errors_), sino que permite el "hot-swapping" de modelos: la capacidad de intercambiar arquitecturas simplemente depositando archivos en una carpeta, permitiendo que el pipeline se reconfigure dinámicamente.

--------------------------------------------------------------------------------

## 2. La Regla de Oro: Identificación por Estructura de Salida

El sistema valida la identidad de un modelo analizando el número y la forma de sus tensores de salida. Esta es la huella digital que define si estamos ante una arquitectura YOLO de detección de puntos clave o de máscaras.

### Modelos de Pose (1 salida)

Están optimizados para la detección de esqueletos humanos en un solo paso. Combinan toda la información en un único tensor de **56 canales**:

- **Canales 0-3:** Coordenadas de la caja (Bounding Box).
- **Canal 4:** Confianza del objeto.
- **Canales 5-55:** 17 puntos clave (_keypoints_), donde cada punto contiene 3 valores: coordenada X, coordenada Y y el índice de visibilidad.

### Modelos de Segmentación (2 salidas)

Utilizan una arquitectura de doble flujo para separar la detección del objeto de la generación de su forma:

- **Salida 0 (Detección):** Un tensor de **84 canales**. Esto incluye las 4 coordenadas de la caja y **80 puntajes de clase** (correspondientes al estándar COCO).
- **Salida 1 (Máscaras):** Contiene los **32 canales de Prototipos de Máscaras**. La resolución espacial (H, W) de esta salida suele ser **1/8** de la resolución de entrada (por ejemplo, 80x80 para una entrada de 640x640), lo que permite una reconstrucción precisa de la instancia.

### Comparativa Visual de Tensores

```text
ESTRUCTURA DE SALIDAS (YOLO v8)

Pose (1 Salida):
└── Output 0: [batch, 56, N]  
    (4 bbox + 1 conf + 51 keypoints [17x3])

Segmentación (2 Salidas):
├── Output 0: [batch, 84, N]  
│   (4 bbox + 80 class scores)
└── Output 1: [batch, 32, H, W] 
    (32 Mask Prototypes @ 1/8 resolution)
```

--------------------------------------------------------------------------------

## 3. Patrones de Nomenclatura y Descubrimiento

El `ModelRepository` utiliza el método `Path.rglob("*.xml")` para realizar un escaneo recursivo en los directorios. Para que el sistema clasifique correctamente el modelo, el nombre del archivo debe seguir patrones específicos (case-insensitive).

|   |   |   |
|---|---|---|
|Tipo de Modelo|Patrones de búsqueda|Ejemplos de Archivos Válidos|
|**Segmentación**|`seg_`, `-seg`, `_seg`, `seg.`, `yolo-seg`|`yolo-seg.xml`, `run_seg_v2.xml`|
|**Pose**|`pose_`, `-pose`, `_pose`, `pose.`, `yolo-pose`|`human_pose.xml`, `yolo-pose-v8.xml`|

[!NOTE] **Insight del Arquitecto:** El sistema está diseñado para fallar de forma segura. Si un modelo no coincide con estos patrones de nombre, se omitirá durante el descubrimiento para evitar cargar configuraciones incompatibles en el motor de inferencia.

--------------------------------------------------------------------------------

## 4. Metadatos, Precisión y Optimización

Una vez validada la estructura, el sistema extrae metadatos técnicos que dictan el comportamiento del hardware y el preprocesamiento:

- `**input_shape**`: Define las dimensiones de entrada. **Requisito Crítico:** El nombre de la capa de entrada en el XML de OpenVINO debe ser obligatoriamente `images`.
- `**precision**`: El sistema busca el marcador `fp16` en la ruta de la carpeta o el nombre del archivo. Si no lo encuentra, el sistema **asume por defecto FP32**. Esto es vital para el balance entre latencia y precisión.
- `**resolution**`: Se deriva de las dimensiones espaciales de entrada.
    - _¿Por qué importa?_ El pipeline utiliza este valor para la **Optimización del Caché de Preprocesamiento**. Si dos modelos (Pose y Seg) comparten la misma resolución, el sistema reutiliza los tensores preprocesados, ahorrando ciclos de CPU/GPU significativos.

--------------------------------------------------------------------------------

## 5. Resumen de Validación y Criterios de Éxito

Para que un modelo sea aceptado por el ecosistema Luna, debe superar este checklist de arquitectura y formato.

### Criterios de Éxito vs. Fallo

|   |   |   |
|---|---|---|
|Característica|Criterio de Éxito|Criterio de Fallo (Rechazo)|
|**Dualidad de Archivos**|Par `.xml` y `.bin` con el mismo nombre base.|Falta el `.bin` o nombres no coinciden.|
|**Nombre de Capa Entrada**|Debe llamarse exactamente `images`.|Cualquier otro nombre (ej. `input_1`).|
|**Conteo de Salidas**|Exactamente 1 (Pose) o 2 (Segmentación).|0 salidas o 3+ salidas.|
|**Estructura de Canales**|56 canales para Pose; 84 para Seg.|Dimensiones de tensor no estándar.|
|**Nomenclatura**|Incluir etiquetas `seg` o `pose` en el archivo.|Nombres genéricos como `model.xml`.|
|**Precisión**|Carpeta o archivo con etiqueta `fp16`.|Sin etiqueta (Resulta en fallback a FP32).|

**Insight final:** Como diseñadores instruccionales, subrayamos que estas convenciones no son arbitrarias: son la **llave maestra** que permite que el `ModelRepository` desacople la complejidad del hardware del flujo de trabajo del desarrollador, garantizando estabilidad y alto rendimiento en el pipeline de visión.