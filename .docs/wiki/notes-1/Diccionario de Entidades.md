# Diccionario de Entidades: El Corazón del Sistema Bakery

**Descripción:** Una guía maestra para comprender el modelo de dominio y la arquitectura de software detrás de la inferencia de visión artificial.

--------------------------------------------------------------------------------

## 1. Introducción al Modelo de Dominio

En el diseño de sistemas complejos, la **Capa Core** (o Núcleo) no es solo una carpeta de archivos; es el motor conceptual que define la realidad del software. En **Bakery**, esta capa actúa como un "Hardware-Agnostic Heart", un corazón independiente del hardware que permite que la lógica de negocio permanezca pura, sin importar si usamos OpenVINO, TensorRT o ONNX.

Siguiendo el principio de **Inversión de Dependencias (Dependency Inversion)**, todos los componentes (como el motor de inferencia o el anotador visual) dependen de estas abstracciones centrales. Aquí, las entidades se diseñan como **Objetos de Valor (Value Objects)**: representaciones precisas, inmutables y validadas que garantizan que los datos que fluyen por el pipeline sean siempre íntegros.

"El diseño orientado al dominio (DDD) nos enseña que el software debe ser un reflejo fiel de la realidad. En Bakery, las entidades capturan la esencia de lo que el sistema 've' y 'entiende', protegiendo la integridad de la información desde que entra como un fotograma hasta que sale como una obra de arte digital."

--------------------------------------------------------------------------------

## 2. El Concepto de Inmutabilidad: ¿Por qué nada cambia?

Para un arquitecto de software, la mutabilidad es la fuente de la mayoría de los errores en sistemas de alto rendimiento. Por ello, las entidades de Bakery son **"frozen dataclasses"** (clases de datos congeladas). Una vez creadas, no pueden ser alteradas.

Esta solidez nos permite aplicar **Invariantes del Dominio** mediante el método `__post_init__`. Por ejemplo, un sistema nunca permitirá que un `BoundingBox` se cree con un ancho negativo. Los beneficios son tres:

- **Validación "Fail-Fast":** El sistema detecta errores de datos en el momento de la creación. Si una coordenada es inválida, el sistema falla de inmediato, evitando que el error se propague por el pipeline de inferencia.
- **Seguridad en hilos (Thread-safety) implícita:** Al ser inmutables, los objetos pueden compartirse entre hilos de procesamiento sin necesidad de bloqueos o semáforos, ya que nadie puede modificar el estado interno.
- **Prevención de efectos secundarios:** Tienes la garantía absoluta de que pasar una detección al `DisneyAnnotator` no alterará las coordenadas originales que otro proceso podría estar utilizando.

**Transición:** _Ahora que entendemos por qué estas piezas son sólidas y fijas, exploremos la base de todo proceso: la imagen original._

--------------------------------------------------------------------------------

## 3. Entidades de Imagen: El Punto de Partida

Todo el flujo de datos nace de la captura de luz. En Bakery, esto se estructura en dos entidades críticas que gestionan el escenario de visión.

|   |   |   |
|---|---|---|
|Entidad|Atributos Clave|Propósito en el sistema|
|**Frame**|`data`, `frame_id`, `width`, `height`|Es el **Aggregate Root** (Raíz del Agregado). Representa el fotograma original y vincula todos los resultados de inferencia a través de su `frame_id`.|
|**CropInfo**|`x`, `y`, `width`, `height`, `scale_factor`|Almacena los metadatos del "Focus Lens". Es el puente matemático necesario para remapear coordenadas de un espacio de inferencia recortado al espacio del frame original.|

El **"Focus Lens"** es nuestra técnica de optimización principal. Al procesar solo una región de interés (ROI), ganamos velocidad. Sin la entidad `CropInfo`, las detecciones realizadas en el recorte estarían "perdidas" geográficamente; esta entidad permite que el sistema sepa exactamente a qué píxeles del mundo real corresponde cada detección.

**Transición:** _Una vez que tenemos el escenario (el Frame), el sistema debe identificar qué objetos hay en él mediante entidades de detección._

--------------------------------------------------------------------------------

## 4. Entidades de Detección: Identificando Objetos

Cuando el sistema procesa un tensor crudo, lo transforma en entidades estructuradas. Aquí, el formato estándar es `**xyxy**`, que utiliza **coordenadas de píxeles absolutas** (`x1, y1` para el punto superior izquierdo y `x2, y2` para el inferior derecho).

### Jerarquía de Detección

La entidad **Segmentation** actúa como el contenedor maestro (Aggregate) que organiza la información de la siguiente manera:

1. **BoundingBox:** Define la ubicación. Incluye `x1, y1, x2, y2`, un `confidence` (0.0 a 1.0) y un `class_id` (vital para que el anotador sepa qué color asignar, por ejemplo, a una "persona").
2. **Mask:** Una máscara binaria que define la silueta exacta del objeto píxel por píxel.
3. **Segmentation (El Contenedor):** Agrupa todas las cajas y máscaras de un `frame_id`.

**Nota de Arquitectura:** Antes de finalizar el agregado `Segmentation`, el sistema aplica **NMS (Non-Maximum Suppression)**. Este proceso utiliza las cajas de texto para eliminar detecciones duplicadas, asegurando que cada objeto físico sea representado por una única entidad en el dominio.

**Transición:** _Pero Bakery no solo ve objetos; también comprende la compleja estructura del movimiento humano._

--------------------------------------------------------------------------------

## 5. Entidades de Pose: La Estructura Humana

Para capturar la dinámica humana, Bakery utiliza una jerarquía de pose basada en el formato estándar **COCO**. A diferencia de la segmentación, que puede espaciarse mediante el parámetro `seg_interval` para ahorrar recursos, la pose se calcula **en cada fotograma** debido a la rapidez del movimiento humano.

### Relación Jerárquica y Cardinalidad

- **KeyPoint (17 unidades):** La unidad mínima. Contiene `x`, `y`, `confidence` y su `name` semántico (ej. "nose", "left_elbow").
- **Skeleton (1 por persona):** Una colección de exactamente 17 `KeyPoint` organizados según la constante `COCO_KEYPOINT_NAMES`.
- **PoseEstimation (1 por frame):** El agregado que contiene todos los esqueletos detectados en el fotograma.

Este diseño permite que, aunque el sistema solo ejecute la segmentación cada 5 fotogramas (usando un caché), los esqueletos de las personas se muevan fluidamente en pantalla a 30 FPS o más.

**Transición:** _Estas piezas individuales no trabajan solas; se ensamblan para crear la experiencia visual final._

--------------------------------------------------------------------------------

## 6. Relaciones y Flujo: Del Dato a la Pantalla

El sistema culmina en el `DisneyAnnotator`, que utiliza las entidades para generar el efecto "Roger Rabbit": personajes vibrantes sobre fondos cinematográficos oscuros. El renderizado sigue un modelo estricto de **8 capas**:

|   |   |   |
|---|---|---|
|Capa|Entidad Consumida|Resultado Visual|
|**1**|`Frame`|Fondo original convertido a B/N y oscurecido (60% darkness).|
|**2-7**|`Mask` & `BoundingBox`|Segmentaciones en color completo. El `class_id` determina la paleta de colores.|
|**8**|`Skeleton`|Estructuras óseas dibujadas con precisión sobre las personas.|
|**Especial**|`FocusLensConfig`|Un efecto de "spotlight" que ilumina intensamente la zona de interés definida en `CropInfo`.|

--------------------------------------------------------------------------------

## 7. Conclusión para el Aprendiz

Dominar este diccionario de entidades es entender por qué Bakery es un sistema robusto. Al separar las entidades (el "qué") de los motores de inferencia (el "cómo"), logramos que el código sea escalable. Si decides cambiar el hardware o el modelo de IA, las entidades de `Segmentation` y `PoseEstimation` seguirán siendo el lenguaje universal que mantiene unido al sistema.

```text
CONCEPTOS CLAVE PARA RECORDAR:
1. CAPA CORE      = Corazón agnóstico al hardware y sin dependencias.
2. AGGREGATE ROOT = El Frame es la raíz que ancla todos los resultados.
3. INMUTABILIDAD  = Frozen dataclasses que previenen efectos secundarios.
4. COORD. MAPPING = Uso de CropInfo para volver del recorte al frame total.
5. XYXY ABSOLUTO  = Coordenadas en píxeles reales, no normalizadas.
6. SEG_INTERVAL   = Optimización que diferencia la cadencia de Seg vs Pose.
```