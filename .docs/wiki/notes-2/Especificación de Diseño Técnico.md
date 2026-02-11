# Especificación de Diseño Técnico: Sistema Focus Lens

## 1. Introducción y Fundamentos Estratégicos

En el despliegue de modelos de visión por computadora de alto rendimiento, la densidad de píxeles efectiva es el recurso más crítico para la precisión de la inferencia. El sistema **Focus Lens** es una solución arquitectónica diseñada para optimizar este recurso mediante la concentración del procesamiento en una Región de Interés (ROI) dinámica o estática. En entornos con alto ruido visual o donde el sujeto ocupa una fracción mínima del campo de visión, la inferencia sobre el cuadro completo diluye la capacidad del modelo.

**Impacto Estratégico:** La implementación de Focus Lens transforma la tubería de inferencia basándose en tres pilares:

- **Resolución Efectiva Elevada:** Al escalar una sub-región para completar la resolución de entrada del modelo, se extraen características morfológicas que serían imperceptibles en un reescalado global.
- **Reducción de Ruido Estructural:** La eliminación de artefactos y distracciones del fondo permite que las capas convolucionales se especialicen en señales visuales pertinentes, elevando el umbral de confianza.
- **Optimización del Pipeline:** Permite el uso de modelos con resoluciones nativas menores (ej. 320px o 416px) sobre áreas críticas, manteniendo una precisión superior a la de un modelo pesado sobre el cuadro completo.

La arquitectura de Focus Lens no es un simple componente de preprocesamiento; es una decisión de diseño que prioriza la fidelidad de los datos en el punto de captura de la inferencia.

--------------------------------------------------------------------------------

## 2. Entidades Centrales y Modelo de Datos

La integridad de una tubería de visión por computadora depende de la inmutabilidad de su configuración. El sistema Focus Lens emplea _frozen dataclasses_ para asegurar que los parámetros de transformación permanezcan constantes durante el ciclo de vida de una trama, evitando efectos secundarios en ejecuciones concurrentes.

### Arquitectura de Datos

El sistema se rige por dos entidades fundamentales que separan la intención de la ejecución: `FocusLensConfig` y `CropInfo`.

#### FocusLensConfig

Define los parámetros operacionales de la lente.

|   |   |   |   |
|---|---|---|---|
|Atributo|Tipo|Descripción|Regla de Validación Crítica|
|`focus_size`|`int`|Dimensión del recorte cuadrado.|Debe ser positivo y múltiplo de 80.|
|`focus_x`|`Optional[int]`|Coordenada X del origen.|No negativo; `None` activa centrado.|
|`focus_y`|`Optional[int]`|Coordenada Y del origen.|No negativo; `None` activa centrado.|
|`strategy`|`str`|Manejo de cuadros pequeños.|Debe ser `"zoom"` o `"pad"`.|

**Propiedad** `**is_centered**`**:** El sistema expone una propiedad booleana que devuelve `True` únicamente cuando tanto `focus_x` como `focus_y` son `None`. Esto simplifica la lógica de orquestación al determinar si la posición debe ser auto-calculada en cada frame.

#### CropInfo

Actúa como el registro de metadatos de la transformación aplicada. Almacena las coordenadas finales tras el proceso de sujeción (_clamping_), las dimensiones reales del recorte y el `scale_factor`. Este último es el factor matemático indispensable para garantizar la reversibilidad de cualquier detección.

**Justificación de Diseño:** La inmutabilidad mediante `frozen dataclasses` garantiza la seguridad de hilos (_thread-safety_). En sistemas donde múltiples modelos de post-procesamiento acceden a los metadatos de la lente, la garantía de que `CropInfo` no ha sido alterado previene errores catastróficos de mapeo de coordenadas.

--------------------------------------------------------------------------------

## 3. Protocolos de Validación y Restricciones Técnicas

El sistema Focus Lens implementa un modelo de "fallo temprano" (_fail-fast_) mediante el método `__post_init__`. Esto garantiza que los errores de configuración se intercepten antes de comprometer recursos costosos de GPU o memoria.

### Especificaciones de Validación y Racional de Stride

1. **Restricción del Múltiplo de 80:** Las arquitecturas modernas (como YOLOv8/v10) utilizan capas de _downsampling_ progresivo. El sistema requiere que el `focus_size` sea divisible por el _stride_ máximo de la red (típicamente entre 32 y 80). El cumplimiento estricto del múltiplo de 80 previene errores de desajuste de dimensiones (_dimension mismatch_) en las capas finales de la red neuronal y optimiza la alineación de memoria en el hardware de inferencia.
2. **Validación de Origen y Estrategia:** Se prohíben coordenadas negativas para evitar accesos fuera de límites del arreglo. La estrategia debe ser explícitamente `"zoom"` o `"pad"`, eliminando ambigüedades en el comportamiento del sistema ante cuadros de baja resolución.

**Garantía de Robustez:** Esta validación rigurosa asegura que la imagen entregada al motor de inferencia sea siempre compatible con las expectativas arquitectónicas del modelo, eliminando una fuente común de latencia por re-ajustes de imagen en tiempo de ejecución.

--------------------------------------------------------------------------------

## 4. Operaciones de Procesamiento y Estrategias de Adaptación

Las funciones integradas en `bakery.utils.focus_lens` constituyen el motor de transformación de imagen. La arquitectura debe gestionar casos donde el cuadro de entrada es menor que el tamaño de lente solicitado.

### Lógica de Transformación y Estrategias

|   |   |   |   |
|---|---|---|---|
|Estrategia|Comportamiento Técnico|`scale_factor`|Impacto en Calidad|
|**Zoom**|`cv2.resize` con `INTER_LINEAR`|> 1.0|Maximiza cobertura; introduce suavizado leve.|
|**Pad**|`cv2.copyMakeBorder`|= 1.0|Preserva píxeles puros; usa `BORDER_CONSTANT` negro.|

#### Algoritmo de Posicionamiento y Sujeción (Clamping)

Independientemente de la entrada del usuario, el sistema aplica una lógica de sujeción obligatoria:

- Si el origen solicitado causa que el recorte exceda los límites de la imagen, el sistema ajusta el origen al valor máximo permitido: x_{max} = frame\_width - focus\_size.
- Esto garantiza que el recorte resultante siempre sea un arreglo válido de dimensiones [focus\_size, focus\_size], protegiendo la estabilidad del puntero de memoria.

**Análisis Operacional:** La elección de `INTER_LINEAR` para el reescalado busca un equilibrio entre velocidad y preservación de bordes para detectores de características. En contraste, la estrategia **Pad** es superior cuando el análisis requiere una integridad absoluta del dato original, evitando artefactos de interpolación.

--------------------------------------------------------------------------------

## 5. Mapeo Inverso y Consistencia de Coordenadas

El desafío técnico crítico del sistema es la proyección de detecciones desde el espacio del recorte (crop-space) hacia el espacio de coordenadas globales del video original.

### Algoritmos de Transformación Matemática

El sistema utiliza fórmulas vectorizadas para revertir las transformaciones de escala y desplazamiento:

#### Detecciones (Bboxes)

Para cada coordenada (x_{crop}, y_{crop}), la posición original se recupera mediante: x_{original} = \frac{x_{crop}}{scale} + \frac{crop\_x}{scale} y_{original} = \frac{y_{crop}}{scale} + \frac{crop\_y}{scale} _Nota: El offset del recorte (_`_crop_x_`_) también debe ser des-escalado si se aplicó una estrategia de Zoom._

#### Segmentación (Máscaras)

Las máscaras se procesan en dos etapas críticas:

1. **Reescalado Inverso:** Se utiliza `cv2.INTER_NEAREST`. Este método es **mandatorio** para máscaras binarias, ya que previene la creación de valores intermedios (píxeles no booleanos) en los bordes de la segmentación.
2. **Inserción Global:** La máscara reescalada se coloca sobre una matriz de ceros del tamaño del cuadro original utilizando el offset calculado.

#### Puntos Clave (Keypoints)

El sistema implementa una lógica semántica de preservación. Los modelos de pose marcan puntos no detectados como (0,0).

- **Regla de Oro:** Si un punto es (0,0), el sistema **no** aplica la transformación de desplazamiento ni escala.
- **Racional:** Ignorar esta protección proyectaría puntos "invisibles" hacia la esquina superior izquierda del recorte, generando falsos positivos críticos en análisis de postura.

**Eficiencia de Post-procesamiento:** Todas las operaciones de mapeo están vectorizadas mediante **NumPy**, permitiendo que la transformación de múltiples detecciones se realice en tiempo constante O(1) respecto a la lógica de control, garantizando una latencia mínima en sistemas de tiempo real.

--------------------------------------------------------------------------------

## 6. Consideraciones de Rendimiento e Integración

### Análisis de Recursos y Complejidad

- **Memoria:** La estrategia de **Zoom** incrementa el consumo de RAM al generar copias escaladas. El mapeo de máscaras es la operación más costosa, con una complejidad de O(N \times H \times W), donde N es el número de detecciones y H, W las dimensiones del cuadro original.
- **Rendimiento:** Las transformaciones de Bboxes y Keypoints operan en O(N), lo que las hace prácticamente gratuitas frente al tiempo de inferencia del modelo.

### Integración en la Tubería (Dual Pipeline)

El sistema se integra de forma transparente en `run_luna.py` mediante los siguientes argumentos de CLI:

- `--focus-size`: Activa la lente y define la resolución de la ROI.
- `--focus-x` / `--focus-y`: Define la posición fija (opcional).
- `--focus-strategy`: Selecciona entre `zoom` o `pad`.

**Limitaciones y Racional Final:** Aunque Focus Lens aumenta drásticamente la precisión sobre el área enfocada, introduce una restricción de campo de visión: los objetos fuera del área de cultivo no serán detectados. Sin embargo, para aplicaciones de monitoreo especializado o seguimiento de sujetos, esta arquitectura permite transformar un modelo base genérico en una herramienta de alta resolución y alta fidelidad sin los costos asociados al re-entrenamiento o al hardware de inferencia de ultra-alta gama.