¿Qué optimizaciones permiten procesar video en tiempo real de forma eficiente?

Para procesar video en tiempo real de forma eficiente, el sistema **Bakery** implementa diversas estrategias de optimización que reducen la carga computacional y aprovechan el hardware disponible. Estas optimizaciones se dividen principalmente en la gestión del flujo de datos, el ahorro de cálculos redundantes y el uso de aceleración por hardware.

A continuación se detallan las optimizaciones clave:

### 1. Programación Inteligente de Segmentación (Smart Scheduling)

La segmentación de instancias es una tarea computacionalmente costosa. Dado que los objetos suelen moverse lentamente entre cuadros, el sistema utiliza el parámetro **`seg_interval`** para ejecutar la segmentación solo cada **N cuadros**.

- **Impacto:** Reduce las llamadas de inferencia de segmentación en un factor de N (por ejemplo, con un intervalo de 5, se reduce en un 80%).
- **Diferenciación:** Mientras que la segmentación se espacia, la **estimación de pose** continúa ejecutándose en cada cuadro para capturar movimientos rápidos con precisión.

### 2. Caché de Preprocesamiento

Cuando los modelos de segmentación y pose comparten la misma resolución de entrada (por ejemplo, ambos a 640x640), el sistema activa un **PreprocessCache**.

- **Funcionamiento:** El primer modelo preprocesa el cuadro y almacena el tensor resultante en la caché; el segundo modelo reutiliza este mismo tensor.
- **Impacto:** Elimina operaciones redundantes de redimensionamiento (letterbox), normalización y asignación de memoria, logrando una mejora de hasta el **50% en el tiempo de preprocesamiento**.

### 3. Sistema de Lente de Enfoque (Focus Lens)

Esta optimización permite procesar solo una **región de interés (ROI)** específica en lugar del cuadro completo.

- **Funcionamiento:** Se realiza un recorte (crop) del cuadro original antes de la inferencia, lo que reduce la resolución efectiva que el modelo debe procesar.
- **Beneficio:** Mejora tanto la velocidad como la precisión al concentrar la densidad de píxeles en las áreas importantes. El sistema incluye funciones para mapear las detecciones del espacio recortado de vuelta a las coordenadas del cuadro completo.

### 4. Optimización de Inferencia con OpenVINO

El sistema utiliza el motor de inferencia de **Intel OpenVINO**, diseñado específicamente para maximizar el rendimiento en hardware de consumo.

- **Aceleración por Hardware:** Permite la ejecución optimizada en CPU, GPU integradas (iGPU) o VPU.
- **Precisión FP16:** El uso de modelos en formato **FP16** (coma flotante de 16 bits) aprovecha mejor las capacidades de las iGPU de Intel, acelerando la ejecución frente a FP32.
- **Compilación Única:** El motor compila los modelos una sola vez durante la inicialización y reutiliza el modelo compilado para todos los cuadros subsiguientes, evitando el alto costo de compilación repetida.

### 5. Eficiencia en el Postprocesamiento

- **Supresión de No Máximos (NMS):** Se aplica para eliminar detecciones redundantes o superpuestas, lo que resulta en una salida más limpia y un renderizado más rápido.
- **Uso de Letterboxing:** Mantiene la relación de aspecto original sin distorsión, lo que evita recomputar escalas complejas y mejora la precisión de los modelos sin añadir carga excesiva.

Estas optimizaciones combinadas permiten que el sistema alcance, por ejemplo, hasta **60 FPS** en configuraciones de baja resolución (320x320) utilizando hardware Intel iGPU.


