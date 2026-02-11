# Bakery-Luna: Arquitectura y Visión Artificial de Estética Roger Rabbit


**bakery-luna** presenta un avanzado sistema de **visión artificial** diseñado para ejecutar inferencias simultáneas de **segmentación de instancias** y **estimación de pose** en tiempo real. Utilizando el entorno **OpenVINO**, el marco de trabajo **Bakery** procesa transmisiones de video para generar una estética visual inspirada en la película _Roger Rabbit_, donde los sujetos resaltan en color sobre fondos oscurecidos. La arquitectura del software está organizada en **capas modulares** que separan las entidades lógicas de las implementaciones de hardware, garantizando un código mantenible y extensible. El sistema optimiza el rendimiento mediante técnicas como el **procesamiento selectivo** de cuadros y el uso de lentes de enfoque para regiones de interés. Finalmente, el proyecto emplea un modelo de desarrollo que rastrea únicamente el **código fuente**, excluyendo archivos binarios pesados para mantener un repositorio ligero y eficiente.


---

> ¿Cómo logra el sistema la estética visual de Roger Rabbit?

El sistema logra la estética visual de **Roger Rabbit** mediante la combinación de técnicas de visión artificial y un sofisticado proceso de renderizado organizado en la **Capa de Presentación**.

Esta estética se basa principalmente en los siguientes pilares técnicos:

### 1. El DisneyAnnotator y el Sistema de 8 Capas

El corazón de este estilo es el **DisneyAnnotator**, un componente que implementa un sistema de **renderizado de 8 capas** inspirado en las técnicas de animación clásica de Disney. Su funcionamiento se divide de la siguiente manera:

- **Capa 1 (Fondo):** El fondo de la imagen se convierte a **blanco y negro** y se oscurece (por defecto al 60%), lo que genera un contraste dramático con los elementos en primer plano.
- **Capas 2-8 (Sujetos y Detalles):** En estas capas se renderizan, en **color real**, las segmentaciones de objetos, las máscaras, las cajas delimitadoras (bounding boxes), los esqueletos de pose y el efecto de foco. Esta técnica crea una profunda **separación visual** entre los elementos "vivos" (en color) y el entorno (en blanco y negro).

### 2. Inferencia de Doble Modelo

Para saber qué elementos deben permanecer en color, el sistema utiliza dos tipos de modelos de aprendizaje profundo:

- **YOLO Segmentation:** Detecta y segmenta objetos precisos mediante máscaras, permitiendo que solo el objeto (y no su recuadro) conserve el color.
- **YOLO Pose:** Detecta esqueletos humanos (17 puntos clave), permitiendo dibujar estructuras anatómicas coloridas sobre el fondo oscuro.

### 3. Efectos de Iluminación y Foco (Focus Lens)

El sistema puede aplicar un **lente de enfoque** opcional que optimiza la inferencia y añade valor estético. Este lente crea una región de **spotlight** (foco brillante), donde el brillo se intensifica para dirigir la atención del espectador hacia un área específica del video.

### 4. Personalización mediante RenderConfig

El usuario puede ajustar la intensidad del efecto visual a través de la entidad **RenderConfig**, que controla tres parámetros clave:

- **`bw_darkness`**: Controla qué tan oscuro se vuelve el fondo en blanco y negro (rango de 0.0 a 1.0).
- **`lens_brightness`**: Ajusta el multiplicador de brillo para la región del lente de enfoque.
- **`spotlight_brightness`**: Define la intensidad del brillo específicamente sobre los objetos detectados.

En resumen, el sistema utiliza la segmentación en tiempo real para "extraer" a los sujetos del fondo, aplicando transformaciones de color y brillo diferenciadas que emulan la integración de dibujos animados en entornos reales.

---

> ¿Cómo se superponen estas 8 capas de renderizado?

El sistema de renderizado de 8 capas, implementado por el **DisneyAnnotator**, superpone elementos visuales de forma jerárquica para crear una separación clara entre los sujetos detectados y su entorno.

La superposición se organiza de la siguiente manera:

- **Capa Base (Capa 1): El Fondo Oscurecido.** El proceso comienza tomando el fotograma original del video y transformándolo en un **fondo en blanco y negro**. Este fondo se oscurece automáticamente (por defecto al **60% de oscuridad**) para que sirva de contraste dramático para los elementos que se añadirán encima.
- **Capas Superiores (Capas 2 a 8): Elementos en Color.** Sobre la base oscura, el sistema superpone secuencialmente las detecciones de los modelos de visión artificial en **color real**, lo que genera el efecto de "dibujo animado" integrado en el mundo real. Estas capas incluyen:
    - **Segmentaciones y Máscaras:** Recortes precisos de los objetos detectados que conservan su color original.
    - **Cajas Delimitadoras (Bounding Boxes):** Marcos visuales que rodean a los objetos.
    - **Esqueletos de Pose:** Representaciones coloridas de las estructuras anatómicas humanas (17 puntos clave).
    - **Lente de Enfoque (Spotlight):** Una región de **brillo intenso** que actúa como un reflector, resaltando áreas específicas de interés dentro de la composición.

Este pipeline de 8 capas está diseñado para crear **profundidad y separación visual**, permitiendo que los objetos y personas "vivos" resalten sobre un entorno apagado. El usuario puede ajustar la intensidad de esta superposición (como el nivel de oscuridad del fondo o el brillo del foco) mediante la configuración de **RenderConfig**.


---

> ¿Cómo puedo personalizar los colores de las capas de renderizado?

De acuerdo con las fuentes, la personalización de la estética visual no se realiza mediante la selección de colores específicos (como elegir "rojo" o "azul"), sino a través del ajuste de la **intensidad y el contraste** de las capas mediante la entidad **`RenderConfig`**.

Para personalizar el aspecto visual de las capas, puedes ajustar los siguientes parámetros en **`RenderConfig`**:

- **`bw_darkness` (Oscuridad del fondo):** Controla el nivel de oscuridad aplicado al fondo en blanco y negro (Capa 1). Un valor de **0.0** dejaría el fondo en blanco y negro original, mientras que **1.0** lo haría completamente negro.
- **`lens_brightness` (Brillo del lente):** Es un multiplicador de brillo para la región del "Focus Lens". Permite resaltar el área de interés por encima del resto de la imagen.
- **`spotlight_brightness` (Brillo del foco):** Define la intensidad del efecto de iluminación tipo "reflector" que se aplica específicamente sobre los objetos detectados para que resalten sobre el fondo oscuro.

### Valores de Personalización Sugeridos

Las fuentes proponen tres configuraciones típicas según el estilo que desees lograr:

|Estilo deseado|`bw_darkness`|`lens_brightness`|`spotlight_brightness`|
|:--|:--|:--|:--|
|**Sutil**|0.3|1.1|1.2|
|**Estándar (Default)**|0.5|1.3|1.5|
|**Dramático**|0.7|1.5|2.0|

### Sobre los colores de los objetos y esqueletos

Es importante notar que el sistema está diseñado para que los objetos detectados y los esqueletos de pose se rendericen en **"color real"** (el color original del video) para crear contraste con el entorno oscurecido en blanco y negro. Las fuentes no mencionan un parámetro dentro de `RenderConfig` para cambiar los colores de los esqueletos o las máscaras a colores arbitrarios (como verde neón o fucsia); la estética de "Roger Rabbit" se basa precisamente en mantener el color natural del sujeto frente a un mundo desaturado.

Si deseas aplicar estos cambios desde la línea de comandos, puedes usar los argumentos `--bw-darkness` y `--brightness` al ejecutar el script principal.


---

> ¿Cómo se configura el nivel de oscuridad del fondo?

El nivel de oscuridad del fondo se configura mediante el parámetro **`bw_darkness`**, el cual forma parte de la entidad de configuración llamada **`RenderConfig`**.

Para ajustar este valor, debes tener en cuenta los siguientes detalles técnicos y prácticos:

- **Rango de valores:** El parámetro acepta un número decimal (float) en un rango de **0.0 a 1.0**. Un valor de 0.0 mantendría el fondo en blanco y negro sin oscurecerlo, mientras que 1.0 lo haría completamente negro.
- **Valor por defecto:** El sistema está preconfigurado para aplicar un **60% de oscuridad** (equivalente a un valor de **0.6**) para generar el fondo dramático característico de la estética Roger Rabbit.
- **Configuración vía CLI:** Si estás ejecutando el sistema desde la línea de comandos, puedes personalizar este nivel utilizando el argumento **`--bw-darkness`** seguido del valor deseado.

### Estilos sugeridos según el nivel de oscuridad

Dependiendo del impacto visual que busques, las fuentes sugieren los siguientes valores típicos:

- **Efecto sutil:** Un valor de **0.3**.
- **Efecto estándar:** Un valor de **0.5** (aunque las especificaciones de la capa de presentación mencionan el 0.6 como el estándar de diseño).
- **Efecto dramático:** Un valor de **0.7**.

Esta configuración es procesada por el **DisneyAnnotator**, que utiliza este nivel de oscuridad para renderizar la primera de sus 8 capas (el fondo en blanco y negro), permitiendo que los elementos en color de las capas superiores resalten significativamente.


---

> ¿Qué valores se recomiendan para un efecto visual dramático?

Para lograr un **efecto visual dramático** en la estética de Roger Rabbit, las fuentes recomiendan configurar los parámetros de **`RenderConfig`** con los siguientes valores específicos:

- **`bw_darkness` (0.7):** Incrementa la oscuridad del fondo en blanco y negro, creando un contraste más fuerte con los sujetos en color.
- **`lens_brightness` (1.5):** Aplica un multiplicador de brillo más alto a la región del lente de enfoque (Focus Lens).
- **`spotlight_brightness` (2.0):** Maximiza la intensidad del efecto de iluminación tipo "reflector" sobre los objetos detectados para que resalten vívidamente sobre el entorno oscuro.

### Comparativa de Estilos

Para contextualizar, estos valores son más intensos que la configuración estándar:

|Parámetro|Estilo Estándar|**Estilo Dramático**|
|:--|:--|:--|
|**Oscuridad fondo (`bw_darkness`)**|0.5|**0.7**|
|**Brillo del lente (`lens_brightness`)**|1.3|**1.5**|
|**Brillo del foco (`spotlight_brightness`)**|1.5|**2.0**|

### Aplicación mediante CLI

Si estás utilizando la aplicación de línea de comandos (`run_luna.py`), puedes aplicar estos niveles de intensidad utilizando los argumentos correspondientes:

- `--bw-darkness 0.7`
- `--brightness` (para ajustar los niveles de brillo del lente y el foco).

Estos ajustes permiten que el sistema resalte con mayor fuerza las segmentaciones y esqueletos de pose en "color real" frente a un fondo significativamente más lúgubre y desaturado.


---

> ¿Cómo afecta el brillo del foco a los objetos detectados?

El **brillo del foco**, controlado específicamente por el parámetro `spotlight_brightness` dentro de la entidad `RenderConfig`, afecta a los objetos detectados intensificando su **iluminación relativa** respecto al resto de la imagen.

De acuerdo con las fuentes, este parámetro funciona de la siguiente manera:

- **Resaltado de objetos:** Define la intensidad del efecto de iluminación tipo "reflector" (spotlight) que se aplica exclusivamente sobre los objetos segmentados y los esqueletos de pose.
- **Contraste visual:** Su propósito es que los sujetos detectados **resalten significativamente** sobre el fondo, el cual ya ha sido convertido a blanco y negro y oscurecido.
- **Rango de intensidad:** El valor se puede ajustar en un rango de **0.0 a 2.0**.
    - Un valor **estándar (1.5)** proporciona una iluminación equilibrada que hace que los objetos se vean claramente "vivos".
    - Un valor **dramático (2.0)** crea un brillo intenso que simula una iluminación de escenario muy marcada sobre los elementos detectados.

Es importante distinguir este parámetro de `lens_brightness`, el cual afecta a toda la **región geográfica** del lente de enfoque, mientras que el `spotlight_brightness` se concentra únicamente en las **siluetas y estructuras detectadas** por los modelos de visión artificial.

---

> ¿Cómo se activan estas capas desde la línea de comandos?

Para activar y controlar las capas de renderizado de la estética **Roger Rabbit** desde la línea de comandos, se utiliza el script principal **`run_luna.py`**. Este sistema de 8 capas está integrado en el **DisneyAnnotator** y se aplica de forma automática al ejecutar el pipeline de inferencia.

A continuación, se detallan los comandos y argumentos específicos para configurar estas capas:

### 1. Comando de Activación Básica

El efecto visual se activa por defecto al procesar un video o usar la cámara web con el siguiente comando básico:

```
python run_luna.py --models <ruta_a_modelos> --video <ruta_al_video>
```

Este comando activa el pipeline que oscurece el fondo al 60% en blanco y negro y resalta los objetos detectados en color.

### 2. Control de Intensidad de las Capas

Puedes personalizar la apariencia de las capas utilizando los siguientes argumentos de la interfaz de línea de comandos (CLI):

- **`--bw-darkness`**: Ajusta el nivel de oscuridad de la **Capa 1** (el fondo). Recibe valores entre **0.0 y 1.0**.
- **`--brightness`**: Controla la intensidad del brillo tanto para el lente de enfoque como para el efecto de "spotlight" sobre los objetos detectados (Capas 2 a 8).

### 3. Activación de la Capa de Foco (Focus Lens)

Para activar la región de iluminación especial o "spotlight" concentrado, se utilizan los parámetros de **Focus Lens**:

- **`--focus-size`**: Define el tamaño del área de enfoque (ej. `--focus-size 640`). Al activarlo, el sistema añade una capa de brillo intenso en esa región.
- **`--focus-x` y `--focus-y`**: Permiten posicionar el foco en coordenadas específicas. Si no se indican, el foco se centra automáticamente en el cuadro.

### 4. Filtrado de Capas por Clase

Si deseas que solo ciertos objetos reciban las capas de color y segmentación, puedes usar el filtro de clases:

- **`--classes`**: Seguido de los IDs de las clases (ej. `--classes 0` para personas). Esto asegura que solo las clases seleccionadas se "extraigan" del fondo oscuro.

### Ejemplo de Configuración Dramática

Para ejecutar el sistema con un fondo muy oscuro y un brillo de foco resaltado, el comando sería:

```
python run_luna.py --models exports/fp16/ --video data/sample.mp4 --bw-darkness 0.7 --brightness 2.0 --focus-size 640
```

Si necesitas profundizar en cómo estos parámetros afectan el rendimiento del sistema, puedo ofrecerte un **reporte detallado** sobre la optimización de inferencia mediante el uso del Focus Lens. ¿Te interesa?