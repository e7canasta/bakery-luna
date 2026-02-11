# Guía Maestra: El Arte y la Ciencia del Renderizado Estilo 'Roger Rabbit'

## 1. La Filosofía del Efecto 'Roger Rabbit' (Spotlight Effect)

En la pedagogía de la computación visual, no basta con que un algoritmo "vea"; el usuario debe entender qué es lo que la IA está priorizando. La aplicación **Luna**, pieza central del framework _Bakery_, utiliza la clase `DisneyAnnotator` para trascender la mera detección de objetos. Inspirada en la técnica de composición de la película _¿Quién engañó a Roger Rabbit?_, esta herramienta busca una simbiosis visual entre la realidad grabada y la interpretación sintética de la IA.

**El Efecto Spotlight:** Es una filosofía de renderizado donde el entorno real se desatura y oscurece para actuar como un escenario de "Cine Negro", permitiendo que los elementos detectados —nuestros personajes animados— resalten con colores vibrantes y mayor luminosidad, guiando el ojo del espectador de forma instintiva.

Para orquestar esta narrativa visual, el sistema manipula tres parámetros maestros en su `RenderConfig`:

- `**bw_darkness**` **(0.6):** Reduce el brillo del fondo al 60% y lo desatura. Esto elimina el "ruido visual" del mundo real, convirtiéndolo en un lienzo neutro.
- `**lens_brightness**` **(1.2):** Eleva el brillo de la zona de enfoque un 20%. Actúa como un seguidor de luz teatral que indica dónde está operando nuestra atención computacional.
- `**spotlight_brightness**` **(1.2):** Incrementa el brillo de los objetos y poses detectadas en un 20%. Esto asegura que la "magia" de la IA nunca se pierda en las sombras del fondo.

Esta intención artística se materializa técnicamente a través de una construcción por capas, donde cada nivel de información se apila con precisión quirúrgica.

--------------------------------------------------------------------------------

## 2. Anatomía de la Imagen: El Sistema de 8 Capas

Pintar una imagen procesada por IA es como crear una celda de animación tradicional. El `DisneyAnnotator` utiliza un sistema de 8 capas jerárquicas. Este orden garantiza que la información más crítica (como etiquetas de texto) nunca quede oculta por elementos visuales más densos (como máscaras de color).

|   |   |   |   |
|---|---|---|---|
|Orden|Elemento Visual|Efecto/Propiedad|Propósito Pedagógico|
|**1**|Fondo (Background)|B&W al **60%** de brillo.|Establece el escenario sin distraer.|
|**2**|Región de Enfoque|Brillo al **120%** (si aplica).|Resalta el área de trabajo del "Focus Lens".|
|**3**|Máscaras (Masks)|Brillo al **100%** (Full Color).|Define la silueta del objeto detectado.|
|**4**|Cuadros (Boxes)|Contornos de Bounding Boxes.|Proporciona una referencia espacial rápida.|
|**5**|Contornos|Bordes de segmentación.|Define la frontera exacta entre IA y realidad.|
|**6**|Esqueletos|Estructuras óseas conectadas.|Visualiza la intención del movimiento humano.|
|**7**|Puntos Clave|Keypoints (nodos de articulación).|Muestra los puntos de datos precisos de la pose.|
|**8**|Etiquetas|Texto de clase y confianza (%).|Prioridad máxima: la información debe ser legible.|

Para proyectar estas capas con exactitud, debemos primero comprender cómo la IA mapea sus hallazgos desde su mundo matemático al lienzo de nuestro video.

--------------------------------------------------------------------------------

## 3. La Magia Detrás de Escena: Transformación de Coordenadas

Imagine que la IA es un proyector de cine. Ella analiza una "diapositiva" pequeña y normalizada (espacio del modelo), pero nosotros necesitamos ver el resultado en la "gran pantalla" (espacio del fotograma). Si no realizamos una transformación inversa, los esqueletos y máscaras aparecerían desplazados o deformados.

### La Fórmula de Proyección

Para devolver los hallazgos de la IA a la realidad del lienzo, aplicamos este cálculo para cada punto x, y:

```text
x_frame = (x_model / ratio) - pad_w
y_frame = (y_model / ratio) - pad_h
```

### El Viaje del Fotograma (Paso a Paso)

1. **Letterbox (Redimensionamiento):** Ajustamos la imagen original al tamaño de entrada del modelo, añadiendo bandas negras (_padding_) para no deformar la realidad.
2. **Inferencia:** La IA trabaja en este espacio cuadrado y controlado.
3. **Transformación Inversa:** Restamos el _padding_ (`pad_w`, `pad_h`) y dividimos por el factor de escala (`ratio`) para recuperar las dimensiones originales.
4. **Clipping:** Recortamos cualquier dato que exceda los límites físicos del video.

**💡 Pro Tip: Preprocessing Cache.** En ingeniería de alto rendimiento, si el modelo de Segmentación y el de Pose comparten la misma resolución (ej. 640x640), el sistema es lo suficientemente inteligente como para calcular el _letterbox_ una sola vez y reutilizarlo, eliminando cálculos redundantes.

**Nota sobre el Focus Lens:** Cuando navegamos en un "mapa dentro de otro mapa" usando zoom, aplicamos un mapeo inverso adicional. Las coordenadas se ajustan usando el `scale_factor` (zoom) y se posicionan sumando el desplazamiento original (`crop_x`, `crop_y`) para que la detección se ancle perfectamente en el video de pantalla completa.

--------------------------------------------------------------------------------

## 4. Eficiencia Inteligente: Segmentación vs. Pose

Un sistema de computación visual magistral no procesa todo por igual; prioriza según la naturaleza del movimiento. Aquí aplicamos el **Smart Scheduling** para equilibrar carga computacional y fluidez visual.

- **Estimación de Pose (Cada fotograma):** El movimiento humano es errático y rápido (_jittery_). Por ello, su eficiencia es siempre **1.0x**; la IA debe recalcular la pose en cada cuadro para que el esqueleto no se desincronice del cuerpo.
- **Segmentación (Intervalo N):** Las máscaras de los objetos son más estables. Si usamos un `--seg-interval 5`, el sistema calcula la segmentación una vez y la "congela" durante los siguientes 4 cuadros.

Esta estrategia permite una **ganancia de eficiencia de hasta 5x** en la segmentación. Mientras la pose mantiene la fluidez del actor, la máscara se mantiene firme, permitiendo que el procesador respire sin que el ojo humano note la diferencia.

--------------------------------------------------------------------------------

## 5. Resumen del Pipeline: Del Array al MP4 Cinematográfico

El flujo de ejecución de Luna es una coreografía técnica que se divide en tres actos:

1. **Iniciación:** El sistema descubre los modelos OpenVINO, valida sus resoluciones y prepara el terreno (video o webcam). Aquí es donde el framework actúa como puente, convirtiendo entidades internas en el **Supervision format** listo para el renderizado.
2. **Procesamiento (El Bucle Maestro):**
    - Captura del fotograma (NumPy array).
    - Ejecución de inferencia dual (Pose + Segmentación inteligente).
    - Mapeo de coordenadas (Transformación inversa y Focus Lens).
    - Pintado de las **8 capas** de la estética Disney.
3. **Cierre y Entrega:** Los cuadros se ensamblan usando el códec `mp4v`. El video resultante hereda los FPS del original, asegurando que la magia de Roger Rabbit se mantenga sincronizada con el tiempo real. Al final, se emite un reporte de métricas donde la eficiencia de segmentación valida nuestro éxito técnico.

La verdadera "magia" de la IA no reside solo en su capacidad de detección, sino en este equilibrio perfecto entre la **eficiencia del Smart Scheduling** y la **precisión visual del renderizado por capas**.