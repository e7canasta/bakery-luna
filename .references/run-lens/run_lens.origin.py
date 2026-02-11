"""
Bakery - FP16 Segmentation + Pose Inference - SINGLE MODEL TRANSLUCENT WITH POSE
=================================================================================
Versión HYBRID: Segmentación para todos los objetos + Pose skeleton para personas.
Estilo visual Disney/Roger Rabbit con skeleton gris translúcido (uniforme y sutil).

Diferencias con versión original:
- Procesa 1 modelo segmentation + 1 modelo pose (hybrid approach)
- Segmentación: máscaras de color para TODOS los objetos
- Pose: skeleton (edges + vertices) solo para personas detectadas
- Estilo translúcido consistente en TODAS las anotaciones (mismo gris sutil)

Filosofía: "Complejidad por diseño, no por accidente"
- Inferencia optimizada OpenVINO FP16 en iGPU Intel Xe
- **Focus Lens**: Crop cuadrado configurable antes de inferir
- **Dual Model Pipeline**: Segmentation + Pose en paralelo
- **Anotaciones multicapa**: Background + Masks + Boxes + Labels + Skeleton
- Live preview opcional para debugging
- Métricas de performance medibles

Visualización estilo Disney/Roger Rabbit (puro + esquinas + skeleton):
  🎬 Mundo B&W: Frame en gris medio (oscurecido para contraste)
  🔍 Focus Lens Region: Sutilmente más claro que el frame (~10% más brillante)
  ✨ Halo: Glow suave alrededor de detecciones (define límites sin líneas)
  📐 Esquinas bbox: Gris claro translúcido (30% opacity) - marcan los límites del bbox
  🎨 Objetos detectados: COLOR COMPLETO + BRILLANTES (como toons iluminados)
  📊 Barra de confianza: Abajo DERECHA, gris translúcido (40% opacity, MUY discreta)
  🏷️ Labels: Abajo IZQUIERDA, TRANSLÚCIDO (40% opacity) - efecto fantasmal elegante
  🦴 Skeleton (Pose): Edges gris claro + Vertices gris medio (30% opacity) - sutil y uniforme
  
Estética clásica Disney (con esquinas + skeleton sutiles):
  - Frame base: gris medio oscurecido (0.6x → más contraste)
  - Focus lens: gris ligeramente más claro (1.1x del base)
  - Halo effect: Glow BLANCO/CELESTE (opacity=0.5) que define contornos como luz pura
  - Esquinas bbox: Gris claro translúcido (30% opacity) → marcan límites sin ser invasivas
  - Objetos detectados: Color original + brightness boost (1.2x) → "spotlight effect"
  - Labels & barra: Translúcidos (40% opacity) → efecto fantasmal
  - Skeleton pose: Gris claro/medio (30% opacity) → líneas y puntos sutiles, no invasivos
  - Efecto: objetos coloridos emergen con aura de luz + anotaciones grises fantasmales

Uso típico (versión HYBRID con POSE):
    # Configuraciones ÓPTIMAS (descubrimiento: resolución baja + modelo grande = muy estable en GPU)
    # Medium @ 256px - Muy eficiente, extremadamente estable
    uv run run_lens.py --model-size m --resolution 256 --focus-size 480

    # Large @ 256px - Alta precisión, aún muy eficiente (RECOMENDADO)
    uv run run_lens.py --model-size l --resolution 256 --focus-size 480

    # Large @ 192px - Extremadamente rápido, sorprendentemente estable
    uv run run_lens.py --model-size l --resolution 192 --focus-size 320

    # Filtrar solo personas (clase 0) - máscaras + skeleton
    uv run run_lens.py --model-size s --resolution 320 --classes 0

    # Filtrar personas y autos (clases 0, 2) - skeleton solo en personas
    uv run run_lens.py --model-size m --resolution 640 --classes 0 2

    # Ajustar umbral de confianza (aplica a personas Y keypoints individuales)
    uv run run_lens.py --model-size s --resolution 320 --conf 0.5

    # Threshold más estricto → menos detecciones, pero más precisas
    uv run run_lens.py --model-size m --resolution 640 --conf 0.7

    # Con video y posición custom
    uv run run_lens.py --model-size s --resolution 320 --video videos/3.mp4 --focus-size 560 --focus-x 300 --focus-y 100

    # Con live preview para debugging
    uv run run_lens.py --model-size m --resolution 640 --focus-size 560 --show

Modelo flexible (descubrimiento: resolución baja + modelo grande = óptimo en iGPU):
    - Tamaños disponibles: s (small), m (medium), l (large), x (xlarge)
    - Resoluciones disponibles: 160px, 192px, 224px, 256px, 288px, 320px, 640px (múltiplos de 32)
    - Todas las combinaciones están soportadas
    - RECOMENDADO: l@256 (mejor balance) o m@256 (muy eficiente)
    - PRINCIPIO: Reducir resolución (cuadrático) > reducir modelo (lineal)
"""

from pathlib import Path
import openvino as ov
import cv2
import numpy as np
import time
import argparse
from typing import Tuple, List
import supervision as sv
from tqdm import tqdm


def discover_segmentation_models(
    base_dir: Path,
    model_size: str = None,
    resolution: int = None
) -> List[Path]:
    """
    Descubre modelos FP16 de segmentación en exports/fp16/sauron_segmentation/.

    Args:
        base_dir: Directorio base (exports/fp16/sauron_segmentation)
        model_size: Tamaño del modelo (s, m, l, x) o None para todos
        resolution: Resolución del modelo (320, 640, 1280) o None para todos

    Returns:
        Lista de paths a archivos .xml de modelos encontrados

    Raises:
        ValueError: Si no se encuentran modelos
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {base_dir}")

    models = []

    # Buscar en todos los subdirectorios
    for lens_dir in sorted(base_dir.iterdir()):
        if not lens_dir.is_dir():
            continue

        for subdir in sorted(lens_dir.iterdir()):
            if not subdir.is_dir() or "_fp16_gpu" not in subdir.name:
                continue

            # Parsear nombre del modelo: yolo11{size}-seg_{resolution}_fp16_gpu
            # Ejemplo: yolo11s-seg_320_fp16_gpu
            parts = subdir.name.replace("_fp16_gpu", "").split("_")
            if len(parts) < 2:
                continue

            # Extraer size y resolution del nombre
            model_part = parts[0]  # yolo11s-seg
            res_part = parts[1]    # 320

            # Extraer size (s, m, l, x) del nombre del modelo
            if "-seg" in model_part:
                size = model_part.split("-seg")[0].replace("yolo11", "")
            else:
                continue

            # Filtrar por model_size si se especificó
            if model_size and size != model_size:
                continue

            # Filtrar por resolution si se especificó
            if resolution and res_part != str(resolution):
                continue

            xml_file = subdir / f"{subdir.name}.xml"
            if xml_file.exists():
                models.append(xml_file)

    if not models:
        filter_msg = []
        if model_size:
            filter_msg.append(f"size={model_size}")
        if resolution:
            filter_msg.append(f"resolution={resolution}")
        filter_str = ", ".join(filter_msg) if filter_msg else "sin filtros"

        raise ValueError(
            f"No se encontraron modelos FP16 segmentation en {base_dir} "
            f"({filter_str})"
        )

    return models


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize imagen manteniendo aspect ratio (letterbox).
    
    Returns:
        img: Imagen resized con padding
        ratio: Ratio de resize
        (dw, dh): Padding aplicado
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def preprocess_with_metadata(
    img: np.ndarray,
    input_shape: Tuple[int, int]
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Pre-procesa imagen para inferencia YOLO segmentation.
    
    Args:
        img: Imagen BGR (OpenCV)
        input_shape: (height, width) del modelo
    
    Returns:
        Tuple con:
        - Tensor listo para inferencia [1, 3, H, W]
        - ratio: Scale ratio aplicado
        - (pad_w, pad_h): Padding aplicado en cada eje
    """
    # Letterbox resize
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=input_shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch, ratio, (dw, dh)


def apply_focus_lens(
    frame: np.ndarray,
    focus_size: int,
    focus_x: int = None,
    focus_y: int = None,
    strategy: str = "zoom"
) -> Tuple[np.ndarray, int, int, float]:
    """
    Aplica "focus lens" - crop cuadrado del frame antes de inferencia.
    
    Filosofía: Batalla naval - crop cuadrado para concentrar píxeles
    en región de interés, mejorando precisión sin cambiar resolución modelo.
    
    Args:
        frame: Frame completo BGR (OpenCV)
        focus_size: Tamaño del crop cuadrado (ej: 640, debe ser múltiplo de 80)
        focus_x: Posición X del crop (None = centrado)
        focus_y: Posición Y del crop (None = centrado)
        strategy: Estrategia si frame < focus_size:
                  - "zoom": Ampliar frame a focus_size (scale up)
                  - "pad": Rellenar con negro (letterbox inverso)
    
    Returns:
        Tuple con:
        - crop: Región cropeada/procesada [focus_size, focus_size, 3]
        - crop_x: Posición X real del crop (después de clamping)
        - crop_y: Posición Y real del crop (después de clamping)
        - scale_factor: Factor de escala aplicado (1.0 si no hubo resize)
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Fail Fast: Validar que focus_size sea múltiplo de 80
    if focus_size % 80 != 0:
        raise ValueError(
            f"focus_size debe ser múltiplo de 80 (80, 160, 240, 320, 400, 480, 560, 640...)\n"
            f"Recibido: {focus_size}"
        )
    
    # Si frame es más chico que focus_size, aplicar estrategia
    if frame_w < focus_size or frame_h < focus_size:
        print(f"⚠️  Frame ({frame_w}x{frame_h}) más chico que focus_size ({focus_size})")
        
        if strategy == "zoom":
            # Ampliar frame a focus_size (scale up)
            scale = focus_size / min(frame_w, frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"   Estrategia 'zoom': Ampliando a {new_w}x{new_h} (scale={scale:.2f}x)")
            
            # Ahora aplicar crop centrado sobre el frame ampliado
            frame = frame_resized
            frame_h, frame_w = frame.shape[:2]
            scale_factor = scale
            
        elif strategy == "pad":
            # Rellenar con negro (letterbox inverso)
            pad_w = max(0, focus_size - frame_w)
            pad_h = max(0, focus_size - frame_h)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            frame_padded = cv2.copyMakeBorder(
                frame,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # Negro
            )
            print(f"   Estrategia 'pad': Rellenando con negro a {frame_padded.shape[1]}x{frame_padded.shape[0]}")
            
            frame = frame_padded
            frame_h, frame_w = frame.shape[:2]
            scale_factor = 1.0
            
        else:
            raise ValueError(f"Estrategia inválida: {strategy}. Usar 'zoom' o 'pad'")
    
    else:
        scale_factor = 1.0
    
    # Centrar automáticamente si no se especifica posición
    if focus_x is None:
        focus_x = max(0, (frame_w - focus_size) // 2)
    if focus_y is None:
        focus_y = max(0, (frame_h - focus_size) // 2)
    
    # Clampear posición para que crop no se salga del frame
    focus_x = max(0, min(focus_x, frame_w - focus_size))
    focus_y = max(0, min(focus_y, frame_h - focus_size))
    
    # Crop cuadrado
    crop = frame[focus_y:focus_y+focus_size, focus_x:focus_x+focus_size]
    
    return crop, focus_x, focus_y, scale_factor


def map_detections_to_full_frame(
    detections: sv.Detections,
    crop_x: int,
    crop_y: int,
    full_frame_width: int,
    full_frame_height: int
) -> sv.Detections:
    """
    Mapea detecciones del crop al frame completo.
    
    Args:
        detections: Detecciones en coordenadas del crop
        crop_x: Offset X del crop en frame completo
        crop_y: Offset Y del crop en frame completo
        full_frame_width: Ancho del frame completo
        full_frame_height: Alto del frame completo
    
    Returns:
        Detecciones en coordenadas del frame completo con máscaras redimensionadas
    """
    if len(detections) == 0:
        return detections
    
    # Crear nuevo array de boxes con offset aplicado
    boxes_full = detections.xyxy.copy()
    boxes_full[:, [0, 2]] += crop_x  # x1, x2
    boxes_full[:, [1, 3]] += crop_y  # y1, y2
    
    # Redimensionar máscaras del crop al frame completo
    masks_full = None
    if detections.mask is not None and len(detections.mask) > 0:
        # Máscaras están en tamaño del crop, necesitamos expandirlas al frame completo
        crop_height, crop_width = detections.mask[0].shape
        masks_full = []
        
        for mask in detections.mask:
            # Crear máscara vacía del tamaño del frame completo
            mask_full = np.zeros((full_frame_height, full_frame_width), dtype=bool)
            
            # Posicionar la máscara del crop en el frame completo
            y_end = min(crop_y + crop_height, full_frame_height)
            x_end = min(crop_x + crop_width, full_frame_width)
            
            # Calcular dimensiones actuales del crop en el frame
            actual_h = y_end - crop_y
            actual_w = x_end - crop_x
            
            # Copiar la máscara del crop (recortada si es necesario)
            mask_full[crop_y:y_end, crop_x:x_end] = mask[:actual_h, :actual_w]
            masks_full.append(mask_full)
        
        masks_full = np.array(masks_full)
    
    # Crear nuevo sv.Detections con boxes offseteadas y máscaras expandidas
    detections_full = sv.Detections(
        xyxy=boxes_full,
        confidence=detections.confidence,
        class_id=detections.class_id,
        mask=masks_full
    )
    
    return detections_full


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convierte boxes de [x_center, y_center, w, h] a [x1, y1, x2, y2]"""
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return boxes_xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    """
    Non-Maximum Suppression (NMS).
    
    Args:
        boxes: Array [N, 4] en formato [x1, y1, x2, y2]
        scores: Array [N] con confidence scores
        iou_threshold: Threshold de IoU
    
    Returns:
        Indices de boxes a mantener
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def process_mask(
    protos: np.ndarray,
    mask_coef: np.ndarray,
    box: np.ndarray,
    input_shape: Tuple[int, int],
    upsample: bool = True
) -> np.ndarray:
    """
    Procesa mask coefficients para generar máscara de segmentación.
    
    Args:
        protos: Prototipos de máscaras [mask_dim, mask_h, mask_w] (ej: [32, 80, 80])
        mask_coef: Coeficientes del mask head [mask_dim] (ej: [32])
        box: Bounding box [x1, y1, x2, y2] en coordenadas del input
        input_shape: (height, width) del input del modelo
        upsample: Si True, upsamples mask a input_shape
    
    Returns:
        Máscara binaria [H, W] en rango [0, 1]
    """
    # Matmul: [mask_dim] @ [mask_dim, mask_h, mask_w] -> [mask_h, mask_w]
    mask = np.einsum('i,ijk->jk', mask_coef, protos)
    
    # Sigmoid para [0, 1]
    mask = 1 / (1 + np.exp(-mask))
    
    # Upsample a input shape si es necesario
    if upsample:
        mask = cv2.resize(mask, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Crop mask a bounding box (IMPORTANTE: bbox ya está en coords del input)
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, mask.shape[1] - 1))
    y1 = max(0, min(y1, mask.shape[0] - 1))
    x2 = max(0, min(x2, mask.shape[1]))
    y2 = max(0, min(y2, mask.shape[0]))
    
    # Máscara solo dentro del bbox
    mask_bbox = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        mask_bbox[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    
    return mask_bbox


def postprocess_segmentation(
    output_boxes: np.ndarray,
    output_masks: np.ndarray,
    input_shape: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-procesa salida de YOLO segmentation.
    
    Args:
        output_boxes: Salida detección [1, num_classes + 4 + mask_dim, num_anchors]
                      Ej: [1, 116, 2100] = [1, 80 + 4 + 32, 2100]
        output_masks: Prototipos de máscaras [1, mask_dim, mask_h, mask_w]
                      Ej: [1, 32, 80, 80]
        input_shape: (height, width) del input del modelo
        conf_threshold: Threshold de confidence
        iou_threshold: Threshold de IoU para NMS
    
    Returns:
        (boxes, scores, class_ids, masks)
        - boxes: [N, 4] en formato [x1, y1, x2, y2]
        - scores: [N]
        - class_ids: [N]
        - masks: [N, H, W] máscaras binarias
    """
    # YOLOv11 segmentation output: [1, 116, 2100] -> [2100, 116]
    output_boxes = output_boxes.squeeze(0)  # [1, 116, 2100] -> [116, 2100]
    
    # Si la dimensión de features está primero, hacer transpose
    if output_boxes.shape[0] < output_boxes.shape[1]:
        output_boxes = output_boxes.T  # [116, 2100] -> [2100, 116]
    
    # Extraer componentes
    boxes = output_boxes[:, :4]  # [x, y, w, h]
    
    # Para YOLOv11-seg: 116 = 4 (box) + 80 (clases) + 32 (mask coefs)
    num_classes = 80
    mask_dim = 32
    
    class_scores = output_boxes[:, 4:4+num_classes]
    mask_coefs = output_boxes[:, 4+num_classes:]
    
    # Verificar dimensiones
    if mask_coefs.shape[1] != mask_dim:
        if verbose:
            print(f"⚠️  Warning: Esperábamos {mask_dim} coefs, encontramos {mask_coefs.shape[1]}")
        mask_dim = mask_coefs.shape[1]
    
    # Scores y class IDs
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    
    # Filtrar por confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    mask_coefs = mask_coefs[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Convert xywh to xyxy
    boxes = xywh2xyxy(boxes)
    
    # NMS
    indices = nms(boxes, scores, iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    mask_coefs = mask_coefs[indices]
    
    # Generar máscaras
    output_masks = output_masks.squeeze(0)  # [1, 32, 80, 80] -> [32, 80, 80]
    
    # Verificar dimensiones
    if len(mask_coefs) > 0:
        if mask_coefs.shape[1] != output_masks.shape[0]:
            if verbose:
                print(f"⚠️  Shape mismatch: mask_coefs={mask_coefs.shape}, output_masks={output_masks.shape}")
            expected_dim = output_masks.shape[0]
            if mask_coefs.shape[1] > expected_dim:
                if verbose:
                    print(f"   Truncando mask_coefs de {mask_coefs.shape[1]} a {expected_dim}")
                mask_coefs = mask_coefs[:, :expected_dim]
            else:
                return boxes, scores, class_ids, np.array([])
    
    masks = []
    for box, coef in zip(boxes, mask_coefs):
        mask = process_mask(output_masks, coef, box, input_shape, upsample=True)
        masks.append(mask)
    
    masks = np.array(masks) if masks else np.array([])
    
    return boxes, scores, class_ids, masks


def discover_pose_models(
    base_dir: Path,
    model_size: str = None,
    resolution: int = None
) -> List[Path]:
    """
    Descubre modelos FP16 de pose en exports/fp16/sauron_pose/.

    Args:
        base_dir: Directorio base (exports/fp16/sauron_pose)
        model_size: Tamaño del modelo (s, m, l, x) o None para todos
        resolution: Resolución del modelo (320, 640, 1280) o None para todos

    Returns:
        Lista de paths a archivos .xml de modelos encontrados

    Raises:
        ValueError: Si no se encuentran modelos
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {base_dir}")

    models = []

    # Buscar en todos los subdirectorios
    for lens_dir in sorted(base_dir.iterdir()):
        if not lens_dir.is_dir():
            continue

        for subdir in sorted(lens_dir.iterdir()):
            if not subdir.is_dir() or "_fp16_gpu" not in subdir.name:
                continue

            # Parsear nombre del modelo: yolo11{size}-pose_{resolution}_fp16_gpu
            # Ejemplo: yolo11s-pose_320_fp16_gpu
            parts = subdir.name.replace("_fp16_gpu", "").split("_")
            if len(parts) < 2:
                continue

            # Extraer size y resolution del nombre
            model_part = parts[0]  # yolo11s-pose
            res_part = parts[1]    # 320

            # Extraer size (s, m, l, x) del nombre del modelo
            if "-pose" in model_part:
                size = model_part.split("-pose")[0].replace("yolo11", "")
            else:
                continue

            # Filtrar por model_size si se especificó
            if model_size and size != model_size:
                continue

            # Filtrar por resolution si se especificó
            if resolution and res_part != str(resolution):
                continue

            xml_file = subdir / f"{subdir.name}.xml"
            if xml_file.exists():
                models.append(xml_file)

    if not models:
        filter_msg = []
        if model_size:
            filter_msg.append(f"size={model_size}")
        if resolution:
            filter_msg.append(f"resolution={resolution}")
        filter_str = ", ".join(filter_msg) if filter_msg else "sin filtros"

        raise ValueError(
            f"No se encontraron modelos FP16 pose en {base_dir} "
            f"({filter_str})"
        )

    return models


def postprocess_pose(
    output_data: np.ndarray,
    input_shape: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-procesa salida de YOLO pose.

    Args:
        output_data: Salida del modelo [1, num_classes + 4 + num_keypoints*3, num_anchors]
                     Ej: [1, 56, 2100] = [1, 1 + 4 + 17*3, 2100] para COCO pose
        input_shape: (height, width) del input del modelo
        conf_threshold: Threshold de confidence
        iou_threshold: Threshold de IoU para NMS

    Returns:
        (boxes, scores, class_ids, keypoints)
        - boxes: [N, 4] en formato [x1, y1, x2, y2]
        - scores: [N]
        - class_ids: [N]
        - keypoints: [N, num_keypoints, 3] (x, y, confidence)
    """
    # YOLO pose output: [1, 56, 2100] -> [2100, 56]
    output_data = output_data.squeeze(0)

    if output_data.shape[0] < output_data.shape[1]:
        output_data = output_data.T  # [56, 2100] -> [2100, 56]

    # Extraer componentes
    # Para COCO pose: 56 = 4 (box) + 1 (class score) + 51 (17 keypoints * 3)
    boxes = output_data[:, :4]  # [x, y, w, h]
    scores = output_data[:, 4]  # confidence para clase persona

    # Keypoints: 17 puntos x 3 valores (x, y, conf)
    num_keypoints = 17
    keypoints_data = output_data[:, 5:5 + num_keypoints * 3]
    keypoints = keypoints_data.reshape(-1, num_keypoints, 3)

    # Filtrar por confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    keypoints = keypoints[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Convert xywh to xyxy
    boxes = xywh2xyxy(boxes)

    # NMS
    indices = nms(boxes, scores, iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    keypoints = keypoints[indices]

    # Class ID siempre es 0 (persona) para pose
    class_ids = np.zeros(len(boxes), dtype=int)

    return boxes, scores, class_ids, keypoints


def convert_to_supervision_keypoints(
    keypoints: np.ndarray,
    ratio: float,
    pad_w: float,
    pad_h: float,
    orig_width: int,
    orig_height: int,
    keypoint_conf_threshold: float = 0.5
) -> sv.KeyPoints:
    """
    Convierte keypoints de OpenVINO a formato supervision.KeyPoints.

    Args:
        keypoints: [N, num_keypoints, 3] keypoints en coords del input model (x, y, conf)
        ratio: Scale ratio del letterbox
        pad_w, pad_h: Padding del letterbox
        orig_width, orig_height: Dimensiones del frame original
        keypoint_conf_threshold: Umbral de confianza para keypoints individuales (default: 0.5)

    Returns:
        sv.KeyPoints con coordenadas en el frame original

    Note:
        Keypoints con confidence < threshold son marcados como [0, 0] (no visibles)
    """
    if len(keypoints) == 0:
        return sv.KeyPoints.empty()

    # Extraer xy y confidence
    xy = keypoints[:, :, :2].copy()  # [N, 17, 2]
    confidence = keypoints[:, :, 2].copy()   # [N, 17]

    # Crear máscara de keypoints con baja confianza (filtrar DESPUÉS de transformaciones)
    low_conf_mask = confidence < keypoint_conf_threshold

    # Primero: Deshacer letterbox (antes de filtrar, para transformar coordenadas correctamente)
    xy[:, :, 0] = (xy[:, :, 0] - pad_w) / ratio  # x
    xy[:, :, 1] = (xy[:, :, 1] - pad_h) / ratio  # y

    # Clampear a dimensiones originales
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, orig_width)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, orig_height)

    # Ahora sí: Marcar keypoints con baja confianza como [0, 0] (DESPUÉS de transformaciones)
    # Convención estándar: [0, 0] = keypoint no visible/confiable
    xy[low_conf_mask] = 0  # Marcar como no visible
    confidence[low_conf_mask] = 0  # Confidence 0 para no visibles

    # Crear sv.KeyPoints
    return sv.KeyPoints(
        xy=xy.astype(np.float32),
        confidence=confidence.astype(np.float32),
        class_id=np.zeros(len(keypoints), dtype=int)  # Siempre persona (clase 0)
    )


def map_keypoints_to_full_frame(
    keypoints: sv.KeyPoints,
    crop_x: int,
    crop_y: int
) -> sv.KeyPoints:
    """
    Mapea keypoints del crop al frame completo.

    Args:
        keypoints: KeyPoints en coordenadas del crop
        crop_x: Offset X del crop en frame completo
        crop_y: Offset Y del crop en frame completo

    Returns:
        KeyPoints en coordenadas del frame completo

    Note:
        Keypoints marcados como [0, 0] (no visibles) NO reciben offset,
        permanecen en [0, 0] para que sean ignorados por annotators.
    """
    if len(keypoints) == 0:
        return keypoints

    # Copiar xy y aplicar offset
    xy_full = keypoints.xy.copy()

    # Crear máscara de keypoints válidos (NO son [0, 0])
    # Usar allclose para manejar pequeños errores de floating point
    valid_mask = ~np.all(np.isclose(xy_full, 0), axis=2)  # Shape: [N, num_keypoints]

    # Aplicar offset SOLO a keypoints válidos
    # Expandir mask a 3D para broadcasting con xy_full
    valid_mask_expanded = np.stack([valid_mask, valid_mask], axis=2)  # [N, num_keypoints, 2]

    # Aplicar offset solo donde valid_mask es True
    xy_full[:, :, 0] = np.where(valid_mask, xy_full[:, :, 0] + crop_x, 0)  # x
    xy_full[:, :, 1] = np.where(valid_mask, xy_full[:, :, 1] + crop_y, 0)  # y

    # Crear nuevo KeyPoints con coordenadas ajustadas
    return sv.KeyPoints(
        xy=xy_full,
        confidence=keypoints.confidence,
        class_id=keypoints.class_id
    )


def convert_to_supervision_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    ratio: float,
    pad_w: float,
    pad_h: float,
    orig_width: int,
    orig_height: int
) -> sv.Detections:
    """
    Convierte outputs de OpenVINO a formato supervision.Detections.
    
    Args:
        boxes: [N, 4] en formato [x1, y1, x2, y2] (coords del input model)
        scores: [N] confidence scores
        class_ids: [N] class IDs
        masks: [N, H, W] máscaras en coords del input model
        ratio: Scale ratio del letterbox
        pad_w, pad_h: Padding del letterbox
        orig_width, orig_height: Dimensiones del frame original
    
    Returns:
        sv.Detections con boxes y masks en coordenadas del frame original
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    # 1. Deshacer letterbox de boxes
    boxes_orig = boxes.copy()
    boxes_orig[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio  # x1, x2
    boxes_orig[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio  # y1, y2
    
    # Clampear a dimensiones originales
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_width)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_height)
    
    # 2. Deshacer letterbox de masks
    masks_orig = []
    for mask in masks:
        # Remover padding de la máscara (mask está en input_shape coords)
        mask_h, mask_w = mask.shape
        pad_top = int(pad_h)
        pad_bottom = int(pad_h)
        pad_left = int(pad_w)
        pad_right = int(pad_w)
        
        # Crop máscara para remover padding
        mask_unpadded = mask[pad_top:mask_h-pad_bottom, pad_left:mask_w-pad_right]
        
        # Resize a dimensiones del frame original
        if mask_unpadded.shape[0] > 0 and mask_unpadded.shape[1] > 0:
            mask_resized = cv2.resize(
                mask_unpadded,
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Fallback: resize directo
            mask_resized = cv2.resize(
                mask,
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Threshold para máscara binaria
        mask_bin = (mask_resized > 0.5).astype(bool)
        masks_orig.append(mask_bin)
    
    masks_orig = np.array(masks_orig)
    
    # 3. Crear sv.Detections
    detections = sv.Detections(
        xyxy=boxes_orig,
        confidence=scores,
        class_id=class_ids.astype(int),
        mask=masks_orig
    )
    
    return detections


def run_inference_fp16_segmentation_sv_lens(
    model_path: Path,
    video_path: str,
    output_dir: Path,
    device: str = "GPU",
    show_live: bool = False,
    focus_size: int = None,
    focus_x: int = None,
    focus_y: int = None,
    focus_strategy: str = "zoom",
    class_filter: List[int] = None,
    conf_threshold: float = 0.25,
    pose_model_path: Path = None
) -> dict:
    """
    Ejecuta inferencia con modelo FP16 segmentation usando supervision + focus lens + pose.

    Args:
        model_path: Ruta al .xml del modelo OpenVINO IR (FP16 segmentation)
        video_path: Ruta al video
        output_dir: Directorio para resultados
        device: Dispositivo OpenVINO (GPU para iGPU Intel)
        show_live: Si True, muestra preview en vivo (cv2.imshow)
        focus_size: Tamaño del crop cuadrado (None = sin crop, múltiplo de 80)
        focus_x: Posición X del crop (None = centrado)
        focus_y: Posición Y del crop (None = centrado)
        focus_strategy: Estrategia si frame < focus_size ("zoom" o "pad")
        class_filter: Lista de IDs de clases a mostrar (None = todas)
        conf_threshold: Umbral de confianza (0.0 - 1.0). Se aplica a:
                        - Detecciones de segmentación
                        - Detecciones de personas (pose)
                        - Keypoints individuales del skeleton
        pose_model_path: Ruta al .xml del modelo pose (None = sin pose estimation)

    Returns:
        Dict con métricas de performance

    Note:
        El threshold se aplica en tres niveles:
        1. Segmentación: filtra detecciones de objetos con baja confianza
        2. Pose: filtra personas detectadas con baja confianza
        3. Keypoints: marca como invisibles (0,0) los keypoints con confianza < threshold
    """
    # Fail Fast: Validar archivos
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = model_path.stem
    focus_info = f" + Focus Lens {focus_size}x{focus_size}" if focus_size else ""
    print(f"\n🎬 Procesando con: {model_name} (FP16 Seg + Supervision{focus_info})")
    print("-" * 60)
    
    if focus_size:
        focus_pos = f"({focus_x}, {focus_y})" if focus_x is not None and focus_y is not None else "(centrado)"
        print(f"🎯 Focus Lens: {focus_size}x{focus_size} en {focus_pos} | Estrategia: {focus_strategy}")
    
    if class_filter is not None:
        print(f"🎨 Filtro de clases: {class_filter}")
    print(f"📊 Umbral de confianza: {conf_threshold}")
    
    # Cargar modelo OpenVINO
    print(f"🔧 Cargando modelo FP16 segmentation en {device}...")
    core = ov.Core()
    model = core.read_model(model_path)
    
    # Verificar que GPU está disponible
    available_devices = core.available_devices
    if device == "GPU" and "GPU" not in available_devices:
        print(f"⚠️  GPU no disponible. Dispositivos: {available_devices}")
        print("   Fallback a CPU...")
        device = "CPU"
    
    compiled_model = core.compile_model(model, device)
    print(f"✅ Modelo compilado en: {device}")

    # Input/Output
    input_layer = compiled_model.input(0)
    output_boxes_layer = compiled_model.output(0)
    output_masks_layer = compiled_model.output(1)

    # Input shape: [1, 3, H, W]
    input_shape = input_layer.shape
    _, _, input_h, input_w = input_shape

    print(f"📐 Input shape: {input_shape}")
    print(f"📐 Output boxes shape: {output_boxes_layer.shape}")
    print(f"📐 Output masks shape: {output_masks_layer.shape}")

    # Cargar modelo pose si se especificó
    compiled_pose_model = None
    pose_input_layer = None
    pose_output_layer = None
    if pose_model_path is not None:
        print(f"\n🔧 Cargando modelo FP16 pose en {device}...")
        pose_model = core.read_model(pose_model_path)
        compiled_pose_model = core.compile_model(pose_model, device)
        pose_input_layer = compiled_pose_model.input(0)
        pose_output_layer = compiled_pose_model.output(0)
        print(f"✅ Modelo pose compilado en: {device}")
        print(f"📐 Pose input shape: {pose_input_layer.shape}")
        print(f"📐 Pose output shape: {pose_output_layer.shape}")
    
    # Configurar supervision annotators para estilo Disney/Roger Rabbit
    # Halo para definir mejor los límites sin líneas duras
    halo_annotator = sv.HaloAnnotator(
        color=sv.Color.from_rgb_tuple((240, 248, 255)),  # Blanco azulado (alice blue) - luz pura
        opacity=0.5,  # Glow visible pero suave
        kernel_size=40  # Tamaño del halo (más grande = más difuso)
    )
    
    # Esquinas de bounding box (sutiles y translúcidas)
    box_corner_annotator = sv.BoxCornerAnnotator(
        color=sv.Color.from_rgb_tuple((200, 200, 200)),  # Gris claro suave
        thickness=1,  # Líneas muy finas (más sutiles)
        corner_length=15  # Esquinas de tamaño medio (no muy grandes)
    )
    
    # Barra de porcentaje de confianza (abajo derecha, MUY sutil y translúcida)
    # Nota: solo dibuja barra visual, no texto - debe pasar desapercibida
    percentage_bar_annotator = sv.PercentageBarAnnotator(
        height=16,  # Altura de la barra (delgada)
        width=55,  # Ancho de la barra (compacta)
        color=sv.Color.from_rgb_tuple((90, 90, 90)),  # Gris medio (sutil)
        border_color=sv.Color.from_rgb_tuple((70, 70, 70)),  # Borde gris oscuro
        border_thickness=1,  # Borde mínimo
        position=sv.Position.BOTTOM_RIGHT  # Abajo derecha (label está abajo izquierda)
    )
    
    # Labels elegantes: blanco sobre fondo oscuro uniforme (abajo)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.BOTTOM_LEFT,  # Abajo para no tapar tanto
        text_scale=0.4,  # Más pequeño, más sutil
        text_thickness=1,
        text_padding=8,  # Más padding para mejor legibilidad
        text_color=sv.Color.from_rgb_tuple((255, 255, 255)),  # Texto blanco puro
        color=sv.Color.from_rgb_tuple((40, 40, 40)),  # Fondo gris muy oscuro (uniforme para todas las clases)
        border_radius=3  # Bordes redondeados (más elegante)
    )

    # Pose annotators (solo si hay modelo pose cargado)
    edge_annotator = None
    vertex_annotator = None
    if compiled_pose_model is not None:
        # Edges: líneas del skeleton en gris claro (mismo estilo que esquinas bbox)
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_rgb_tuple((200, 200, 200)),  # Gris claro suave (como esquinas)
            thickness=1  # Líneas muy finas (sutiles como las esquinas)
        )

        # Vertices: círculos en keypoints (gris medio, más visible que edges)
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_rgb_tuple((160, 160, 160)),  # Gris medio (más oscuro que edges)
            radius=3  # Círculos pequeños y sutiles
        )
    
    # Obtener info del video
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    
    # Output video
    output_video = output_dir / f"output_{model_name}_sv.mp4"
    
    # Métricas
    start_time = time.time()
    frame_count = 0
    total_detections = 0
    
    # Procesar video con supervision
    print(f"\n🎥 Procesando {video_info.total_frames} frames...")
    if show_live:
        print("👁️  Live preview habilitado (presiona 'q' para salir)")
    
    with sv.VideoSink(target_path=str(output_video), video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Frames"):
            # Aplicar focus lens si está configurado
            if focus_size:
                crop, crop_x, crop_y, scale_factor = apply_focus_lens(
                    frame, focus_size, focus_x, focus_y, focus_strategy
                )
                frame_to_infer = crop
                crop_width = crop.shape[1]
                crop_height = crop.shape[0]
            else:
                frame_to_infer = frame
                crop_x = 0
                crop_y = 0
                crop_width = video_info.width
                crop_height = video_info.height
                scale_factor = 1.0
            
            # Pre-process
            input_tensor, ratio, (pad_w, pad_h) = preprocess_with_metadata(
                frame_to_infer, (input_h, input_w)
            )
            
            # Inference OpenVINO
            result = compiled_model([input_tensor])
            output_boxes = result[output_boxes_layer]
            output_masks = result[output_masks_layer]
            
            # Post-process (verbose solo en primer frame) con umbral personalizado
            boxes, scores, class_ids, masks = postprocess_segmentation(
                output_boxes, output_masks, (input_h, input_w),
                conf_threshold=conf_threshold,
                verbose=(frame_count == 0)
            )
            
            total_detections += len(boxes)
            
            # Convertir a supervision.Detections (en coords del crop)
            detections = convert_to_supervision_detections(
                boxes, scores, class_ids, masks,
                ratio, pad_w, pad_h,
                crop_width, crop_height
            )
            
            # Filtrar por clases si se especificó
            if class_filter is not None and len(detections) > 0:
                mask = np.isin(detections.class_id, class_filter)
                detections = detections[mask]

            # Si usamos focus lens, mapear detecciones al frame completo
            if focus_size:
                detections = map_detections_to_full_frame(
                    detections, crop_x, crop_y,
                    video_info.width, video_info.height
                )

            # Inferencia de pose (solo si hay modelo pose cargado)
            keypoints = sv.KeyPoints.empty()
            if compiled_pose_model is not None:
                # Reusar el mismo crop y pre-processing
                pose_result = compiled_pose_model([input_tensor])
                pose_output = pose_result[pose_output_layer]

                # Post-process pose (verbose solo en primer frame)
                pose_boxes, pose_scores, pose_class_ids, pose_keypoints = postprocess_pose(
                    pose_output, (input_h, input_w),
                    conf_threshold=conf_threshold,
                    iou_threshold=0.45
                )

                # Convertir a supervision.KeyPoints (en coords del crop)
                if len(pose_keypoints) > 0:
                    keypoints = convert_to_supervision_keypoints(
                        pose_keypoints, ratio, pad_w, pad_h,
                        crop_width, crop_height,
                        keypoint_conf_threshold=conf_threshold  # Usar mismo threshold que detecciones
                    )

                    # Si usamos focus lens, mapear keypoints al frame completo
                    if focus_size:
                        keypoints = map_keypoints_to_full_frame(
                            keypoints, crop_x, crop_y
                        )
            
            # 🎨 Efecto Disney/Roger Rabbit PURO: B&W world + Colorful bright objects
            # ========================================================================
            
            # 1. Convertir frame completo a B&W y OSCURECER para más contraste
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)  # Back to 3 channels
            # Oscurecer a gris medio (0.6x) para que objetos resalten más
            frame_bw = np.clip(frame_bw.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
            
            # 2. Aclarar MUY SUTILMENTE la región del focus lens (si está activo)
            annotated_frame = frame_bw.copy()
            if focus_size:
                # Región del focus lens: solo 10% más claro que el base
                lens_region = annotated_frame[crop_y:crop_y+focus_size, crop_x:crop_x+focus_size].copy()
                lens_region = np.clip(lens_region.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
                annotated_frame[crop_y:crop_y+focus_size, crop_x:crop_x+focus_size] = lens_region
            
            # 3. Aplicar HALO para definir mejor los límites (antes de pintar color)
            if len(detections) > 0:
                annotated_frame = halo_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )
            
            # 3.5. Aplicar esquinas de bounding box (translúcidas, sutiles)
            if len(detections) > 0:
                # Crear overlay para efecto translúcido
                overlay_corners = annotated_frame.copy()
                overlay_corners = box_corner_annotator.annotate(
                    scene=overlay_corners,
                    detections=detections
                )
                # Blend: 70% frame + 30% esquinas = muy sutiles
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay_corners, 0.3, 0)
            
            # 4. "Pintar" objetos detectados con color original + BRIGHTNESS BOOST
            if len(detections) > 0 and detections.mask is not None:
                for mask in detections.mask:
                    # Expandir máscara a 3 canales
                    mask_3ch = np.stack([mask] * 3, axis=-1)
                    
                    # Color del objeto con brightness boost (1.2x = 20% más brillante)
                    frame_bright = np.clip(frame.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
                    
                    # Donde mask=True, usar color brillante; donde mask=False, mantener B&W
                    annotated_frame = np.where(mask_3ch, frame_bright, annotated_frame)
            
            # 5. Barra de porcentaje de confianza (abajo derecha, translúcida)
            if len(detections) > 0:
                # Crear overlay para efecto translúcido
                overlay = annotated_frame.copy()
                overlay = percentage_bar_annotator.annotate(
                    scene=overlay,
                    detections=detections
                )
                # Blend: 60% frame original + 40% barra = translúcida
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, overlay, 0.4, 0)
            
            # 6. Labels con ID y confianza (abajo izquierda, TRANSLÚCIDO)
            if len(detections) > 0:
                labels = [
                    f"ID:{class_id} {conf:.2f}"
                    for class_id, conf in zip(detections.class_id, detections.confidence)
                ]
                # Crear overlay para efecto translúcido (igual que la barra)
                overlay_labels = annotated_frame.copy()
                overlay_labels = label_annotator.annotate(
                    scene=overlay_labels,
                    detections=detections,
                    labels=labels
                )
                # Blend: 60% frame original + 40% labels = translúcido fantasmal
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, overlay_labels, 0.4, 0)
            
            # 7. Marcador SUTIL del focus lens (si está activo) - opcional
            if focus_size:
                # Línea muy delgada y semi-transparente
                overlay = annotated_frame.copy()
                cv2.rectangle(
                    overlay,
                    (crop_x, crop_y),
                    (crop_x + focus_size, crop_y + focus_size),
                    color=(150, 150, 150),  # Gris medio
                    thickness=1
                )
                # Blend con alpha=0.3 (muy sutil)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

            # 8. Skeleton de pose (edges + vertices) - TRANSLÚCIDO gris sutil
            if compiled_pose_model is not None and len(keypoints) > 0:
                # Edges primero (líneas del skeleton)
                overlay_skeleton = annotated_frame.copy()
                overlay_skeleton = edge_annotator.annotate(
                    scene=overlay_skeleton,
                    key_points=keypoints
                )
                # Blend: 70% frame + 30% edges = muy sutil (como esquinas bbox)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay_skeleton, 0.3, 0)

                # Vertices encima (círculos en keypoints)
                overlay_vertices = annotated_frame.copy()
                overlay_vertices = vertex_annotator.annotate(
                    scene=overlay_vertices,
                    key_points=keypoints
                )
                # Blend: 70% frame + 30% vertices = igual de sutil que edges
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay_vertices, 0.3, 0)

            # Guardar frame
            sink.write_frame(frame=annotated_frame)
            
            # Live preview (opcional)
            if show_live:
                # Resize para display si es muy grande
                display_frame = annotated_frame
                if video_info.width > 1280:
                    scale = 1280 / video_info.width
                    new_w = int(video_info.width * scale)
                    new_h = int(video_info.height * scale)
                    display_frame = cv2.resize(annotated_frame, (new_w, new_h))
                
                cv2.imshow(f"Inference - {model_name}", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Live preview cancelado por usuario")
                    break
            
            frame_count += 1
    
    if show_live:
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    
    # Métricas
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    
    metrics = {
        "model": f"{model_name} (FP16 Seg + Supervision)",
        "device": device,
        "total_frames": frame_count,
        "elapsed_time": elapsed_time,
        "fps": fps,
        "avg_detections": avg_detections,
        "output_path": output_video
    }
    
    print(f"⏱️  Tiempo total: {elapsed_time:.2f}s")
    print(f"🎞️  FPS: {fps:.2f}")
    print(f"📦 Detecciones promedio: {avg_detections:.1f} por frame")
    print(f"💾 Video guardado: {output_video}")
    
    return metrics


def main():
    """Pipeline de inferencia FP16 segmentation con supervision - SINGLE MODEL"""
    
    # Parse argumentos
    parser = argparse.ArgumentParser(
        description="Inferencia FP16 Segmentation con Supervision (iGPU Intel Xe) - SINGLE MODEL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos (versión HYBRID con POSE):

  # Configuraciones recomendadas (balance óptimo GPU):
  # Medium @ 256px - Muy eficiente y estable
  uv run run_lens.py --model-size m --resolution 256 --focus-size 480

  # Large @ 256px - Alta precisión, aún muy eficiente
  uv run run_lens.py --model-size l --resolution 256 --focus-size 480

  # Large @ 192px - Extremadamente rápido
  uv run run_lens.py --model-size l --resolution 192 --focus-size 320

  # Medium @ 320px - Resolución estándar (muy estable)
  uv run run_lens.py --model-size m --resolution 320 --focus-size 560

  # Comparación resoluciones (mismo modelo):
  uv run run_lens.py --model-size l --resolution 192  # Rápido
  uv run run_lens.py --model-size l --resolution 256  # Balanceado
  uv run run_lens.py --model-size l --resolution 320  # Preciso

  # Filtrar solo personas (clase 0) - máscaras + skeleton
  uv run run_lens.py --model-size s --resolution 320 --classes 0

  # Filtrar personas y autos (clases 0, 2) - skeleton solo en personas
  uv run run_lens.py --model-size m --resolution 640 --classes 0 2

  # Ajustar umbral de confianza (aplica a personas Y keypoints)
  uv run run_lens.py --model-size s --resolution 320 --conf 0.5

  # Threshold alto → solo keypoints muy confiables (skeleton más limpio)
  uv run run_lens.py --model-size m --resolution 640 --conf 0.7

  # Combinar filtros + focus lens
  uv run run_lens.py --model-size s --resolution 320 --focus-size 560 --classes 0 --conf 0.4

  # Con video y posición custom
  uv run run_lens.py --model-size m --resolution 640 --video videos/3.mp4 --focus-size 560 --focus-x 300 --focus-y 100

  # Con live preview para debugging
  uv run run_lens.py --model-size s --resolution 320 --focus-size 560 --show

CLASES COCO MÁS COMUNES:
  0: persona          1: bicicleta       2: auto            3: motocicleta
  5: bus              7: camión          15: pájaro         16: gato
  17: perro           18: caballo        19: oveja          20: vaca
  59: bed             60: escritorio     61: televisión     62: laptop 
  56: chair           57: piso           58: puerta         59: ventana

IMPORTANTE - Umbral de Confianza (--conf):
  El umbral se aplica en TRES niveles:
  1. Detecciones de segmentación (todas las clases)
  2. Detecciones de personas (pose boxes)
  3. Keypoints individuales (cada punto del skeleton)

  → Threshold bajo (0.25): Más detecciones, más ruido
  → Threshold medio (0.5): Balance óptimo
  → Threshold alto (0.7+): Solo detecciones muy confiables, skeleton limpio

IMPORTANTE - Focus Size:
  - Debe ser múltiplo de 80: 80, 160, 240, 320, 400, 480, 560, 640...
  - Error si no es múltiplo de 80

Estrategias si frame < focus_size:
  - zoom (default): Amplía el frame (scale up) → mejor calidad, puede distorsionar
  - pad: Rellena con negro → mantiene aspect ratio, agrega barras negras
        """
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["s", "m", "l", "x"],
        required=True,
        help="Tamaño del modelo YOLO (s=small, m=medium, l=large, x=xlarge). REQUERIDO."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[160, 192, 224, 256, 288, 320, 640],
        required=True,
        help="Resolución del modelo en píxeles (160-640, múltiplos de 32). REQUERIDO."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Dispositivo OpenVINO (GPU, CPU, AUTO). Default: GPU (iGPU Intel)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/vehicles-1280x720.mp4",
        help="Ruta al video de entrada"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_fp16_segmentation_sv",
        help="Directorio de salida para resultados"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar live preview durante inferencia (útil para debugging)"
    )
    parser.add_argument(
        "--focus-size",
        type=int,
        default=None,
        help="Tamaño del focus lens (crop cuadrado). Ej: 640. Si no se especifica, usa frame completo."
    )
    parser.add_argument(
        "--focus-x",
        type=int,
        default=None,
        help="Posición X del focus lens (None = centrado)"
    )
    parser.add_argument(
        "--focus-y",
        type=int,
        default=None,
        help="Posición Y del focus lens (None = centrado)"
    )
    parser.add_argument(
        "--focus-strategy",
        type=str,
        choices=["zoom", "pad"],
        default="zoom",
        help="Estrategia si frame < focus_size: 'zoom' (ampliar) o 'pad' (rellenar negro). Default: zoom"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="IDs de clases COCO a detectar (ej: 0=persona, 2=auto). Si no se especifica, todas las clases."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza (0.0-1.0). Aplica a: detecciones de objetos, personas Y keypoints individuales. Default: 0.25"
    )
    
    args = parser.parse_args()
    
    # Configuración
    VIDEO_PATH = args.video
    OUTPUT_DIR = Path(args.output)
    SEG_DIR = Path("exports/fp16/sauron_segmentation")
    POSE_DIR = Path("exports/fp16/sauron_pose")
    DEVICE = args.device
    SHOW_LIVE = args.show
    FOCUS_SIZE = args.focus_size
    FOCUS_X = args.focus_x
    FOCUS_Y = args.focus_y
    FOCUS_STRATEGY = args.focus_strategy
    CLASS_FILTER = args.classes
    CONF_THRESHOLD = args.conf
    
    print("🎯 Bakery - FP16 Segmentation + Pose (iGPU)")
    print("=" * 70)
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🖥️  Dispositivo target: {DEVICE}")
    print(f"🎯 Modelo: YOLO11{args.model_size} @ {args.resolution}px")
    if FOCUS_SIZE:
        focus_pos = f"({FOCUS_X}, {FOCUS_Y})" if FOCUS_X is not None and FOCUS_Y is not None else "(centrado)"
        print(f"🎯 Focus Lens: {FOCUS_SIZE}x{FOCUS_SIZE} en {focus_pos} | Estrategia: {FOCUS_STRATEGY}")
    if CLASS_FILTER is not None:
        print(f"🎨 Filtro de clases: {CLASS_FILTER}")
    print(f"📊 Umbral de confianza: {CONF_THRESHOLD} (objetos + personas + keypoints)")
    if SHOW_LIVE:
        print(f"👁️  Live preview: ENABLED")
    
    # Verificar GPU disponible
    core = ov.Core()
    available_devices = core.available_devices
    print(f"\n🔍 Dispositivos OpenVINO disponibles: {available_devices}")
    
    if "GPU" not in available_devices:
        print("\n⚠️  GPU no detectada por OpenVINO")
        print("   Posibles causas:")
        print("   - Drivers Intel no instalados (intel-opencl-icd)")
        print("   - iGPU deshabilitada en BIOS")
        print("   - OpenVINO no compilado con soporte GPU")
    
    # Descubrir modelo de segmentación específico
    try:
        seg_models = discover_segmentation_models(SEG_DIR, args.model_size, args.resolution)
        if not seg_models:
            raise ValueError(f"No se encontraron modelos segmentation para size={args.model_size}, resolution={args.resolution}")

        # Tomar el primer modelo encontrado
        seg_model_path = seg_models[0]
        lens_name = seg_model_path.parent.parent.name
        print(f"\n📦 Modelo segmentation: {lens_name}/{seg_model_path.parent.name}/{seg_model_path.name}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Ejecuta 'uv run export_sauron_segmentation.py' para exportar modelos primero")
        return

    # Descubrir modelo de pose (mismo tamaño y resolución que segmentación)
    pose_model_path = None
    try:
        pose_models = discover_pose_models(POSE_DIR, args.model_size, args.resolution)
        if pose_models:
            pose_model_path = pose_models[0]
            print(f"📦 Modelo pose:         {lens_name}/{pose_model_path.parent.name}/{pose_model_path.name}")
        else:
            print(f"⚠️  No se encontraron modelos pose para size={args.model_size}, resolution={args.resolution}")
            print("   Continuar solo con segmentación (sin skeleton)")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n⚠️  Modelos pose no disponibles: {e}")
        print("   Continuando solo con segmentación (sin skeleton)")
    
    # Fail Fast: Validar video
    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ Error: Video no encontrado: {VIDEO_PATH}")
        return
    
    print("\n✅ Archivos validados")
    print("=" * 70)
    
    # Inferencia con modelos (segmentation + pose)
    try:
        metrics = run_inference_fp16_segmentation_sv_lens(
            seg_model_path, VIDEO_PATH, OUTPUT_DIR, DEVICE, SHOW_LIVE,
            FOCUS_SIZE, FOCUS_X, FOCUS_Y, FOCUS_STRATEGY,
            CLASS_FILTER, CONF_THRESHOLD,
            pose_model_path  # Puede ser None si no hay modelo pose
        )
        
        # Mostrar resumen de métricas
        print("\n" + "=" * 70)
        print("📊 MÉTRICAS DE INFERENCIA")
        print("=" * 70)
        print(f"Modelo:              {metrics['model']}")
        print(f"Device:              {metrics['device']}")
        print(f"Frames procesados:   {metrics['total_frames']:,}")
        print(f"Tiempo total:        {metrics['elapsed_time']:.2f}s")
        print(f"FPS:                 {metrics['fps']:.2f}")
        print(f"Detecciones/frame:   {metrics['avg_detections']:.1f}")
        print(f"Video output:        {metrics['output_path']}")
        
    except Exception as e:
        print(f"❌ Error durante inferencia: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("🎉 Inferencia FP16 Segmentation completada!")
    print(f"\n💡 Para comparar modelos:")
    print(f"   1. Prueba diferentes tamaños: --model-size s, m, l, x")
    print(f"   2. Prueba diferentes resoluciones: --resolution 320, 640, 1280")
    print(f"   3. Compara FPS vs precisión de cada combinación")
    print(f"   Ejemplo: s@320 (rápido) vs m@640 (balanceado) vs l@1280 (preciso)")


if __name__ == "__main__":
    main()

