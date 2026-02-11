#!/usr/bin/env python3
"""
Bakery - Extract Calibration Frames for INT8 Quantization
==========================================================

Extrae frames de un video o directorio de imágenes para calibración INT8 con NNCF.

El dataset de calibración debe ser representativo del dominio de inferencia
para obtener rangos de cuantización óptimos y preservar accuracy.

Uso:
    # Extraer 200 frames de un video
    uv run extract_calibration_frames.py --source videos/sample.mp4 --num-frames 200

    # Extraer con resolución específica
    uv run extract_calibration_frames.py --source videos/sample.mp4 --resolution 640

    # Extraer de múltiples videos
    uv run extract_calibration_frames.py --source videos/ --num-frames 300

    # Usar selección aleatoria en lugar de uniforme
    uv run extract_calibration_frames.py --source videos/sample.mp4 --strategy random

    # Saltar frames muy similares
    uv run extract_calibration_frames.py --source videos/sample.mp4 --skip-similar 0.95

Referencias:
    - NNCF: https://github.com/openvinotoolkit/nncf
    - Spec: .docs/next/int8_calibration/INT8_CALIBRATION_SPEC.md
"""

import argparse
from pathlib import Path
from typing import List, Optional, Iterator
import numpy as np
import cv2
from tqdm import tqdm


def letterbox_preprocess(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Preprocesa imagen para YOLO (letterbox resize).

    Mantiene aspect ratio y añade padding gris (114) para completar el cuadrado.
    Este preprocesamiento debe ser IDÉNTICO al usado en inferencia.

    Args:
        image: Imagen BGR de OpenCV [H, W, 3]
        target_size: Tamaño objetivo (320 o 640)

    Returns:
        Tensor preprocesado [1, 3, target_size, target_size], float32, [0, 1]
    """
    h, w = image.shape[:2]

    # Calcular escala manteniendo aspect ratio
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize con interpolación bilineal
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crear canvas con padding gris (114 es el valor estándar de YOLO)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Centrar imagen en el canvas
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Convertir BGR -> RGB
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Convertir a tensor normalizado [0, 1]
    tensor = canvas.astype(np.float32) / 255.0

    # HWC -> CHW (OpenVINO/PyTorch format)
    tensor = tensor.transpose(2, 0, 1)

    # Añadir dimensión de batch
    tensor = np.expand_dims(tensor, 0)

    return tensor  # [1, 3, H, W]


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula similitud entre dos imágenes usando histogramas.

    Args:
        img1: Primera imagen BGR
        img2: Segunda imagen BGR

    Returns:
        Similitud en rango [0, 1] (1 = idénticas)
    """
    # Convertir a escala de grises para comparación rápida
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcular histogramas
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalizar
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Comparar usando correlación
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return max(0.0, similarity)  # Asegurar [0, 1]


def get_video_frames(
    video_path: Path,
    num_frames: int,
    strategy: str = "uniform",
    skip_similar: Optional[float] = None
) -> Iterator[np.ndarray]:
    """
    Extrae frames de un video.

    Args:
        video_path: Ruta al archivo de video
        num_frames: Número de frames a extraer
        strategy: "uniform" (distribuidos) o "random" (aleatorio)
        skip_similar: Umbral de similitud para saltar frames (0-1), None para desactivar

    Yields:
        Frames BGR como numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   Video: {video_path.name}")
    print(f"   Total frames: {total_frames}, FPS: {fps:.1f}")

    if total_frames < num_frames:
        print(f"   ⚠️  Video tiene menos frames ({total_frames}) que los solicitados ({num_frames})")
        num_frames = total_frames

    # Seleccionar índices según estrategia
    if strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif strategy == "random":
        indices = np.sort(np.random.choice(total_frames, num_frames, replace=False))
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    # Extraer frames
    prev_frame = None
    extracted = 0
    skipped = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Verificar similitud si está habilitado
        if skip_similar is not None and prev_frame is not None:
            similarity = compute_histogram_similarity(frame, prev_frame)
            if similarity > skip_similar:
                skipped += 1
                continue

        prev_frame = frame.copy()
        extracted += 1
        yield frame

    cap.release()

    if skipped > 0:
        print(f"   ℹ️  Frames saltados por similitud: {skipped}")


def get_image_files(source_dir: Path) -> List[Path]:
    """
    Obtiene lista de archivos de imagen en un directorio.

    Args:
        source_dir: Directorio con imágenes

    Returns:
        Lista de paths a imágenes
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []

    for ext in extensions:
        files.extend(source_dir.glob(f"*{ext}"))
        files.extend(source_dir.glob(f"*{ext.upper()}"))

    return sorted(files)


def get_video_files(source_dir: Path) -> List[Path]:
    """
    Obtiene lista de archivos de video en un directorio.

    Args:
        source_dir: Directorio con videos

    Returns:
        Lista de paths a videos
    """
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = []

    for ext in extensions:
        files.extend(source_dir.glob(f"*{ext}"))
        files.extend(source_dir.glob(f"*{ext.upper()}"))

    return sorted(files)


def extract_from_images(
    image_files: List[Path],
    num_frames: int,
    strategy: str = "uniform"
) -> Iterator[np.ndarray]:
    """
    Extrae frames de una lista de imágenes.

    Args:
        image_files: Lista de paths a imágenes
        num_frames: Número de frames a extraer
        strategy: "uniform" o "random"

    Yields:
        Imágenes BGR como numpy arrays
    """
    total = len(image_files)

    if total < num_frames:
        print(f"   ⚠️  Menos imágenes ({total}) que las solicitadas ({num_frames})")
        num_frames = total

    # Seleccionar índices
    if strategy == "uniform":
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.sort(np.random.choice(total, num_frames, replace=False))

    for idx in indices:
        img = cv2.imread(str(image_files[idx]))
        if img is not None:
            yield img


def validate_output(output_dir: Path, resolution: int, expected_count: int) -> bool:
    """
    Valida el dataset de calibración generado.

    Args:
        output_dir: Directorio con archivos .npy
        resolution: Resolución esperada
        expected_count: Número esperado de samples

    Returns:
        True si válido, False si hay errores
    """
    files = list(output_dir.glob("*.npy"))

    if len(files) < expected_count * 0.9:  # Permitir 10% de pérdida
        print(f"   ⚠️  Pocos samples: {len(files)} (esperados: {expected_count})")
        return False

    # Verificar algunos samples
    for f in files[:5]:
        arr = np.load(f)

        if arr.shape != (1, 3, resolution, resolution):
            print(f"   ❌ Shape incorrecto en {f.name}: {arr.shape}")
            return False

        if arr.dtype != np.float32:
            print(f"   ❌ Dtype incorrecto en {f.name}: {arr.dtype}")
            return False

        if arr.min() < 0 or arr.max() > 1:
            print(f"   ❌ Rango incorrecto en {f.name}: [{arr.min():.3f}, {arr.max():.3f}]")
            return False

    print(f"   ✅ Dataset válido: {len(files)} samples")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames para calibración INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Extraer de un video
  uv run extract_calibration_frames.py --source videos/sample.mp4 --num-frames 200

  # Extraer de múltiples videos
  uv run extract_calibration_frames.py --source videos/ --num-frames 300

  # Extraer de imágenes
  uv run extract_calibration_frames.py --source images/ --num-frames 200

  # Con resolución 640 y selección aleatoria
  uv run extract_calibration_frames.py --source videos/sample.mp4 --resolution 640 --strategy random
        """
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Video, directorio de videos, o directorio de imágenes"
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="Número de frames a extraer (default: 200)"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=320,
        choices=[192, 256, 320, 480, 640],
        help="Resolución del modelo (default: 320)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration_data"),
        help="Directorio de salida (default: calibration_data/)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["uniform", "random"],
        default="uniform",
        help="Estrategia de selección: uniform (distribuidos) o random (default: uniform)"
    )

    parser.add_argument(
        "--skip-similar",
        type=float,
        default=None,
        help="Umbral de similitud para saltar frames (0-1). Ej: 0.95 salta frames muy similares"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )

    args = parser.parse_args()

    # Configurar semilla
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("🎯 Bakery - Extract Calibration Frames")
    print("=" * 60)

    # Validar source
    if not args.source.exists():
        print(f"❌ Fuente no encontrada: {args.source}")
        return 1

    # Crear directorio de salida
    output_dir = args.output / f"preprocessed_{args.resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Fuente: {args.source}")
    print(f"📐 Resolución: {args.resolution}x{args.resolution}")
    print(f"🔢 Frames a extraer: {args.num_frames}")
    print(f"📁 Salida: {output_dir}")
    print(f"🎲 Estrategia: {args.strategy}")

    # Recopilar frames
    frames = []

    if args.source.is_file():
        # Un solo video
        print(f"\n📹 Procesando video...")
        frames = list(get_video_frames(
            args.source,
            args.num_frames,
            args.strategy,
            args.skip_similar
        ))

    elif args.source.is_dir():
        # Directorio - verificar contenido
        video_files = get_video_files(args.source)
        image_files = get_image_files(args.source)

        if video_files:
            print(f"\n📹 Encontrados {len(video_files)} videos")
            frames_per_video = max(1, args.num_frames // len(video_files))

            for video in video_files:
                video_frames = list(get_video_frames(
                    video,
                    frames_per_video,
                    args.strategy,
                    args.skip_similar
                ))
                frames.extend(video_frames)

        elif image_files:
            print(f"\n🖼️  Encontradas {len(image_files)} imágenes")
            frames = list(extract_from_images(
                image_files,
                args.num_frames,
                args.strategy
            ))

        else:
            print(f"❌ No se encontraron videos ni imágenes en {args.source}")
            return 1

    print(f"\n📦 Frames recopilados: {len(frames)}")

    # Preprocesar y guardar
    print(f"\n🔧 Preprocesando y guardando...")

    for i, frame in enumerate(tqdm(frames, desc="Preprocesando")):
        # Aplicar letterbox
        tensor = letterbox_preprocess(frame, args.resolution)

        # Guardar como .npy
        output_path = output_dir / f"frame_{i:04d}.npy"
        np.save(output_path, tensor)

    # Validar resultado
    print(f"\n🔍 Validando dataset...")
    valid = validate_output(output_dir, args.resolution, len(frames))

    if valid:
        print(f"\n✅ Dataset de calibración generado exitosamente")
        print(f"   📁 {output_dir}")
        print(f"   📦 {len(list(output_dir.glob('*.npy')))} samples")
        print(f"\n💡 Próximo paso:")
        print(f"   uv run calibrate_int8.py --model yolo11n-seg --resolution {args.resolution}")
    else:
        print(f"\n⚠️  Dataset generado con advertencias")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
