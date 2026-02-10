"""
Bakery - CPU Load Benchmark
============================
Mide carga operacional del CPU durante inferencia.

Filosofía: "Complejidad por diseño, no por accidente"
- Métricas cuantificables: CPU%, memoria, temperatura
- Comparación objetiva: 320 vs 640, ONNX vs INT8
- Decisiones basadas en datos, no intuición
"""

from pathlib import Path
import openvino as ov
import cv2
import numpy as np
import time
import psutil
import threading
import argparse
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class CPUMetrics:
    """Métricas de carga operacional del CPU"""
    model_name: str
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    elapsed_time: float
    total_frames: int
    fps: float
    cpu_seconds: float  # Tiempo × CPU% promedio
    avg_temp_celsius: float = 0.0  # Opcional


class CPUMonitor:
    """
    Monitor de CPU en background thread.

    Registra CPU%, memoria y temperatura cada 0.1s.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.running = False
        self.thread = None
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.temp_samples: List[float] = []
        self.process = psutil.Process()

    def _monitor_loop(self):
        """Loop de monitoreo en background"""
        while self.running:
            # CPU % (del proceso actual)
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)

            # Memoria (MB)
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
            self.memory_samples.append(memory_mb)

            # Temperatura (si está disponible)
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps and temps['coretemp']:
                    # Promedio de cores
                    avg_temp = sum(t.current for t in temps['coretemp']) / len(temps['coretemp'])
                    self.temp_samples.append(avg_temp)
            except (AttributeError, KeyError):
                # sensors_temperatures no disponible en este sistema
                pass

            time.sleep(self.interval)

    def start(self):
        """Inicia monitoreo"""
        self.running = True
        self.cpu_samples = []
        self.memory_samples = []
        self.temp_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """
        Detiene monitoreo y retorna estadísticas.

        Returns:
            Dict con promedios y máximos
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        stats = {
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'max_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'avg_temp_celsius': sum(self.temp_samples) / len(self.temp_samples) if self.temp_samples else 0,
        }
        return stats


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize imagen manteniendo aspect ratio (letterbox)"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
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


def preprocess(img: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
    """Pre-procesa imagen para inferencia YOLO"""
    img, ratio, (dw, dh) = letterbox(img, new_shape=input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def benchmark_int8(
    model_path: Path,
    video_path: str,
    device: str = "CPU"
) -> CPUMetrics:
    """
    Ejecuta benchmark de carga CPU con modelo INT8.

    Args:
        model_path: Ruta al .xml del modelo OpenVINO IR
        video_path: Ruta al video
        device: Dispositivo OpenVINO (CPU, GPU, etc.)

    Returns:
        CPUMetrics con estadísticas de carga
    """
    # Fail Fast
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    model_name = model_path.stem
    print(f"\n🔬 Benchmark: {model_name} (INT8)")
    print("-" * 60)

    # Cargar modelo
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    _, _, input_h, input_w = input_layer.shape

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iniciar monitor de CPU
    monitor = CPUMonitor(interval=0.1)
    monitor.start()

    # Procesar frames (sin guardar video para benchmark puro)
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process
        input_tensor = preprocess(frame, (input_h, input_w))

        # Inference
        result = compiled_model([input_tensor])[output_layer]

        frame_count += 1

    elapsed_time = time.time() - start_time

    # Detener monitor
    cpu_stats = monitor.stop()

    cap.release()

    # Calcular métricas
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cpu_seconds = elapsed_time * (cpu_stats['avg_cpu_percent'] / 100.0)

    metrics = CPUMetrics(
        model_name=model_name,
        avg_cpu_percent=cpu_stats['avg_cpu_percent'],
        max_cpu_percent=cpu_stats['max_cpu_percent'],
        avg_memory_mb=cpu_stats['avg_memory_mb'],
        max_memory_mb=cpu_stats['max_memory_mb'],
        elapsed_time=elapsed_time,
        total_frames=frame_count,
        fps=fps,
        cpu_seconds=cpu_seconds,
        avg_temp_celsius=cpu_stats['avg_temp_celsius']
    )

    # Print métricas
    print(f"⏱️  Tiempo total: {elapsed_time:.2f}s")
    print(f"🎞️  FPS: {fps:.2f}")
    print(f"💻 CPU promedio: {cpu_stats['avg_cpu_percent']:.1f}%")
    print(f"💻 CPU máximo: {cpu_stats['max_cpu_percent']:.1f}%")
    print(f"🧮 CPU-segundos: {cpu_seconds:.2f}s (carga operacional total)")
    print(f"🧠 Memoria promedio: {cpu_stats['avg_memory_mb']:.1f} MB")
    print(f"🧠 Memoria máxima: {cpu_stats['max_memory_mb']:.1f} MB")
    if cpu_stats['avg_temp_celsius'] > 0:
        print(f"🌡️  Temperatura promedio: {cpu_stats['avg_temp_celsius']:.1f}°C")

    return metrics


def compare_metrics(metrics_list: List[CPUMetrics]) -> None:
    """
    Compara métricas de carga CPU entre modelos.

    Muestra tabla comparativa y análisis.
    """
    print("\n" + "=" * 90)
    print("📊 COMPARACIÓN DE CARGA OPERACIONAL")
    print("=" * 90)
    print(f"{'Modelo':<30} {'FPS':<8} {'CPU%':<8} {'CPU-s':<10} {'Mem(MB)':<10} {'Temp°C':<8}")
    print("-" * 90)

    for m in metrics_list:
        temp_str = f"{m.avg_temp_celsius:.1f}" if m.avg_temp_celsius > 0 else "N/A"
        print(
            f"{m.model_name:<30} "
            f"{m.fps:<8.2f} "
            f"{m.avg_cpu_percent:<8.1f} "
            f"{m.cpu_seconds:<10.2f} "
            f"{m.avg_memory_mb:<10.1f} "
            f"{temp_str:<8}"
        )

    # Análisis
    print("\n" + "=" * 90)
    print("🔍 ANÁLISIS")
    print("=" * 90)

    # Mejor FPS
    best_fps = max(metrics_list, key=lambda x: x.fps)
    print(f"🏆 Mejor FPS: {best_fps.model_name} ({best_fps.fps:.2f} FPS)")

    # Menor carga operacional (CPU-segundos)
    best_load = min(metrics_list, key=lambda x: x.cpu_seconds)
    print(f"⚡ Menor carga operacional: {best_load.model_name} ({best_load.cpu_seconds:.2f} CPU-s)")

    # Menor memoria
    best_mem = min(metrics_list, key=lambda x: x.avg_memory_mb)
    print(f"🧠 Menor memoria: {best_mem.model_name} ({best_mem.avg_memory_mb:.1f} MB)")

    # Comparación 320 vs 640
    models_320 = [m for m in metrics_list if '320' in m.model_name]
    models_640 = [m for m in metrics_list if '640' in m.model_name]

    if models_320 and models_640:
        avg_cpu_320 = sum(m.cpu_seconds for m in models_320) / len(models_320)
        avg_cpu_640 = sum(m.cpu_seconds for m in models_640) / len(models_640)
        ratio = avg_cpu_640 / avg_cpu_320 if avg_cpu_320 > 0 else 0

        print(f"\n📐 Relación 320→640:")
        print(f"   CPU-segundos: 640 requiere {ratio:.2f}x más carga que 320")
        print(f"   (Teórico: 4x píxeles, Real: {ratio:.2f}x carga operacional)")


def main():
    """Pipeline de benchmark de CPU"""

    # Parse argumentos
    parser = argparse.ArgumentParser(description="Benchmark de carga operacional con OpenVINO")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Dispositivo OpenVINO (CPU, GPU, AUTO, etc.)"
    )
    args = parser.parse_args()

    # Configuración
    VIDEO_PATH = "videos/vehicles-1280x720.mp4"
    DEVICE = args.device

    # Modelos INT8 a probar
    MODELS_INT8 = [
        Path("exports/int8/yolov11n_320_int8/yolov11n_320_int8.xml"),
        Path("exports/int8/yolov11n_640_int8/yolov11n_640_int8.xml"),
    ]

    print("🎯 Bakery - Benchmark de Carga Operacional")
    print("=" * 90)
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🖥️  Dispositivo: {DEVICE}")
    print(f"🔧 Modelos INT8: {len(MODELS_INT8)}")
    print("\n⚠️  Nota: Este benchmark mide CARGA OPERACIONAL, no solo tiempo.")
    print("   - CPU%: Utilización promedio del procesador")
    print("   - CPU-segundos: Tiempo × CPU% (carga total)")
    print("   - Memoria: RAM usada por el proceso")

    # Fail Fast: Validar video
    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ Error: Video no encontrado: {VIDEO_PATH}")
        return

    print("\n✅ Video validado")

    # Benchmark por modelo
    all_metrics = []
    for model_path in MODELS_INT8:
        if not model_path.exists():
            print(f"⚠️  Modelo no encontrado: {model_path}")
            continue

        try:
            metrics = benchmark_int8(model_path, VIDEO_PATH, DEVICE)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error con {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Comparar resultados
    if all_metrics:
        compare_metrics(all_metrics)

    print("\n" + "=" * 90)
    print("🎉 Benchmark completado!")


if __name__ == "__main__":
    main()
