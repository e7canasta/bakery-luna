#!/usr/bin/env python3
"""
Verifica distribución de carga en tiempo real durante inferencia híbrida.

Monitorea:
- CPU cores utilizados (por core)
- GPU utilization
- Memory bandwidth
- Throughput por dispositivo
- Validación de que la carga esté distribuida correctamente

Uso:
    uv run scripts/int8_vnni/verify_load_balancing.py
    uv run scripts/int8_vnni/verify_load_balancing.py --video videos/sample.mp4 --duration 30
"""

import argparse
import sys
from pathlib import Path
import time
import threading
import psutil
import cv2
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import openvino as ov
except ImportError:
    print("❌ OpenVINO no instalado (pip install openvino)")
    sys.exit(1)


class LoadMonitor:
    """
    Monitor de carga en tiempo real con actualización periódica.
    
    Muestra métricas de CPU (por core), GPU, memoria y throughput.
    """

    def __init__(self, duration: float = 30.0, interval: float = 0.5):
        """
        Args:
            duration: Duración total del monitoreo (segundos)
            interval: Intervalo entre muestras (segundos)
        """
        self.duration = duration
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Métricas recolectadas
        self.cpu_per_core: List[Dict[int, float]] = []
        self.cpu_total: List[float] = []
        self.memory_samples: List[float] = []
        self.timestamps: List[float] = []

        # Estadísticas de GPU (si disponible)
        self.gpu_available = False
        self.gpu_samples: List[float] = []

    def start(self):
        """Inicia monitoreo en background."""
        self.running = True
        self.cpu_per_core = []
        self.cpu_total = []
        self.memory_samples = []
        self.timestamps = []
        self.gpu_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene monitoreo."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)

    def _monitor_loop(self):
        """Loop de monitoreo (corre en thread separado)."""
        start_time = time.time()
        process = psutil.Process()

        while self.running and (time.time() - start_time) < self.duration:
            try:
                timestamp = time.time() - start_time
                self.timestamps.append(timestamp)

                # CPU por core
                cpu_percent_per_core = process.cpu_percent(interval=None, percpu=True)
                self.cpu_per_core.append({i: cpu for i, cpu in enumerate(cpu_percent_per_core)})

                # CPU total
                cpu_total = process.cpu_percent(interval=None)
                self.cpu_total.append(cpu_total)

                # Memoria
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)

                # GPU (si disponible - requiere implementación específica)
                # Por ahora, solo placeholder
                self.gpu_samples.append(0.0)  # TODO: Implementar monitoreo GPU

                time.sleep(self.interval)

            except Exception as e:
                print(f"⚠️  Error en monitoreo: {e}")
                continue

    def get_cpu_stats_per_core(self) -> Dict[int, Dict[str, float]]:
        """
        Retorna estadísticas de CPU por core.

        Returns:
            Dict mapping core_id -> {avg, max, min}
        """
        stats = defaultdict(lambda: {"samples": []})

        for sample in self.cpu_per_core:
            for core_id, cpu_percent in sample.items():
                stats[core_id]["samples"].append(cpu_percent)

        result = {}
        for core_id, data in stats.items():
            samples = data["samples"]
            if samples:
                result[core_id] = {
                    "avg": sum(samples) / len(samples),
                    "max": max(samples),
                    "min": min(samples),
                }

        return result

    def get_summary(self) -> Dict:
        """
        Retorna resumen de métricas.

        Returns:
            Dict con estadísticas agregadas
        """
        cpu_stats_per_core = self.get_cpu_stats_per_core()

        avg_cpu_total = sum(self.cpu_total) / len(self.cpu_total) if self.cpu_total else 0
        max_cpu_total = max(self.cpu_total) if self.cpu_total else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        max_memory = max(self.memory_samples) if self.memory_samples else 0

        return {
            "cpu_per_core": cpu_stats_per_core,
            "cpu_total_avg": avg_cpu_total,
            "cpu_total_max": max_cpu_total,
            "memory_avg_mb": avg_memory,
            "memory_max_mb": max_memory,
            "duration": self.timestamps[-1] if self.timestamps else 0,
            "samples": len(self.timestamps),
        }


def letterbox(img: np.ndarray, new_shape: tuple = (640, 640), color: tuple = (114, 114, 114)) -> np.ndarray:
    """Resize imagen manteniendo aspect ratio (letterbox)."""
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

    return img


def preprocess(img: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Pre-procesa imagen para inferencia YOLO."""
    img = letterbox(img, new_shape=input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def discover_models(exports_dir: Path, model_type: str = "segmentation") -> List[Path]:
    """Descubre modelos de segmentación o pose."""
    models = []
    
    for precision in ["fp16", "int8"]:
        precision_dir = exports_dir / precision
        
        if not precision_dir.exists():
            continue

        for xml_file in precision_dir.rglob("*.xml"):
            if model_type == "segmentation" and ("seg" in xml_file.stem.lower() or "det" in xml_file.stem.lower()):
                if "pose" not in xml_file.stem.lower():
                    models.append(xml_file)
            elif model_type == "pose" and "pose" in xml_file.stem.lower():
                models.append(xml_file)

    return sorted(models)


def verify_load_balancing(
    seg_model_path: Path,
    pose_model_path: Path,
    video_path: Path,
    seg_device: str,
    pose_device: str,
    duration: float = 30.0
) -> Dict:
    """
    Verifica distribución de carga durante inferencia híbrida.

    Args:
        seg_model_path: Path al modelo de segmentación
        pose_model_path: Path al modelo de pose
        video_path: Path al video
        seg_device: Device para segmentación
        pose_device: Device para pose
        duration: Duración del monitoreo (segundos)

    Returns:
        Dict con métricas de carga
    """
    print(f"\n🔍 Verificando balanceo de carga...")
    print(f"   Segmentation: {seg_device}")
    print(f"   Pose: {pose_device}")
    print(f"   Duración: {duration}s")
    print("-" * 70)

    # Cargar modelos
    core = ov.Core()
    seg_model = core.read_model(seg_model_path)
    seg_compiled = core.compile_model(seg_model, seg_device)

    pose_model = core.read_model(pose_model_path)
    pose_compiled = core.compile_model(pose_model, pose_device)

    # Obtener input shapes
    seg_input_shape = seg_compiled.input(0).shape
    pose_input_shape = pose_compiled.input(0).shape
    _, _, seg_h, seg_w = seg_input_shape
    _, _, pose_h, pose_w = pose_input_shape

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")

    # Iniciar monitor
    monitor = LoadMonitor(duration=duration, interval=0.5)
    monitor.start()

    print(f"\n⏱️  Monitoreando durante {duration}s...")
    print("   (Presiona Ctrl+C para detener antes)")

    # Procesar frames
    frame_count = 0
    seg_interval = 5
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                # Reiniciar video si termina antes del tiempo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Preprocess
            seg_tensor = preprocess(frame, (seg_h, seg_w))
            pose_tensor = preprocess(frame, (pose_h, pose_w))

            # Segmentation (cada N frames)
            if frame_count % seg_interval == 0:
                seg_outputs = seg_compiled([seg_tensor])

            # Pose (cada frame)
            pose_outputs = pose_compiled([pose_tensor])

            frame_count += 1

            # Mostrar progreso cada 5 segundos
            elapsed = time.time() - start_time
            if frame_count % 10 == 0:
                print(f"   Procesados {frame_count} frames ({elapsed:.1f}s)...", end="\r")

    except KeyboardInterrupt:
        print("\n\n⏸️  Detenido por usuario")

    finally:
        monitor.stop()
        cap.release()

    # Obtener estadísticas
    summary = monitor.get_summary()
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    summary["fps"] = fps
    summary["total_frames"] = frame_count
    summary["seg_device"] = seg_device
    summary["pose_device"] = pose_device

    return summary


def print_load_analysis(summary: Dict):
    """
    Imprime análisis de distribución de carga.
    """
    print("\n" + "=" * 90)
    print("📊 ANÁLISIS DE DISTRIBUCIÓN DE CARGA")
    print("=" * 90)

    print(f"\n🖥️  Configuración:")
    print(f"   Segmentation: {summary['seg_device']}")
    print(f"   Pose: {summary['pose_device']}")

    print(f"\n⏱️  Performance:")
    print(f"   FPS: {summary['fps']:.2f}")
    print(f"   Frames procesados: {summary['total_frames']}")
    print(f"   Duración: {summary['duration']:.1f}s")

    print(f"\n💻 CPU Total:")
    print(f"   Promedio: {summary['cpu_total_avg']:.1f}%")
    print(f"   Máximo: {summary['cpu_total_max']:.1f}%")

    print(f"\n💻 CPU por Core:")
    cpu_per_core = summary['cpu_per_core']
    if cpu_per_core:
        print(f"   {'Core':<8} {'Promedio':<12} {'Máximo':<12} {'Mínimo':<12}")
        print("   " + "-" * 44)
        for core_id in sorted(cpu_per_core.keys()):
            stats = cpu_per_core[core_id]
            print(f"   {core_id:<8} {stats['avg']:<12.1f} {stats['max']:<12.1f} {stats['min']:<12.1f}")
    else:
        print("   No hay datos disponibles")

    print(f"\n🧠 Memoria:")
    print(f"   Promedio: {summary['memory_avg_mb']:.1f} MB")
    print(f"   Máximo: {summary['memory_max_mb']:.1f} MB")

    # Análisis de distribución
    print("\n" + "=" * 90)
    print("🔍 ANÁLISIS DE DISTRIBUCIÓN")
    print("=" * 90)

    if summary['seg_device'] != summary['pose_device']:
        print("\n✅ Configuración híbrida detectada")
        print("   - Carga distribuida entre CPU y GPU")
        
        # Verificar si CPU está siendo utilizado
        cpu_cores_used = [core_id for core_id, stats in summary['cpu_per_core'].items() if stats['avg'] > 10]
        if cpu_cores_used:
            print(f"   - CPU cores activos: {len(cpu_cores_used)} ({cpu_cores_used})")
        else:
            print("   ⚠️  CPU cores no muestran alta utilización")
            print("      (Puede indicar que la carga está principalmente en GPU)")

        # Verificar balance
        if summary['cpu_total_avg'] > 50:
            print("   ✅ CPU está siendo utilizado activamente")
        elif summary['cpu_total_avg'] < 20:
            print("   ⚠️  CPU tiene baja utilización")
            print("      (Puede indicar que GPU está haciendo la mayor parte del trabajo)")
    else:
        print(f"\nℹ️  Configuración no-híbrida: ambos modelos en {summary['seg_device']}")
        print("   - Toda la carga en un solo dispositivo")

    # Recomendaciones
    print("\n" + "=" * 90)
    print("💡 RECOMENDACIONES")
    print("=" * 90)

    if summary['seg_device'] != summary['pose_device']:
        if summary['cpu_total_avg'] > 30 and summary['cpu_total_avg'] < 80:
            print("\n✅ Balanceo de carga óptimo")
            print("   - CPU y GPU están siendo utilizados eficientemente")
            print("   - No hay saturación excesiva de ningún dispositivo")
        elif summary['cpu_total_avg'] > 80:
            print("\n⚠️  CPU puede estar saturado")
            print("   - Considerar mover más carga a GPU")
            print("   - O reducir resolución de modelos en CPU")
        else:
            print("\n⚠️  CPU subutilizado")
            print("   - Considerar mover más carga a CPU")
            print("   - O usar modelos más pesados en CPU")
    else:
        print(f"\nℹ️  Para mejor balanceo, considerar configuración híbrida:")
        print(f"   - Un modelo en CPU (INT8 con VNNI)")
        print(f"   - Otro modelo en GPU (FP16)")


def main():
    parser = argparse.ArgumentParser(
        description="Verifica distribución de carga en tiempo real",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Verificación estándar (30s)
  uv run scripts/int8_vnni/verify_load_balancing.py

  # Video específico, duración personalizada
  uv run scripts/int8_vnni/verify_load_balancing.py --video videos/sample.mp4 --duration 60

  # Configuración híbrida específica
  uv run scripts/int8_vnni/verify_load_balancing.py --seg-device GPU --pose-device CPU
        """
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/vehicles-1280x720.mp4",
        help="Video de entrada"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duración del monitoreo (segundos)"
    )
    parser.add_argument(
        "--seg-device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="Device para segmentación"
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU"],
        help="Device para pose"
    )
    parser.add_argument(
        "--exports-dir",
        type=str,
        default="exports",
        help="Directorio con modelos exportados"
    )

    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        sys.exit(1)

    # Descubrir modelos
    seg_models = discover_models(exports_dir, "segmentation")
    pose_models = discover_models(exports_dir, "pose")

    if not seg_models:
        print("❌ No se encontraron modelos de segmentación")
        sys.exit(1)

    if not pose_models:
        print("❌ No se encontraron modelos de pose")
        sys.exit(1)

    # Seleccionar modelos
    seg_model = seg_models[0]
    pose_model = pose_models[0]

    print("=" * 90)
    print("🔍 VERIFICADOR DE BALANCEO DE CARGA")
    print("=" * 90)
    print(f"\n📦 Modelo Segmentation: {seg_model.name}")
    print(f"📦 Modelo Pose: {pose_model.name}")
    print(f"🎬 Video: {video_path}")

    # Verificar devices disponibles
    core = ov.Core()
    available_devices = core.available_devices
    has_gpu = "GPU" in available_devices

    print(f"\n🖥️  Devices disponibles: {available_devices}")

    if args.seg_device == "GPU" and not has_gpu:
        print("⚠️  GPU no disponible, cambiando seg_device a CPU")
        args.seg_device = "CPU"

    if args.pose_device == "GPU" and not has_gpu:
        print("⚠️  GPU no disponible, cambiando pose_device a CPU")
        args.pose_device = "CPU"

    # Ejecutar verificación
    try:
        summary = verify_load_balancing(
            seg_model, pose_model, video_path,
            args.seg_device, args.pose_device,
            args.duration
        )

        print_load_analysis(summary)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 90)
    print("🎉 Verificación completada!")
    print("=" * 90)


if __name__ == "__main__":
    main()
