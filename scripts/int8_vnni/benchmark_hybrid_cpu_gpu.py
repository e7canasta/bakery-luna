#!/usr/bin/env python3
"""
Benchmark híbrido: Pipeline dual con balanceo de carga CPU/GPU.

Compara diferentes estrategias de distribución de carga:
- Ambos modelos en CPU (INT8)
- Ambos modelos en GPU (FP16)
- Híbrido: Segmentation en GPU (FP16) + Pose en CPU (INT8)
- Híbrido: Segmentation en CPU (INT8) + Pose en GPU (FP16)

Métricas:
- FPS total del pipeline
- CPU% por core
- GPU utilization
- Memoria (RAM + VRAM)
- Temperatura
- Carga operacional por dispositivo

Uso:
    uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py
    uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py --video videos/sample.mp4
"""

import argparse
import sys
from pathlib import Path
import time
import threading
import psutil
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import openvino as ov
except ImportError:
    print("❌ OpenVINO no instalado (pip install openvino)")
    sys.exit(1)


@dataclass
class HybridMetrics:
    """Métricas de pipeline híbrido"""
    config_name: str
    seg_device: str
    pose_device: str
    fps: float
    elapsed_time: float
    total_frames: int
    avg_cpu_percent: float
    max_cpu_percent: float
    cpu_seconds: float
    avg_memory_mb: float
    max_memory_mb: float
    avg_temp_celsius: Optional[float] = None


class SystemMonitor:
    """
    Monitor de sistema en background thread.
    
    Registra CPU%, memoria y temperatura cada 0.1s.
    """

    def __init__(self, process_pid: int):
        self.process = psutil.Process(process_pid)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Métricas recolectadas
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.temp_samples: List[float] = []

    def start(self):
        """Inicia monitoreo en background."""
        self.running = True
        self.cpu_samples = []
        self.memory_samples = []
        self.temp_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene monitoreo."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Loop de monitoreo (corre en thread separado)."""
        while self.running:
            try:
                # CPU% del proceso
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)

                # Memoria del proceso (RSS en MB)
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)

                # Temperatura CPU (si disponible)
                try:
                    temps = psutil.sensors_temperatures()
                    if "coretemp" in temps:
                        # Promedio de cores
                        core_temps = [t.current for t in temps["coretemp"]]
                        avg_temp = sum(core_temps) / len(core_temps)
                        self.temp_samples.append(avg_temp)
                except (AttributeError, KeyError):
                    pass  # Temperatura no disponible

                time.sleep(0.1)

            except Exception:
                continue

    def get_metrics(self) -> Dict:
        """
        Retorna métricas agregadas.

        Returns:
            dict: Métricas promedio
        """
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        max_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_temp = sum(self.temp_samples) / len(self.temp_samples) if self.temp_samples else None

        return {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "avg_temp_celsius": avg_temp,
        }


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
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


def preprocess(img: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
    """Pre-procesa imagen para inferencia YOLO."""
    img = letterbox(img, new_shape=input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def discover_models(exports_dir: Path, model_type: str = "segmentation") -> List[Path]:
    """
    Descubre modelos de segmentación o pose.

    Args:
        exports_dir: Directorio base (exports/)
        model_type: "segmentation" o "pose"

    Returns:
        Lista de paths a archivos .xml
    """
    models = []
    
    # Buscar en fp16 e int8
    for precision in ["fp16", "int8"]:
        precision_dir = exports_dir / precision
        
        if not precision_dir.exists():
            continue

        # Buscar modelos recursivamente
        for xml_file in precision_dir.rglob("*.xml"):
            # Filtrar por tipo
            if model_type == "segmentation" and ("seg" in xml_file.stem.lower() or "det" in xml_file.stem.lower()):
                # Excluir pose
                if "pose" not in xml_file.stem.lower():
                    models.append(xml_file)
            elif model_type == "pose" and "pose" in xml_file.stem.lower():
                models.append(xml_file)

    return sorted(models)


def benchmark_hybrid_pipeline(
    seg_model_path: Path,
    pose_model_path: Path,
    video_path: Path,
    seg_device: str,
    pose_device: str,
    seg_precision: str = "fp16",
    pose_precision: str = "int8"
) -> HybridMetrics:
    """
    Ejecuta benchmark de pipeline dual con diferentes devices.

    Args:
        seg_model_path: Path al modelo de segmentación
        pose_model_path: Path al modelo de pose
        video_path: Path al video
        seg_device: Device para segmentación (CPU, GPU)
        pose_device: Device para pose (CPU, GPU)
        seg_precision: Precisión esperada para seg (fp16, int8)
        pose_precision: Precisión esperada para pose (fp16, int8)

    Returns:
        HybridMetrics con resultados
    """
    # Fail Fast
    if not seg_model_path.exists():
        raise FileNotFoundError(f"Modelo seg no encontrado: {seg_model_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Modelo pose no encontrado: {pose_model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    config_name = f"Seg@{seg_device}({seg_precision.upper()}) + Pose@{pose_device}({pose_precision.upper()})"
    print(f"\n🔬 Benchmark: {config_name}")
    print("-" * 70)

    # Cargar modelos
    core = ov.Core()
    
    # Compilar modelos en sus respectivos devices
    seg_model = core.read_model(seg_model_path)
    try:
        seg_compiled = core.compile_model(seg_model, seg_device)
        print(f"   ✅ Segmentation compilado en {seg_device}")
    except Exception as e:
        print(f"   ❌ Error compilando seg en {seg_device}: {e}")
        raise

    pose_model = core.read_model(pose_model_path)
    try:
        pose_compiled = core.compile_model(pose_model, pose_device)
        print(f"   ✅ Pose compilado en {pose_device}")
    except Exception as e:
        print(f"   ❌ Error compilando pose en {pose_device}: {e}")
        raise

    # Obtener input shapes
    seg_input_shape = seg_compiled.input(0).shape
    pose_input_shape = pose_compiled.input(0).shape
    _, _, seg_h, seg_w = seg_input_shape
    _, _, pose_h, pose_w = pose_input_shape

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iniciar monitor
    monitor = SystemMonitor(psutil.Process().pid)
    monitor.start()

    # Procesar frames (simulando pipeline dual con seg_interval=5)
    start_time = time.time()
    frame_count = 0
    seg_interval = 5  # Run segmentation every 5 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess para ambos modelos
        seg_tensor = preprocess(frame, (seg_h, seg_w))
        pose_tensor = preprocess(frame, (pose_h, pose_w))

        # Segmentation (cada N frames)
        if frame_count % seg_interval == 0:
            seg_outputs = seg_compiled([seg_tensor])
            # No postprocess, solo medir inferencia

        # Pose (cada frame)
        pose_outputs = pose_compiled([pose_tensor])
        # No postprocess, solo medir inferencia

        frame_count += 1

    elapsed_time = time.time() - start_time

    # Detener monitor
    monitor.stop()
    system_metrics = monitor.get_metrics()

    cap.release()

    # Calcular métricas
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cpu_seconds = elapsed_time * (system_metrics["avg_cpu_percent"] / 100.0)

    return HybridMetrics(
        config_name=config_name,
        seg_device=seg_device,
        pose_device=pose_device,
        fps=fps,
        elapsed_time=elapsed_time,
        total_frames=frame_count,
        avg_cpu_percent=system_metrics["avg_cpu_percent"],
        max_cpu_percent=system_metrics["max_cpu_percent"],
        cpu_seconds=cpu_seconds,
        avg_memory_mb=system_metrics["avg_memory_mb"],
        max_memory_mb=system_metrics["max_memory_mb"],
        avg_temp_celsius=system_metrics["avg_temp_celsius"]
    )


def compare_hybrid_results(results: List[HybridMetrics]) -> None:
    """
    Compara resultados de diferentes configuraciones híbridas.
    """
    print("\n" + "=" * 90)
    print("📊 COMPARACIÓN DE CONFIGURACIONES HÍBRIDAS")
    print("=" * 90)

    # Header
    print(f"\n{'Configuración':<50} {'FPS':>8} {'CPU%':>8} {'CPU-s':>8} {'Mem(MB)':>10} {'Temp°C':>8}")
    print("-" * 90)

    # Rows
    for r in results:
        temp_str = f"{r.avg_temp_celsius:.1f}" if r.avg_temp_celsius else "N/A"
        print(
            f"{r.config_name:<50} "
            f"{r.fps:>8.2f} "
            f"{r.avg_cpu_percent:>8.1f} "
            f"{r.cpu_seconds:>8.2f} "
            f"{r.avg_memory_mb:>10.1f} "
            f"{temp_str:>8}"
        )

    # Análisis
    print("\n" + "=" * 90)
    print("🔍 ANÁLISIS")
    print("=" * 90)

    # Mejor FPS
    best_fps = max(results, key=lambda x: x.fps)
    print(f"\n🏆 Mejor FPS: {best_fps.config_name} ({best_fps.fps:.2f} FPS)")

    # Menor carga operacional
    best_cpu = min(results, key=lambda x: x.cpu_seconds)
    print(f"⚡ Menor carga operacional: {best_cpu.config_name} ({best_cpu.cpu_seconds:.2f} CPU-s)")

    # Menor memoria
    best_mem = min(results, key=lambda x: x.avg_memory_mb)
    print(f"🧠 Menor memoria: {best_mem.config_name} ({best_mem.avg_memory_mb:.1f} MB)")

    # Comparar híbrido vs no-híbrido
    hybrid_configs = [r for r in results if r.seg_device != r.pose_device]
    non_hybrid_configs = [r for r in results if r.seg_device == r.pose_device]

    if hybrid_configs and non_hybrid_configs:
        best_hybrid = max(hybrid_configs, key=lambda x: x.fps)
        best_non_hybrid = max(non_hybrid_configs, key=lambda x: x.fps)

        speedup = best_hybrid.fps / best_non_hybrid.fps if best_non_hybrid.fps > 0 else 0
        print(f"\n📈 Híbrido vs No-híbrido:")
        print(f"   Mejor híbrido: {best_hybrid.fps:.2f} FPS")
        print(f"   Mejor no-híbrido: {best_non_hybrid.fps:.2f} FPS")
        print(f"   Speedup: {speedup:.2f}x")

        if speedup > 1.1:
            print("   ✅ Híbrido es más rápido (carga distribuida eficientemente)")
        elif speedup < 0.9:
            print("   ⚠️  No-híbrido es más rápido (overhead de múltiples devices)")
        else:
            print("   ≈ Performance similar")

    # Recomendaciones
    print("\n" + "=" * 90)
    print("💡 RECOMENDACIONES")
    print("=" * 90)

    if best_fps.seg_device != best_fps.pose_device:
        print(f"\n✅ Configuración híbrida óptima: {best_fps.config_name}")
        print("   - Carga distribuida entre CPU y GPU")
        print("   - Reduce saturación de un solo dispositivo")
        print("   - Recomendado para pipelines duales en producción")
    else:
        print(f"\n✅ Configuración óptima: {best_fps.config_name}")
        print("   - Ambos modelos en el mismo dispositivo")
        print("   - Menor overhead de transferencia")
        print("   - Recomendado si un dispositivo es claramente superior")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark híbrido: Pipeline dual con balanceo CPU/GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Benchmark completo (todas las configuraciones)
  uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py

  # Video específico
  uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py --video videos/sample.mp4

  # Solo configuraciones híbridas
  uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py --hybrid-only
        """
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/vehicles-1280x720.mp4",
        help="Video de entrada"
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Solo probar configuraciones híbridas (CPU+GPU)"
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

    # Seleccionar modelos (preferir INT8 para pose, FP16 para seg)
    seg_model = None
    pose_model = None

    # Buscar FP16 para seg, INT8 para pose
    for model in seg_models:
        if "fp16" in str(model) or "FP16" in str(model):
            seg_model = model
            break
    if not seg_model:
        seg_model = seg_models[0]

    for model in pose_models:
        if "int8" in str(model) or "INT8" in str(model):
            pose_model = model
            break
    if not pose_model:
        pose_model = pose_models[0]

    print("=" * 90)
    print("🚀 BENCHMARK HÍBRIDO CPU/GPU")
    print("=" * 90)
    print(f"\n📦 Modelo Segmentation: {seg_model.name}")
    print(f"📦 Modelo Pose: {pose_model.name}")
    print(f"🎬 Video: {video_path}")

    results = []

    # Verificar devices disponibles
    core = ov.Core()
    available_devices = core.available_devices
    has_gpu = "GPU" in available_devices

    print(f"\n🖥️  Devices disponibles: {available_devices}")
    if not has_gpu:
        print("⚠️  GPU no disponible, solo se probarán configuraciones CPU")

    # Configuraciones a probar
    configs = []

    if not args.hybrid_only:
        # 1. Ambos en CPU (INT8)
        configs.append(("CPU", "CPU", "int8", "int8", "CPU+CPU (INT8)"))

        # 2. Ambos en GPU (FP16) - solo si GPU disponible
        if has_gpu:
            configs.append(("GPU", "GPU", "fp16", "fp16", "GPU+GPU (FP16)"))

    # 3. Híbrido: Seg GPU + Pose CPU
    if has_gpu:
        configs.append(("GPU", "CPU", "fp16", "int8", "Seg@GPU + Pose@CPU"))

    # 4. Híbrido: Seg CPU + Pose GPU
    if has_gpu:
        configs.append(("CPU", "GPU", "int8", "fp16", "Seg@CPU + Pose@GPU"))

    print(f"\n📋 Configuraciones a probar: {len(configs)}")
    for i, (seg_dev, pose_dev, seg_prec, pose_prec, name) in enumerate(configs, 1):
        print(f"   {i}. {name}")

    # Ejecutar benchmarks
    for seg_dev, pose_dev, seg_prec, pose_prec, name in configs:
        try:
            # Ajustar paths según precisión
            if seg_prec == "fp16":
                seg_path = seg_model if "fp16" in str(seg_model) else seg_models[0]
            else:
                seg_path = seg_model if "int8" in str(seg_model) else seg_models[0]

            if pose_prec == "fp16":
                pose_path = pose_model if "fp16" in str(pose_model) else pose_models[0]
            else:
                pose_path = pose_model if "int8" in str(pose_model) else pose_models[0]

            result = benchmark_hybrid_pipeline(
                seg_path, pose_path, video_path,
                seg_dev, pose_dev, seg_prec, pose_prec
            )
            results.append(result)
            print(f"   ✅ {result.fps:.2f} FPS, {result.cpu_seconds:.2f} CPU-s")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Comparar resultados
    if results:
        compare_hybrid_results(results)
    else:
        print("\n❌ No se pudo ejecutar ningún benchmark")

    print("\n" + "=" * 90)
    print("🎉 Benchmark completado!")
    print("=" * 90)


if __name__ == "__main__":
    main()
