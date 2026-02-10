#!/usr/bin/env python3
"""
Benchmark cruzado: INT8 en CPU vs GPU vs FP16 en GPU.

Compara:
- INT8 CPU (con/sin VNNI)
- INT8 GPU (iGPU Intel Xe-LP)
- FP16 GPU (baseline GPU)

Métricas:
- FPS (frames per second)
- CPU% (utilización de CPU)
- CPU-segundos (carga operacional total)
- Memoria (RAM usage)
- Speedup relativo

Uso:
    uv run benchmark_cross_device.py
    uv run benchmark_cross_device.py --model yolov11n --resolution 320
"""

import argparse
from pathlib import Path
import sys
import time
import threading
import psutil
import cv2
import numpy as np


class SystemMonitor:
    """
    Monitor de sistema en background thread.

    Registra CPU%, memoria y temperatura cada 0.1s.
    """

    def __init__(self, process_pid):
        self.process = psutil.Process(process_pid)
        self.running = False
        self.thread = None

        # Métricas recolectadas
        self.cpu_samples = []
        self.memory_samples = []
        self.temp_samples = []

    def start(self):
        """Inicia monitoreo en background."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene monitoreo."""
        self.running = False
        if self.thread:
            self.thread.join()

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
                    pass  # Temperatura no disponible en este sistema

                time.sleep(0.1)

            except Exception:
                continue

    def get_metrics(self):
        """
        Retorna métricas agregadas.

        Returns:
            dict: Métricas promedio
        """
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        avg_temp = sum(self.temp_samples) / len(self.temp_samples) if self.temp_samples else None

        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "avg_temp_celsius": avg_temp,
        }


def discover_models(exports_dir: Path, precision: str):
    """
    Descubre modelos de una precisión específica.

    Args:
        exports_dir: Path a exports/
        precision: "int8" o "fp16"

    Returns:
        List[dict]: Lista de modelos disponibles
    """
    precision_dir = exports_dir / precision

    if not precision_dir.exists():
        return []

    models = []

    for model_dir in sorted(precision_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir():
                continue

            # Buscar archivo .xml (IR format)
            xml_files = list(variant_dir.glob("*.xml"))
            if not xml_files:
                continue

            xml_path = xml_files[0]

            # Extraer resolución
            parts = xml_path.stem.split("_")
            resolution = None
            for part in parts:
                if part.isdigit():
                    resolution = int(part)
                    break

            models.append({
                "name": model_name,
                "resolution": resolution,
                "path": xml_path,
                "precision": precision,
            })

    return models


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize con aspect ratio mantenido (padding).
    Estándar YOLO.
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img


def preprocess(img, input_shape):
    """
    Pre-procesa frame: letterbox → RGB → normalize → CHW → batch.
    """
    img = letterbox(img, new_shape=(input_shape[2], input_shape[3]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def benchmark_model(model_path: Path, video_path: Path, device: str, precision: str):
    """
    Ejecuta benchmark de un modelo en un device específico.

    Args:
        model_path: Path al modelo (.xml)
        video_path: Path al video de entrada
        device: "CPU" o "GPU"
        precision: "int8" o "fp16"

    Returns:
        dict: Métricas de performance
    """
    try:
        import openvino as ov
    except ImportError:
        print("❌ OpenVINO no instalado")
        sys.exit(1)

    # Load model
    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device)

    # Get input shape
    input_layer = compiled.input(0)
    input_shape = input_layer.shape

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ No se pudo abrir video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iniciar monitor
    monitor = SystemMonitor(psutil.Process().pid)
    monitor.start()

    # Inferencia
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess(frame, input_shape)

        # Inference
        compiled([input_data])

        frame_count += 1

    end_time = time.time()
    elapsed = end_time - start_time

    # Detener monitor
    monitor.stop()
    system_metrics = monitor.get_metrics()

    cap.release()

    # Calcular métricas
    fps = frame_count / elapsed if elapsed > 0 else 0
    cpu_seconds = elapsed * (system_metrics["avg_cpu_percent"] / 100.0)

    return {
        "model_name": model_path.stem,
        "device": device,
        "precision": precision,
        "fps": fps,
        "elapsed_time": elapsed,
        "frames_processed": frame_count,
        "avg_cpu_percent": system_metrics["avg_cpu_percent"],
        "cpu_seconds": cpu_seconds,
        "avg_memory_mb": system_metrics["avg_memory_mb"],
        "avg_temp_celsius": system_metrics["avg_temp_celsius"],
    }


def compare_results(results: list):
    """
    Compara resultados entre devices y precisiones.
    """
    print("\n" + "=" * 90)
    print("📊 COMPARACIÓN CROSS-DEVICE")
    print("=" * 90)

    # Header
    print(f"\n{'Config':<30} {'FPS':>8} {'CPU%':>8} {'CPU-s':>8} {'Mem(MB)':>10} {'Temp°C':>8}")
    print("-" * 90)

    # Rows
    for r in results:
        config = f"{r['precision'].upper()} @ {r['device']}"
        temp_str = f"{r['avg_temp_celsius']:.1f}" if r['avg_temp_celsius'] else "N/A"

        print(
            f"{config:<30} "
            f"{r['fps']:>8.2f} "
            f"{r['avg_cpu_percent']:>8.1f} "
            f"{r['cpu_seconds']:>8.2f} "
            f"{r['avg_memory_mb']:>10.1f} "
            f"{temp_str:>8}"
        )

    # Análisis comparativo
    print("\n" + "=" * 90)
    print("🔍 ANÁLISIS COMPARATIVO")
    print("=" * 90)

    # Mejor FPS
    best_fps = max(results, key=lambda x: x["fps"])
    print(f"\n🏆 Mejor FPS: {best_fps['precision'].upper()} @ {best_fps['device']} ({best_fps['fps']:.2f} FPS)")

    # Menor carga operacional
    best_cpu = min(results, key=lambda x: x["cpu_seconds"])
    print(f"⚡ Menor carga operacional: {best_cpu['precision'].upper()} @ {best_cpu['device']} ({best_cpu['cpu_seconds']:.2f} CPU-s)")

    # Menor memoria
    best_mem = min(results, key=lambda x: x["avg_memory_mb"])
    print(f"🧠 Menor memoria: {best_mem['precision'].upper()} @ {best_mem['device']} ({best_mem['avg_memory_mb']:.1f} MB)")

    # Speedup INT8 vs FP16
    int8_cpu = [r for r in results if r["precision"] == "int8" and r["device"] == "CPU"]
    fp16_gpu = [r for r in results if r["precision"] == "fp16" and r["device"] == "GPU"]

    if int8_cpu and fp16_gpu:
        speedup = int8_cpu[0]["fps"] / fp16_gpu[0]["fps"]
        print(f"\n📈 Speedup INT8 CPU vs FP16 GPU: {speedup:.2f}x")

        if speedup > 1:
            print("   ✅ INT8 CPU es más rápido (probablemente VNNI habilitado)")
        else:
            print("   ⚠️  FP16 GPU es más rápido (GPU saturada eficientemente)")

    # Speedup INT8 GPU vs INT8 CPU
    int8_gpu = [r for r in results if r["precision"] == "int8" and r["device"] == "GPU"]

    if int8_cpu and int8_gpu:
        speedup = int8_gpu[0]["fps"] / int8_cpu[0]["fps"]
        print(f"\n📈 Speedup INT8 GPU vs INT8 CPU: {speedup:.2f}x")

        if speedup > 1.2:
            print("   ✅ GPU tiene ventaja (resolución alta favorece GPU)")
        elif speedup < 0.8:
            print("   ⚠️  CPU es más rápido (overhead GPU > ganancia paralela)")
        else:
            print("   ≈ Performance similar (depende del workload)")

    # Recomendaciones
    print("\n" + "=" * 90)
    print("💡 RECOMENDACIONES")
    print("=" * 90)

    if best_fps["device"] == "CPU" and best_fps["precision"] == "int8":
        print("\n✅ Configuración óptima: INT8 en CPU")
        print("   - VNNI probablemente habilitado")
        print("   - Baja latencia, buen throughput")
        print("   - Recomendado para edge devices")
    elif best_fps["device"] == "GPU":
        print(f"\n✅ Configuración óptima: {best_fps['precision'].upper()} en GPU")
        print("   - GPU saturada eficientemente")
        print("   - Mayor throughput en resoluciones altas")
        print("   - Recomendado para batch processing")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cruzado: INT8 CPU vs GPU vs FP16 GPU"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Modelo específico (ej: yolov11n)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[320, 640],
        help="Resolución específica"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/vehicles-1280x720.mp4",
        help="Video de entrada"
    )

    args = parser.parse_args()

    exports_dir = Path("exports")
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        sys.exit(1)

    # Descubrir modelos INT8 y FP16
    int8_models = discover_models(exports_dir, "int8")
    fp16_models = discover_models(exports_dir, "fp16")

    if not int8_models:
        print("❌ No se encontraron modelos INT8")
        sys.exit(1)

    if not fp16_models:
        print("⚠️  No se encontraron modelos FP16 (solo se testeará INT8)")

    # Filtrar por modelo/resolución si se especificó
    if args.model:
        int8_models = [m for m in int8_models if m["name"] == args.model]
        fp16_models = [m for m in fp16_models if m["name"] == args.model]

    if args.resolution:
        int8_models = [m for m in int8_models if m["resolution"] == args.resolution]
        fp16_models = [m for m in fp16_models if m["resolution"] == args.resolution]

    if not int8_models:
        print("❌ No se encontraron modelos con los filtros especificados")
        sys.exit(1)

    # Tomar primer modelo (misma arquitectura para comparar)
    int8_model = int8_models[0]
    fp16_model = fp16_models[0] if fp16_models else None

    print("=" * 90)
    print("🚀 BENCHMARK CROSS-DEVICE")
    print("=" * 90)
    print(f"\n📦 Modelo: {int8_model['name']}")
    print(f"📐 Resolución: {int8_model['resolution']}")
    print(f"🎬 Video: {video_path}")

    results = []

    # 1. INT8 en CPU
    print("\n1️⃣  Benchmarking INT8 @ CPU...")
    result = benchmark_model(int8_model["path"], video_path, "CPU", "int8")
    if result:
        results.append(result)
        print(f"   ✅ {result['fps']:.2f} FPS, {result['cpu_seconds']:.2f} CPU-s")

    # 2. INT8 en GPU (si disponible)
    print("\n2️⃣  Benchmarking INT8 @ GPU...")
    try:
        result = benchmark_model(int8_model["path"], video_path, "GPU", "int8")
        if result:
            results.append(result)
            print(f"   ✅ {result['fps']:.2f} FPS, {result['cpu_seconds']:.2f} CPU-s")
    except Exception as e:
        print(f"   ⚠️  GPU no disponible o error: {e}")

    # 3. FP16 en GPU (baseline)
    if fp16_model:
        print("\n3️⃣  Benchmarking FP16 @ GPU (baseline)...")
        try:
            result = benchmark_model(fp16_model["path"], video_path, "GPU", "fp16")
            if result:
                results.append(result)
                print(f"   ✅ {result['fps']:.2f} FPS, {result['cpu_seconds']:.2f} CPU-s")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

    # Comparar resultados
    if results:
        compare_results(results)
    else:
        print("\n❌ No se pudo ejecutar ningún benchmark")


if __name__ == "__main__":
    main()
