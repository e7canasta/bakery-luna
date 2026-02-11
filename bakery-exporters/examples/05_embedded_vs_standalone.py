"""
Example 5: Embedded vs Standalone Configuration

Demonstrates how bakery-exporters handles configuration in two modes:
1. Embedded: Using bakery.catalog.config within bakery-luna
2. Standalone: Using bakery_exporters.config independently

Usage:
    python examples/05_embedded_vs_standalone.py
"""

from pathlib import Path
import sys


def show_embedded_mode():
    """Configuration mode 1: Embedded within bakery-luna"""
    print("\n" + "=" * 70)
    print("Mode 1: EMBEDDED (within bakery-luna)")
    print("=" * 70)

    try:
        from bakery.catalog import config as catalog_config
        from bakery_exporters import ExportPipeline

        print("\nConfiguration Source: bakery.catalog.BakeryConfig")
        print(f"  models_dir: {catalog_config.models_dir}")
        print(f"  calibration_dir: {catalog_config.calibration_dir}")
        print(f"  default_yolo_version: {catalog_config.default_yolo_version}")
        print(f"  onnx_opset: {catalog_config.onnx_opset}")

        print("\nBenefit: Single source of truth across entire bakery-luna")
        print("Use case: Running within bakery-luna application")

        # Can use in pipeline
        pipeline = ExportPipeline()
        print(f"\n✓ ExportPipeline created with catalog config")

    except ImportError as e:
        print(f"\n⚠️  bakery.catalog not available: {e}")
        print("   (This is OK if running bakery-exporters standalone)")


def show_standalone_mode():
    """Configuration mode 2: Standalone bakery-exporters"""
    print("\n" + "=" * 70)
    print("Mode 2: STANDALONE (independent package)")
    print("=" * 70)

    from bakery_exporters import config as exporters_config
    from bakery_exporters import ExportPipeline, ExportersConfig

    print("\nConfiguration Source: bakery_exporters.ExportersConfig")
    print(f"  models_dir: {exporters_config.models_dir}")
    print(f"  calibration_dir: {exporters_config.calibration_dir}")
    print(f"  cache_dir: {exporters_config.cache_dir}")
    print(f"  temp_dir: {exporters_config.temp_dir}")
    print(f"  default_yolo_version: {exporters_config.default_yolo_version}")
    print(f"  onnx_opset: {exporters_config.onnx_opset}")
    print(f"  int8_preset: {exporters_config.int8_preset}")

    print("\nConfiguration Sources (priority order):")
    print("  1. Runtime assignment (highest)")
    print("  2. Environment variables (BAKERY_*)")
    print("  3. .env file")
    print("  4. Default values (lowest)")

    # Show how to override
    print("\nExample: Override at runtime")
    original = exporters_config.models_dir
    exporters_config.models_dir = Path("/custom/models")
    print(f"  Before: {original}")
    print(f"  After: {exporters_config.models_dir}")
    exporters_config.models_dir = original

    # Show export
    print("\nConfiguration as dictionary:")
    config_dict = exporters_config.to_dict()
    for key, value in list(config_dict.items())[:5]:
        print(f"  {key}: {value}")
    print(f"  ... ({len(config_dict)} total keys)")

    print("\nBenefit: No bakery-luna dependency required")
    print("Use case: Standalone tool, other projects, CI/CD pipelines")

    # Can use in pipeline
    pipeline = ExportPipeline()
    print(f"\n✓ ExportPipeline created with exporters config")


def show_config_conversion():
    """Show how to convert between configs"""
    print("\n" + "=" * 70)
    print("BONUS: Converting Between Configs")
    print("=" * 70)

    try:
        from bakery.catalog import config as catalog_config
        from bakery_exporters import ExportersConfig

        print("\nConverting catalog config to exporters config:")
        print(f"  Source: bakery.catalog.BakeryConfig")

        exporter_config = ExportersConfig.from_catalog(catalog_config)

        print(f"  Result: bakery_exporters.ExportersConfig")
        print(f"    models_dir: {exporter_config.models_dir}")
        print(f"    onnx_opset: {exporter_config.onnx_opset}")

        print("\n✓ Useful when embedding bakery-exporters in other contexts")

    except ImportError:
        print("\n⚠️  bakery.catalog not available for conversion")


def main():
    """Show both configuration modes"""
    print("\nBakery Exporters - Configuration Modes")
    print("=" * 70)

    # Show embedded mode
    show_embedded_mode()

    # Show standalone mode
    show_standalone_mode()

    # Show conversion
    show_config_conversion()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Bakery Exporters supports two configuration modes:

1. EMBEDDED (within bakery-luna)
   - Uses: bakery.catalog.BakeryConfig
   - Env vars: BAKERY_*
   - Benefits: Single source of truth
   - Use: Inside bakery-luna application

2. STANDALONE (independent)
   - Uses: bakery_exporters.ExportersConfig
   - Env vars: BAKERY_*, BAKERY_EXPORTERS_*
   - Benefits: No dependencies, flexible
   - Use: Standalone tools, other projects

Both modes:
- Load from .env files (searches up 3 levels)
- Support environment variables
- Support runtime configuration
- Provide same export functionality
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
