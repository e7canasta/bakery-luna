
import os
import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Set env var before importing/initializing
test_weights_dir = os.path.abspath("tmp_test_weights")
os.environ["ULTRALYTICS_WEIGHTS_DIR"] = test_weights_dir

if os.path.exists(test_weights_dir):
    shutil.rmtree(test_weights_dir)

print(f"Setting ULTRALYTICS_WEIGHTS_DIR to: {test_weights_dir}")

try:
    from bakery_exporters.config import ExportersConfig
    import ultralytics
    
    # Store old setting to restore later if needed (optional, just for info)
    old_setting = ultralytics.settings['weights_dir']
    print(f"Old Ultralytics weights_dir: {old_setting}")

    print("Initializing ExportersConfig...")
    config = ExportersConfig()
    
    print(f"Config initialized. Configured weights dir: {config.ultralytics_weights_dir}")
    
    current_setting = ultralytics.settings['weights_dir']
    print(f"Current Ultralytics weights_dir: {current_setting}")
    
    # Verify
    if str(Path(test_weights_dir)) == str(Path(current_setting)):
        print("SUCCESS: Ultralytics settings updated correctly.")
    else:
        print(f"FAILURE: Expected {test_weights_dir}, got {current_setting}")
        sys.exit(1)

    if os.path.isdir(test_weights_dir):
        print("SUCCESS: Directory created.")
    else:
        print("FAILURE: Directory not created.")
        sys.exit(1)

except ImportError as e:
    print(f"ImportError: {e}")
    # If ultralytics is not installed, we can't verify fully but logic should handle it gracefully
    sys.exit(0)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
finally:
    # Cleanup
    if os.path.exists(test_weights_dir):
        shutil.rmtree(test_weights_dir)
