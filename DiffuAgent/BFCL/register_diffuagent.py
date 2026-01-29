#!/usr/bin/env python3
"""
Register DiffuAgent models in BFCL MODEL_CONFIG_MAPPING

Run this script to apply the DiffuAgent model registration patch.
"""

import os
import sys
import shutil
from datetime import datetime


def backup_file(filepath):
    """Backup a file with timestamp."""
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def register_diffuagent():
    """Register DiffuAgent models in MODEL_CONFIG_MAPPING."""

    print("="*60)
    print("Registering DiffuAgent Models")
    print("="*60)
    print()

    # Check we're in the right directory
    model_config_path = "bfcl_eval/constants/model_config.py"
    if not os.path.exists(model_config_path):
        print(f"❌ ERROR: {model_config_path} not found")
        print("Please run this script from the BFCL root directory")
        return False

    print(f"✓ Found {model_config_path}")
    print()

    # Backup original file
    print("[Step 1] Backing up original file...")
    backup_path = backup_file(model_config_path)
    print(f"✓ Backed up to: {backup_path}")
    print()

    # Read the file
    with open(model_config_path, 'r') as f:
        content = f.read()

    # Check if already registered
    if 'diffuagent_model_map' in content:
        print("⚠ DiffuAgent models already registered")
        print("Skipping registration...")
        print()
        return True

    # Find MODEL_CONFIG_MAPPING line
    print("[Step 2] Adding DiffuAgent registration...")
    lines = content.split('\n')
    model_config_idx = None

    for i, line in enumerate(lines):
        if line.startswith('MODEL_CONFIG_MAPPING = {'):
            model_config_idx = i
            break

    if model_config_idx is None:
        print("❌ ERROR: Cannot find MODEL_CONFIG_MAPPING")
        return False

    print(f"  Found MODEL_CONFIG_MAPPING at line {model_config_idx + 1}")

    # Prepare the registration code
    registration_code = '''
# DiffuAgent model configurations
from bfcl_eval.build_handlers_diffuagent import add_diffuagent_model_configs
diffuagent_model_map = add_diffuagent_model_configs()
'''

    # Insert registration code before MODEL_CONFIG_MAPPING
    lines.insert(model_config_idx, registration_code.strip())

    # Add **diffuagent_model_map to the merge
    # Find the MODEL_CONFIG_MAPPING line again (it moved)
    for i, line in enumerate(lines):
        if line.startswith('MODEL_CONFIG_MAPPING = {'):
            model_config_idx = i
            # Check if next line has the merge
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check if diffuagent is already in the merge
                if 'diffuagent_model_map' not in next_line:
                    # Insert **diffuagent_model_map at the start of the merge
                    lines[i + 1] = lines[i + 1].replace(
                        '**api_inference_model_map,',
                        '**diffuagent_model_map,\n    **api_inference_model_map,'
                    )
            break

    # Write back
    with open(model_config_path, 'w') as f:
        f.write('\n'.join(lines))

    print("✓ Added DiffuAgent registration")
    print()

    # Verify
    print("[Step 3] Verifying installation...")
    try:
        # Clear module cache
        if 'bfcl_eval.constants.model_config' in sys.modules:
            del sys.modules['bfcl_eval.constants.model_config']

        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        # Count DiffuAgent models
        diffuagent_models = [k for k in MODEL_CONFIG_MAPPING.keys() if 'diffuagent' in k.lower()]

        if len(diffuagent_models) > 0:
            print(f"✓ Found {len(diffuagent_models)} DiffuAgent models")
            print()
            print("Sample models:")
            for model in sorted(diffuagent_models)[:5]:
                handler = MODEL_CONFIG_MAPPING[model].model_handler
                print(f"  - {model} -> {handler.__name__}")

            if len(diffuagent_models) > 5:
                print(f"  ... and {len(diffuagent_models) - 5} more")
            print()
            print("="*60)
            print("✓ Installation Successful!")
            print("="*60)
            print()
            print("You can now run:")
            print("  bfcl generate --model diffuagent-chatbase/qwen3-8b ...")
            print()
            print(f"To undo: mv {backup_path} {model_config_path}")
            return True
        else:
            print("✗ ERROR: No DiffuAgent models found")
            return False

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
        print(f"Restoring backup: {backup_path}")
        shutil.copy2(backup_path, model_config_path)
        return False


if __name__ == "__main__":
    success = register_diffuagent()
    sys.exit(0 if success else 1)
