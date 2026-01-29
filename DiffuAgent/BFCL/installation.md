# Installation Guide for Berkeley Function Call Leaderboard (BFCL)

This guide provides step-by-step instructions for setting up BFCL in the unified environment with sparse checkout.

## Prerequisites

- Git (version 2.25+ for sparse checkout support)
- Python 3.10+
- No pip installation required (code-only setup)

## Step 1: Clone this repository

```bash
git clone git@github.com:Coldmist-Lu/DiffuAgent.git
cd DiffuAgent
```

## Step 2: Create unified_envs working directory

```bash
mkdir -p unified_envs
cd unified_envs
```

## Step 3: Sparse checkout BFCL (berkeley-function-call-leaderboard folder only)

Using modern sparse checkout method to download only the `berkeley-function-call-leaderboard` folder from the Gorilla repository:

```bash
# Clone with sparse filter (partial clone)
git clone --depth 1 --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git
cd gorilla

# Set sparse checkout to only include berkeley-function-call-leaderboard folder
git sparse-checkout set berkeley-function-call-leaderboard
```

This downloads only the `berkeley-function-call-leaderboard` folder (~15MB) instead of the full repository.

## Step 4: Merge with DiffuAgent BFCL code

DiffuAgent provides custom BFCL enhancements including 52 model configurations and custom handlers. Merge these enhancements:

```bash
# Go back to gorilla directory (current directory should be unified_envs/gorilla/)
cd /path/to/unified_envs/gorilla

# Simple one-command merge: copy all DiffuAgent BFCL code
cp -r ../../DiffuAgent/BFCL/* ./berkeley-function-call-leaderboard/
```

This will merge:
- Custom model handlers (DiffuAgent base, mixins, handlers)
- 52 model configurations (LLM + DLLM variants)
- Utility functions (LLM/DLLM request tools)
- Test scripts
- Configuration patches
- DEBUG logging utilities

## Step 5: Register DiffuAgent Models (Required)

**IMPORTANT**: After merging the code, you must register DiffuAgent models in BFCL's model configuration. This step adds the 52 DiffuAgent model configurations to BFCL's MODEL_CONFIG_MAPPING.

```bash
# Change to BFCL directory
cd /path/to/unified_envs/gorilla/berkeley-function-call-leaderboard

# Run the registration script
python3 register_diffuagent.py
```

**Expected Output**:
```
============================================================
Registering DiffuAgent Models
============================================================

✓ Found bfcl_eval/constants/model_config.py

[Step 1] Backing up original file...
✓ Backed up to: bfcl_eval/constants/model_config.py.backup.20250127_XXXXXX

[Step 2] Adding DiffuAgent registration...
  Found MODEL_CONFIG_MAPPING at line XXX
✓ Added Diffuagent registration

[Step 3] Verifying installation...
✓ Found 52 DiffuAgent models

Sample models:
  - diffuagent-chatbase/ministral-8b -> LLMHandler
  - diffuagent-chatbase/qwen3-8b -> LLMHandler
  - diffuagent-selector-chatbase/ministral-8b -> SelectorLLMHandler
  - diffuagent-editor-chatbase/qwen3-8b -> EditorLLMHandler
  - diffuagent-selector-editor-chatbase/ministral-8b -> SelectorEditorLLMHandler
  ... and 47 more

============================================================
✓ Installation Successful!
============================================================

You can now run:
  bfcl generate --model diffuagent-chatbase/qwen3-8b ...

To undo: mv bfcl_eval/constants/model_config.py.backup.XXXXXX bfcl_eval/constants/model_config.py
```

### What This Step Does

The registration script modifies `bfcl_eval/constants/model_config.py` to:

1. **Import DiffuAgent configuration builder**:
   ```python
   from bfcl_eval.build_handlers_diffuagent import add_diffuagent_model_configs
   ```

2. **Generate model configurations**:
   ```python
   diffuagent_model_map = add_diffuagent_model_configs()
   ```

3. **Merge into BFCL's model mapping**:
   ```python
   MODEL_CONFIG_MAPPING = {
       **diffuagent_model_map,  # ← Adds 52 DiffuAgent models
       **api_inference_model_map,
       **local_inference_model_map,
       **third_party_inference_model_map,
   }
   ```

### Verification

After registration, verify it worked:

```bash
python3 << 'EOF'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

# Check DiffuAgent models are registered
diffuagent_models = [k for k in MODEL_CONFIG_MAPPING.keys() if 'diffuagent' in k.lower()]
print(f"✓ Found {len(diffuagent_models)} DiffuAgent models")

# List some models
for model in sorted(diffuagent_models)[:5]:
    print(f"  - {model}")
EOF
```

**Should output**:
```
✓ Found 52 DiffuAgent models
  - diffuagent-chatbase/ministral-8b
  - diffuagent-chatbase/qwen3-8b
  - diffuagent-diff-chatbase/dream
  - diffuagent-diff-chatbase/fdllmv2
  - diffuagent-diff-chatbase/llada
```

### Manual Registration (Alternative)

If the script doesn't work, you can manually register:

1. Edit `bfcl_eval/constants/model_config.py`:
   ```bash
   vim bfcl_eval/constants/model_config.py
   ```

2. Add this **before** the `MODEL_CONFIG_MAPPING = {` line:
   ```python
   # DiffuAgent model configurations
   from bfcl_eval.build_handlers_diffuagent import add_diffuagent_model_configs
   diffuagent_model_map = add_diffuagent_model_configs()
   ```

3. Add `**diffuagent_model_map,` as the **first** item in MODEL_CONFIG_MAPPING:
   ```python
   MODEL_CONFIG_MAPPING = {
       **diffuagent_model_map,  # ← Add this line
       **api_inference_model_map,
       **local_inference_model_map,
       **third_party_inference_model_map,
   }
   ```

4. Save and verify with the verification command above.

## Running BFCL Evaluations

After setup, use the provided test script to run DiffuAgent evaluations:

### Quick Test

```bash
# From the DiffuAgent repository
cd /path/to/DiffuAgent/DiffuAgent/BFCL

# Run the test script (uses test_case_ids_to_generate.json)
bash tests/test_run_ids.sh
```

The test script will:
1. Generate responses for test cases specified in `test_case_ids_to_generate.json`
2. Save results to the BFCL results directory

### Customizing the Test

Edit `tests/test_run_ids.sh` to configure:

- **BFCL_PROJECT_ROOT**: Path to berkeley-function-call-leaderboard
- **MODEL_PATH**: Path to your model
- **MODEL_NAME**: Model identifier for BFCL
- **API keys and URLs**: Main agent and features backend endpoints

### Test Case Selection

Edit `test_case_ids_to_generate.json` to select which test cases to run:

```json
{
  "simple_java": ["1-5", "10-15"],
  "simple_javascript": ["1-3"],
  "simple_python": ["1-10"]
}
```

## BFCL Test Categories

BFCL includes the following test categories:

- **Simple**: Simple function calls with single function
- **Multiple**: Multiple function calls in one query
- **Parallel**: Parallel function execution
- **Multi-turn**: Multi-turn conversations with function calls
- **AST**: Abstract Syntax Tree validation
- **REST**: REST API calls
- **Web Search**: Agentic web search capabilities
- **Memory**: Agentic memory management

See `TEST_CATEGORIES.md` in the berkeley-function-call-leaderboard directory for detailed descriptions of each category.

## Adding Custom Models

To add a new model for evaluation:

1. Check `SUPPORTED_MODELS.md` for the list of supported models
2. Add your model configuration in `bfcl_eval/model_handler/`
3. Run evaluation with your custom model

## Output and Logging

Evaluation results are saved in:

```
/path/to/unified_envs/gorilla/berkeley-function-call-leaderboard/
└── results/
    ├── {model_name}_results.json      # Generated responses
    ├── {model_name}_evaluation.json   # Evaluation scores
    └── logs/
        └── {model_name}_eval.log      # Detailed logs
```

## Integration with DiffuAgent

This BFCL setup is part of the unified environment that includes:

- **AgentBoard**: Embodied AI evaluation (ALFWorld, ScienceWorld, BabyAI)
  - Location: `unified_envs/AgentBoard/agentboard/`
  - Installation: `DiffuAgent/Agentboard/installation.md`
  - Requires: PROJECT_PATH environment variable

- **BFCL**: Function calling evaluation
  - Location: `unified_envs/gorilla/berkeley-function-call-leaderboard/`
  - Installation: `DiffuAgent/BFCL/installation.md` (this file)
  - No environment variables required

Both projects use the same sparse checkout strategy for minimal footprint in the unified_envs directory.

## Additional Resources

- **BFCL Homepage**: https://gorilla.cs.berkeley.edu/leaderboard.html
- **Gorilla GitHub**: https://github.com/ShishirPatil/gorilla
- **Documentation**: See `README.md` in berkeley-function-call-leaderboard
- **Supported Models**: See `SUPPORTED_MODELS.md`
- **Test Categories**: See `TEST_CATEGORIES.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Logging Guide**: See `LOG_GUIDE.md`

## Next Steps

- See `README.md` in `DiffuAgent/BFCL/` for DiffuAgent-specific BFCL integration
- See `SUPPORTED_MODELS.md` for the list of supported models
- See `tests/` directory in `DiffuAgent/BFCL/` for custom test cases
