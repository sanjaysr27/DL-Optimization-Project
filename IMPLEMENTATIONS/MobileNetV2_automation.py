# automation code for mobilenetv2

import subprocess
import sys
import re
import json
import csv
from pathlib import Path
from datetime import datetime
import os                                    # <-- ADD

os.environ["PYTHONIOENCODING"] = "utf-8"

# ============================================================
# 1) EDIT HERE — path to python.exe inside your PS1_TF env
# ============================================================
PYTHON_EXE = r"C:\Users\PS1-2026\.conda\envs\PS1_TF\python.exe"

# ============================================================
# 2) EDIT HERE — filenames of your actual scripts
# ============================================================
BASE_SCRIPT           = "MobileNetV2_Leather_Base.py"
CHANNEL_PRUNE_SCRIPT  = "MobileNetV2_Leather_ChannelPruning.py"
DROPOUT_SCRIPT        = "MobileNetV2_Leather_Dropout.py"
LAYER_PRUNE_SCRIPT    = "MobileNetV2_Leather_LayerPruning.py"

QUANT_INT8_SCRIPT     = "MobileNetV2_Leather_INT8_PTQ.py"
QUANT_INT16_SCRIPT    = "MobileNetV2_Leather_INT16_PTQ.py"
QUANT_INT4_SCRIPT     = "MobileNetV2_Leather_INT4_PTQ.py"

# ============================================================
# 3) EDIT HERE — path where the FINAL fine-tuned base model gets saved.
#    This is the file that channel pruning / dropout / layer pruning all
#    load as their starting point. It MUST correspond to the variable
#    named `final_SAVED_BASE_MODEL_PATH` inside MobileNetV2_base.py.
#    The driver rewrites that variable automatically (see step 1 in main),
#    so you only need to set the path once, here.
# ============================================================
BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"

# ============================================================
# 4) EDIT HERE — base folder for saved models / tensorboard logs
#    (used to build unique paths per ratio automatically)
# ============================================================
MODELS_DIR      = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather"
TENSORBOARD_DIR = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs"

# ============================================================
# 5) Ratios
# ============================================================
CHANNEL_PRUNE_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #
DROPOUT_RATIOS       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #
LAYER_PRUNE_BLOCKS   = [1, 2, 3, 4, 5, 6, 7, 8] #

# ============================================================
# Where the metrics json gets written by each script
# (must match the filename used in the snippet you add to each script)
# ============================================================
METRICS_FILE = "last_run_metrics.json"

# ============================================================
# CSV results log
# ============================================================
RESULTS_CSV = "results_mobilenetv2.csv"


def init_results_csv():
    if not Path(RESULTS_CSV).exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "stage", "ratio_or_blocks", "test_loss", "test_accuracy"])


def log_result(stage, ratio_or_blocks):
    metrics_path = Path(METRICS_FILE)
    if not metrics_path.exists():
        print(f"  WARNING: {METRICS_FILE} not found, skipping metrics log for {stage} ({ratio_or_blocks})")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    test_loss = metrics.get("test_loss", "")
    test_acc = metrics.get("test_accuracy", "")

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), stage, ratio_or_blocks, test_loss, test_acc])

    print(f"  -> Logged: {stage} | {ratio_or_blocks} | loss={test_loss} | acc={test_acc}")

    # remove it so a failed next run doesn't accidentally reuse stale metrics
    metrics_path.unlink()


# ============================================================
# Helper: edit a variable's value inside a script file
# ============================================================
def set_variable_in_file(filepath, var_name, new_value):
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")

    pattern = rf"^(\s*{re.escape(var_name)}\s*=\s*).+$"

    if isinstance(new_value, str):
        replacement_value = f'r"{new_value}"'
    else:
        replacement_value = str(new_value)

    # IMPORTANT: use a function replacement, not a string replacement.
    # re.sub() treats backslashes in a *string* replacement as escape
    # sequences (\1, \g<1>, etc). Windows paths are full of backslashes
    # (e.g. "...\channel_pruned_0.2.keras"), and a backslash followed by
    # a letter like \c, \p, \t is not a valid template escape, so re.sub
    # would raise "re.error: bad escape \c" the moment a path like that
    # got substituted in. A callable replacement sidesteps this because
    # its return value is inserted literally, with no escape processing.
    new_text, count = re.subn(
        pattern,
        lambda m: m.group(1) + replacement_value,
        text,
        flags=re.MULTILINE,
    )

    if count == 0:
        raise ValueError(
            f"Could not find a line assigning '{var_name}' in {filepath}. "
            f"Check the variable name matches exactly."
        )
    if count > 1:
        print(f"WARNING: found {count} lines assigning '{var_name}' in {filepath}, all changed.")

    path.write_text(new_text, encoding="utf-8")
    print(f"  -> Set {var_name} = {replacement_value} in {filepath}")


# ============================================================
# Helper: run a script with live output streaming
# ============================================================
def run_script(script_path, label):
    print(f"\n{'='*60}")
    print(f"Running {label}: {script_path}")
    print(f"{'='*60}\n")

    process = subprocess.Popen(
        [PYTHON_EXE, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        line = line.rstrip("\n")
        if "/" in line and ("[" in line or "]" in line):
            sys.stdout.write(f"\r{line[:110]:<110}")
        else:
            sys.stdout.write(f"\n{line}")
        sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        print(f"\n{label} FAILED with exit code {process.returncode}. Continuing...\n")
        return False
    else:
        print(f"\n{label} completed successfully.\n")
        return True


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    init_results_csv()

    # Make sure output directories exist up front, otherwise model.save()
    # and TensorBoard(log_dir=...) will fail on a missing folder.
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(TENSORBOARD_DIR).mkdir(parents=True, exist_ok=True)
    Path(BASE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train base model ----
    # Force the base script to save its FINAL fine-tuned model to
    # BASE_MODEL_PATH, so every downstream stage loads the exact same file
    # this driver knows about. Also redirect the interim (frozen-backbone)
    # checkpoint next to it, so nothing tries to write to a path that
    # doesn't exist on this machine.
    interim_base_path = str(Path(BASE_MODEL_PATH).with_name(
        Path(BASE_MODEL_PATH).stem + "_frozen" + Path(BASE_MODEL_PATH).suffix
    ))
    set_variable_in_file(BASE_SCRIPT, "SAVED_BASE_MODEL_PATH", interim_base_path)
    set_variable_in_file(BASE_SCRIPT, "final_SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)

    ok = run_script(BASE_SCRIPT, "MobileNetV2 base training")
    if ok:
        log_result("base", "-")

    # ---- 2. Channel pruning ----
    for r in CHANNEL_PRUNE_RATIOS:
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "PRUNING_RATIO", r)
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "PRUNED_MODEL_PATH",
                              rf"{MODELS_DIR}\MobileNetV2_channel_pruned_{r}.keras")
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "TENSORBOARD_LOG_DIR_PRUNED",
                              rf"{TENSORBOARD_DIR}\MobileNetV2_channel_pruned_{r}")
        ok = run_script(CHANNEL_PRUNE_SCRIPT, f"MobileNetV2 channel pruning @ {r}")
        if ok:
            log_result("channel_pruning", r)

    # ---- 3. Dropout ----
    for r in DROPOUT_RATIOS:
        set_variable_in_file(DROPOUT_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(DROPOUT_SCRIPT, "CONV_DROPOUT_RATE", r)
        set_variable_in_file(DROPOUT_SCRIPT, "HEAD_DROPOUT_RATE", r)
        set_variable_in_file(DROPOUT_SCRIPT, "DROPOUT_MODEL_PATH",
                              rf"{MODELS_DIR}\MobileNetV2_dropout_{r}.keras")
        set_variable_in_file(DROPOUT_SCRIPT, "TENSORBOARD_LOG_DIR_DROPOUT",
                              rf"{TENSORBOARD_DIR}\MobileNetV2_dropout_{r}")
        ok = run_script(DROPOUT_SCRIPT, f"MobileNetV2 dropout @ {r}")
        if ok:
            log_result("dropout", r)

    # ---- 4. Layer pruning ----
    for blocks in LAYER_PRUNE_BLOCKS:
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "NUM_BLOCKS_TO_REMOVE", blocks)
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "LAYER_PRUNED_MODEL_PATH",
                              rf"{MODELS_DIR}\MobileNetV2_layer_pruned_{blocks}blocks.keras")
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "TENSORBOARD_LOG_DIR_LAYERPRUNED",
                              rf"{TENSORBOARD_DIR}\MobileNetV2_layer_pruned_{blocks}blocks")
        ok = run_script(LAYER_PRUNE_SCRIPT, f"MobileNetV2 layer pruning @ {blocks} blocks")
        if ok:
            log_result("layer_pruning", f"{blocks}_blocks")

    # ---- 5. Quantization (int8, int16-activation/int8-weight, int4-equivalent) ----
    # These run once each (no ratio sweep) and always quantize the same
    # FP32 base model, so only SAVED_BASE_MODEL_PATH and the output
    # path(s) need to be set before each run.
    set_variable_in_file(QUANT_INT8_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT8_SCRIPT, "QUANTIZED_MODEL_PATH",
                          rf"{MODELS_DIR}\MobileNetV2_quant_int8.tflite")
    ok = run_script(QUANT_INT8_SCRIPT, "MobileNetV2 quantization (int8)")
    if ok:
        log_result("quantization_int8", "-")

    set_variable_in_file(QUANT_INT16_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT16_SCRIPT, "QUANTIZED_MODEL_PATH",
                          rf"{MODELS_DIR}\MobileNetV2_quant_int16.tflite")
    ok = run_script(QUANT_INT16_SCRIPT, "MobileNetV2 quantization (int16-activation/int8-weight)")
    if ok:
        log_result("quantization_int16", "-")

    set_variable_in_file(QUANT_INT4_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT4_SCRIPT, "CLUSTERED_MODEL_PATH",
                          rf"{MODELS_DIR}\MobileNetV2_quant_int4_clustered.keras")
    set_variable_in_file(QUANT_INT4_SCRIPT, "TFLITE_MODEL_PATH",
                          rf"{MODELS_DIR}\MobileNetV2_quant_int4_clustered.tflite")
    ok = run_script(QUANT_INT4_SCRIPT, "MobileNetV2 quantization (int4-equivalent clustering)")
    if ok:
        log_result("quantization_int4", "-")

    print(f"\nAll MobileNetV2 stages finished. Results saved to {RESULTS_CSV}")
    
    
    
    


#ResNet

# ============================================================
# 2) EDIT HERE — filenames of your actual ResNet50 scripts
# ============================================================
BASE_SCRIPT           = "ResNet_50_applications.py"
CHANNEL_PRUNE_SCRIPT  = "ResNet50_Leather_ChannelPruning.py"
DROPOUT_SCRIPT        = "ResNet50_Leather_Dropout.py"
LAYER_PRUNE_SCRIPT    = "ResNet50_Leather_LayerPruning.py"

QUANT_INT8_SCRIPT     = "ResNet50_Leather_INT8_PTQ.py"
QUANT_INT16_SCRIPT    = "ResNet50_Leather_INT16_PTQ.py"
QUANT_INT4_SCRIPT     = "ResNet50_Leather_INT4_PTQ.py"

# ============================================================
# 3) EDIT HERE — path where the FINAL fine-tuned base model gets saved.
#    This is the file that channel pruning / dropout / layer pruning /
#    all three quantization scripts load as their starting point. It MUST
#    correspond to the variable named `FINAL_SAVED_MODEL_PATH` inside
#    ResNet50_Leather_Base.py (NOT `SAVED_BASE_MODEL_PATH`, which in that
#    script is the interim, frozen-backbone/head-only checkpoint saved
#    before the full fine-tune phase). The driver rewrites both variables
#    automatically (see step 1 in main), so you only need to set the path
#    once, here.base_applications_ResNet50_Full_Train
# ============================================================
BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50_Full_Train.keras"

# ============================================================
# 4) EDIT HERE — base folder for saved models / tensorboard logs
#    (used to build unique paths per ratio automatically)
# ============================================================
MODELS_DIR      = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather"
TENSORBOARD_DIR = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\logs"

# ============================================================
# 5) Ratios — kept identical to the MobileNetV2 driver, as requested
# ============================================================
CHANNEL_PRUNE_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]#
DROPOUT_RATIOS       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #
LAYER_PRUNE_BLOCKS   = [1, 2, 3, 4, 5, 6, 7, 8] #   # ResNet50 cap: <= 12 candidates

# ============================================================
# Where the metrics json gets written by each script
# (must match the filename used in the snippet you add to each script)
# ============================================================
METRICS_FILE = "last_run_metrics.json"

# ============================================================
# CSV results log
# ============================================================
RESULTS_CSV = "results_resnet50.csv"


def init_results_csv():
    if not Path(RESULTS_CSV).exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "stage", "ratio_or_blocks", "test_loss", "test_accuracy"])


def log_result(stage, ratio_or_blocks):
    metrics_path = Path(METRICS_FILE)
    if not metrics_path.exists():
        print(f"  WARNING: {METRICS_FILE} not found, skipping metrics log for {stage} ({ratio_or_blocks})")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    test_loss = metrics.get("test_loss", "")
    test_acc = metrics.get("test_accuracy", "")

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), stage, ratio_or_blocks, test_loss, test_acc])

    print(f"  -> Logged: {stage} | {ratio_or_blocks} | loss={test_loss} | acc={test_acc}")

    # remove it so a failed next run doesn't accidentally reuse stale metrics
    metrics_path.unlink()


# ============================================================
# Helper: edit a variable's value inside a script file
# ============================================================
def set_variable_in_file(filepath, var_name, new_value):
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")

    pattern = rf"^(\s*{re.escape(var_name)}\s*=\s*).+$"

    if isinstance(new_value, str):
        replacement_value = f'r"{new_value}"'
    else:
        replacement_value = str(new_value)

    # IMPORTANT: use a function replacement, not a string replacement.
    # re.sub() treats backslashes in a *string* replacement as escape
    # sequences (\1, \g<1>, etc). Windows paths are full of backslashes
    # (e.g. "...\channel_pruned_0.2.keras"), and a backslash followed by
    # a letter like \c, \p, \t is not a valid template escape, so re.sub
    # would raise "re.error: bad escape \c" the moment a path like that
    # got substituted in. A callable replacement sidesteps this because
    # its return value is inserted literally, with no escape processing.
    new_text, count = re.subn(
        pattern,
        lambda m: m.group(1) + replacement_value,
        text,
        flags=re.MULTILINE,
    )

    if count == 0:
        raise ValueError(
            f"Could not find a line assigning '{var_name}' in {filepath}. "
            f"Check the variable name matches exactly."
        )
    if count > 1:
        print(f"WARNING: found {count} lines assigning '{var_name}' in {filepath}, all changed.")

    path.write_text(new_text, encoding="utf-8")
    print(f"  -> Set {var_name} = {replacement_value} in {filepath}")


# ============================================================
# Helper: run a script with live output streaming
# ============================================================
def run_script(script_path, label):
    print(f"\n{'='*60}")
    print(f"Running {label}: {script_path}")
    print(f"{'='*60}\n")

    process = subprocess.Popen(
        [PYTHON_EXE, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        line = line.rstrip("\n")
        if "/" in line and ("[" in line or "]" in line):
            sys.stdout.write(f"\r{line[:110]:<110}")
        else:
            sys.stdout.write(f"\n{line}")
        sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        print(f"\n{label} FAILED with exit code {process.returncode}. Continuing...\n")
        return False
    else:
        print(f"\n{label} completed successfully.\n")
        return True


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    init_results_csv()

    # Make sure output directories exist up front, otherwise model.save()
    # and TensorBoard(log_dir=...) will fail on a missing folder.
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(TENSORBOARD_DIR).mkdir(parents=True, exist_ok=True)
    Path(BASE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train base model ----
    # Force the base script to save its interim (frozen-backbone,
    # head-only) checkpoint next to BASE_MODEL_PATH, and its FINAL
    # fine-tuned model to BASE_MODEL_PATH itself, so every downstream
    # stage loads the exact same file this driver knows about.
    #
    # NOTE the variable name difference from the MobileNetV2 driver:
    # ResNet50_Leather_Base.py names its final-save variable
    # `FINAL_SAVED_MODEL_PATH`, not `final_SAVED_BASE_MODEL_PATH`.
    interim_base_path = str(Path(BASE_MODEL_PATH).with_name(
        Path(BASE_MODEL_PATH).stem.replace("_Full_Train", "") + Path(BASE_MODEL_PATH).suffix
    ))
    set_variable_in_file(BASE_SCRIPT, "SAVED_BASE_MODEL_PATH", interim_base_path)
    set_variable_in_file(BASE_SCRIPT, "FINAL_SAVED_MODEL_PATH", BASE_MODEL_PATH)

    ok = run_script(BASE_SCRIPT, "ResNet50 base training")
    if ok:
        log_result("base", "-")

    # ---- 2. Channel pruning ----
    for r in CHANNEL_PRUNE_RATIOS:
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "PRUNING_RATIO", r)
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "PRUNED_MODEL_PATH",
                              rf"{MODELS_DIR}\resnet50_channel_pruned_{r}.keras")
        set_variable_in_file(CHANNEL_PRUNE_SCRIPT, "TENSORBOARD_LOG_DIR_PRUNED",
                              rf"{TENSORBOARD_DIR}\resnet50_channel_pruned_{r}")
        ok = run_script(CHANNEL_PRUNE_SCRIPT, f"ResNet50 channel pruning @ {r}")
        if ok:
            log_result("channel_pruning", r)

    # ---- 3. Dropout ----
    for r in DROPOUT_RATIOS:
        set_variable_in_file(DROPOUT_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(DROPOUT_SCRIPT, "CONV_DROPOUT_RATE", r)
        set_variable_in_file(DROPOUT_SCRIPT, "HEAD_DROPOUT_RATE", r)
        set_variable_in_file(DROPOUT_SCRIPT, "DROPOUT_MODEL_PATH",
                              rf"{MODELS_DIR}\resnet50_dropout_{r}.keras")
        set_variable_in_file(DROPOUT_SCRIPT, "TENSORBOARD_LOG_DIR_DROPOUT",
                              rf"{TENSORBOARD_DIR}\resnet50_dropout_{r}")
        ok = run_script(DROPOUT_SCRIPT, f"ResNet50 dropout @ {r}")
        if ok:
            log_result("dropout", r)

    # ---- 4. Layer pruning ----
    for blocks in LAYER_PRUNE_BLOCKS:
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "NUM_BLOCKS_TO_REMOVE", blocks)
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "LAYER_PRUNED_MODEL_PATH",
                              rf"{MODELS_DIR}\resnet50_layer_pruned_{blocks}blocks.keras")
        set_variable_in_file(LAYER_PRUNE_SCRIPT, "TENSORBOARD_LOG_DIR_LAYERPRUNED",
                              rf"{TENSORBOARD_DIR}\resnet50_layer_pruned_{blocks}blocks")
        ok = run_script(LAYER_PRUNE_SCRIPT, f"ResNet50 layer pruning @ {blocks} blocks")
        if ok:
            log_result("layer_pruning", f"{blocks}_blocks")

    # ---- 5. Quantization (int8, int16-activation/int8-weight, int4-equivalent) ----
    # These run once each (no ratio sweep) and always quantize the same
    # FP32 base model, so only SAVED_BASE_MODEL_PATH and the output
    # path(s) need to be set before each run.
    set_variable_in_file(QUANT_INT8_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT8_SCRIPT, "QUANTIZED_MODEL_PATH",
                          rf"{MODELS_DIR}\resnet50_quant_int8.tflite")
    ok = run_script(QUANT_INT8_SCRIPT, "ResNet50 quantization (int8)")
    if ok:
        log_result("quantization_int8", "-")

    set_variable_in_file(QUANT_INT16_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT16_SCRIPT, "QUANTIZED_MODEL_PATH",
                          rf"{MODELS_DIR}\resnet50_quant_int16.tflite")
    ok = run_script(QUANT_INT16_SCRIPT, "ResNet50 quantization (int16-activation/int8-weight)")
    if ok:
        log_result("quantization_int16", "-")

    set_variable_in_file(QUANT_INT4_SCRIPT, "SAVED_BASE_MODEL_PATH", BASE_MODEL_PATH)
    set_variable_in_file(QUANT_INT4_SCRIPT, "CLUSTERED_MODEL_PATH",
                          rf"{MODELS_DIR}\resnet50_quant_int4_clustered.keras")
    set_variable_in_file(QUANT_INT4_SCRIPT, "TFLITE_MODEL_PATH",
                          rf"{MODELS_DIR}\resnet50_quant_int4_clustered.tflite")
    ok = run_script(QUANT_INT4_SCRIPT, "ResNet50 quantization (int4-equivalent clustering)")
    if ok:
        log_result("quantization_int4", "-")

    print(f"\nAll ResNet50 stages finished. Results saved to {RESULTS_CSV}")