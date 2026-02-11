"""Estimate cycles per layer for kwsmlp_w3a3.onnx.

Uses FINN's built-in estimate_only_dataflow_steps pipeline.
Results are written to output/report/.
"""

import json
import os

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    estimate_only_dataflow_steps,
)

MODEL_PATH = "example_nets/unsw_nb15-mlp-w2a2.onnx"
OUTPUT_DIR = "output"

cfg = DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    synth_clk_period_ns=5.0,
    fpga_part="xc7z020clg400-1",
    target_fps=1000,
    generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
    steps=estimate_only_dataflow_steps,
    save_intermediate_models=True,
    enable_build_pdb_debug=False,
)

ret = build_dataflow_cfg(MODEL_PATH, cfg)

if ret == 0:
    report_dir = os.path.join(OUTPUT_DIR, "report")
    for report_file in sorted(os.listdir(report_dir)):
        if report_file.endswith(".json"):
            print(f"\n=== {report_file} ===")
            with open(os.path.join(report_dir, report_file)) as f:
                print(json.dumps(json.load(f), indent=2))
