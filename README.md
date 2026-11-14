# Hesitation-aware robot handoff

Code and data behind **"A Simulation Study of Hesitation-Aware Robotic Handoff: Safety-Efficiency Tradeoffs Across Collaborative Environment Classes"**, accepted for oral presentation at the 16th IEOM International Conference on Industrial Engineering and Operations Management, Ankara, June 2026 (Paper ID 222).

When a robot hands a part to a person, the person sometimes pauses, reaches back or retries. This project asks whether a robot that recognises that hesitation and slows down gets fewer near-collisions than one moving at a fixed speed, and what that costs in task time.

- **Simulation:** a 2-D handoff model in MATLAB/Octave. `simulations/run_paper_benchmark.m` runs paired Monte Carlo episodes across 5 factory-floor environments, from a low-conflict open cell to precision insertion, with 50 seeds by default (`NUM_SEEDS`). For each seed and environment, both policies start from the same random seed.
- **Policy:** a hesitation-state classifier (Python package in `src/hesitation/`) drives an adaptive speed policy, compared with a fixed-speed baseline.
- **Evaluation:** ISO/TS 15066 safety limits, Cohen's d and Wilcoxon signed-rank tests, and a 27-combination sensitivity sweep.
- **Data (`data/`):** ISO/TS 15066 limits, 9 reference HRC dataset entries used for scenario motivation (not for validation), and 38 paper records from arXiv keyword queries, some of them off-topic.

**Status:** research code as of April 2026. `RESEARCH_PAPER.md` is an early working draft: it describes a 500-seed, 8-environment setup that the current code does not use, and its results tables are unfilled. The accepted paper above is the reference.

---

## 📁 **Project Structure**

```
IEOM/
├── 📂 simulations/           # MATLAB/Octave simulation files
│   ├── baseline_handoff_simulation.m          # Main simulation (MATLAB)
│   ├── baseline_handoff_simulation_octave.m   # Octave-compatible version
│   ├── baseline_handoff_simulation_integrated.m # Enhanced integrated version
│   ├── matlab_verify_three_scenarios.m        # Verification script
│   ├── matlab_cli_debug_smoke.m             # Debug harness
│   ├── matlab_ab_policy_smoke.m              # A/B policy smoke test
│   ├── matlab_ab_expected_check.m           # A/B expected results check
│   ├── matlab_baseline_expected_check.m     # Baseline expected results
│   ├── matlab_python_bridge_smoke.m         # Python bridge integration test
│   ├── matlab_run_ab_policy_benchmark.m      # Run A/B policy benchmark
│   ├── test_matlab_cli_integration.m         # CLI integration test
│   ├── test_matlab_integration.m             # Full integration test
│   └── test_octave_struct.m                  # Octave structure test
│
├── 📂 data/                 # Dataset and reference files
│   ├── hrc_datasets.csv                     # 9 HRC-relevant datasets (records 1-9); records 10-20 are unrelated Zenodo scrapes excluded from analysis
│   ├── hrc_papers.csv                      # 38 paper records from arXiv keyword queries (includes off-topic hits)
│   └── iso_safety_limits.csv                # ISO/TS 15066 safety standards
│
├── 📂 docs/                 # Documentation and guides
│   ├── README_DATASETS.md                  # Dataset integration guide
│   ├── QUICK_START.txt                     # Quick start guide
│   ├── DATASET_INTEGRATION_GUIDE.md        # Technical details
│   ├── INTEGRATION_SUMMARY.txt             # Feature overview
│   ├── HRC_SIMULATION_INTEGRATION_SUMMARY.md # Complete summary
│   └── [other documentation files]
│
├── 📂 outputs/figures/       # Generated visualizations
│   ├── figure1_trajectories.png             # Position trajectories
│   ├── figure2_separation.png              # Separation distance
│   ├── figure3_metrics.png                 # Performance metrics
│   ├── figure4_phase_portrait.png          # Phase portrait
│   ├── figure5_iso_limits.png             # ISO safety limits
│   ├── figure6_datasets.png               # Dataset overview
│   └── figure7_dataset_validation_octave.png # Validation analysis
│
├── 📂 src/                  # Restored Python hesitation package + MATLAB bridge
│   ├── hesitation/                        # Hesitation simulation/inference package
│   └── matlab/                            # MATLAB bridge helpers for Python integration
│
├── 📂 scripts/              # Python and utility scripts
│   ├── generate_synthetic_dataset.py      # Synthetic data generation
│   ├── phase2_cli.py                      # Training/inference/deep workflow CLI
│   ├── run_baseline.py                    # Rules baseline runner
│   ├── hrc_data_scraper.py                # Dataset scraper
│   ├── hrc_data_scraper_v2.py             # Enhanced scraper
│   ├── phase3_verify_outputs.py           # Output verification
│   └── temp.py                            # Temporary utilities
│
├── 📂 archive/              # Archived files and databases
│   ├── ieom_model/                        # Model files
│   ├── ieom_plan_tracking.db              # Tracking database
│   └── [other archived items]
│
├── 📂 updates/              # Project updates and guides
│   └── [update documentation files]
│
├── 📄 README.md              # Main project documentation
├── 📄 PROGRAM_OVERVIEW.md   # Detailed program overview
└── 📄 RESEARCH_PAPER.md     # Early paper draft (results tables unfilled)
```

---

## 🚀 **Quick Start**

### **Run Enhanced Simulation**

#### **MATLAB Environment**
```matlab
cd simulations
run('baseline_handoff_simulation.m')
```

#### **Octave Environment**
```bash
cd simulations
octave baseline_handoff_simulation_octave.m
```

#### **Terminal Execution**
```bash
cd simulations
matlab -batch "baseline_handoff_simulation"
```

### **Run Restored Hesitation Sim**

#### **Python Package Setup**
```bash
pip install -e ".[dev,serve]"
python -c "import hesitation"
```

#### **Synthetic Data + CLI Flow**
```bash
python scripts/generate_synthetic_dataset.py --output /tmp/ieom_synth.jsonl --n-sessions 8 --seed 17
python scripts/phase2_cli.py train-classical --input /tmp/ieom_synth.jsonl --output-dir /tmp/ieom_phase2
```

### **What You Get**
- ✅ **7 visualization figures** including dataset validation analysis
- ✅ **Comprehensive console output** with safety compliance checking
- ✅ **ISO/TS 15066 scenario motivation** from 9 HRC-relevant datasets
- ✅ **Performance metrics** and safety distribution analysis
- ✅ **Restored hesitation package** for Python inference, evaluation, and MATLAB bridge workflows

---

## 📊 **Key Features**

### **Dataset Integration**
- **9 HRC-relevant datasets** used for scenario motivation and ISO parameter grounding
- **38 paper records** from arXiv keyword queries; some are off-topic hits (for example ISO astronomy papers matched by an ISO 15066 query)
- **ISO safety standards** integration
- **Note**: `hrc_datasets.csv` contains additional non-HRC Zenodo records (records 10–20) from a broad scraper run; these are excluded from all analysis

### **Simulation Capabilities**
- **Three speed scenarios**: Slow, Moderate, Aggressive
- **Real-time safety assessment** with threshold monitoring
- **ISO compliance checking** per body region
- **Dataset validation** framework

### **Advanced Analysis**
- **Speed vs Safety** scatter plots
- **Robot platform** distribution analysis
- **Performance metrics** dashboard
- **Safety distribution** statistics

---

## 📈 **Validation Results**

### **Dataset Coverage**
```
HRC-Relevant Datasets: 9
├── Used for: scenario motivation, ISO parameter grounding
├── NOT used for: trajectory comparison, empirical validation
└── Full list: data/hrc_datasets.csv (rows 1–9)
```

### **Baseline Safety Analysis (1-D kinematic model, fixed-speed policy)**
```
ISO Compliance (fixed-speed baseline): 30% of scenarios
Unsafe scenarios (fixed-speed baseline): 70%
This result motivates adaptive speed control — it is NOT a Policy B result.
See artifacts/paper_results/ for A/B benchmark statistics.
```

### **A/B Benchmark (run run_paper_benchmark.m for current numbers)**
```
Primary metric : overlap_risk_event_count (Policy A vs. Policy B)
Secondary metric: task_completion_time_sec
N seeds: 50 (NUM_SEEDS) | Environments: 5 | Statistical test: Wilcoxon signed-rank
See artifacts/paper_results/tables/main_ab_benchmark_summary.csv
```

---

## 📋 **Requirements**

### **Software Requirements**
- **MATLAB R2018b+** for full feature support
- **GNU Octave 11.1+** for open-source compatibility
- **Python 3.10+** for the restored hesitation package and CLI workflows

### **Dependencies**
- **CSV data files** in `data/` directory
- **Graphics display** capability
- **readtable()** function support
- **Optional Python extras**: `.[dev]`, `.[serve]`, and `.[deep]`

---

## 🔧 **Usage Instructions**

### **1. Dataset Validation**
```matlab
% Simulation checks each dataset row marked reachable in hrc_datasets.csv
% Results shown in console output and Figure 7
```

### **2. Custom Analysis**
```matlab
% Modify simulation parameters in Lines 57-77
% Adjust safety thresholds as needed
% Add new datasets to hrc_datasets.csv
```

### **3. Hesitation CLI**
```bash
python scripts/phase2_cli.py train-deep --input /tmp/ieom_synth.jsonl --output-dir /tmp/ieom_deep
python scripts/phase2_cli.py evaluate-deep --input /tmp/ieom_synth.jsonl --model-path /tmp/ieom_deep/deep_model.pt
```

### **4. Output Management**
```matlab
% Figures saved to outputs/figures/
% Console reports generated automatically
% Data exported in CSV format
```

### **5. Generating Paper-Ready Benchmark Results**
```matlab
% To run the canonical multi-seed A/B benchmark comparing the baseline against the trained classical hesitation model:
cd simulations
run('run_paper_benchmark.m')
% Aggregate tables (CSV) and figures (PNG) will be saved to artifacts/paper_results/
```

---

## 📚 **Documentation**

### **Essential Reading**
- **`docs/QUICK_START.txt`** - Quick start guide
- **`docs/README_DATASETS.md`** - Dataset integration details
- **`docs/HRC_SIMULATION_INTEGRATION_SUMMARY.md`** - Complete technical summary
- **`docs/PHASE_1_SUMMARY.md`** - Hesitation package and inference interface summary

### **Technical References**
- **`docs/DATASET_INTEGRATION_GUIDE.md`** - Implementation details
- **`docs/INTEGRATION_SUMMARY.txt`** - Feature overview
- **`docs/MATLAB_TEAM_HANDOFF.md`** - Handoff notes for the MATLAB team
- **`docs/[other files]**** - Specialized documentation

---

## 🎯 **Research Applications**

### **Academic Research**
- **Simulation validation** against real-world data
- **Safety standards** compliance checking
- **Performance benchmarking** across datasets
- **Comparative analysis** of robot platforms

### **Industrial Applications**
- **Risk assessment** for collaborative robots
- **Safety parameter** optimization
- **Controller design** with data-driven insights
- **Regulatory compliance** verification

---

## 📞 **Support & Maintenance**

### **File Locations**
```
Project Root: repository root
Main Simulation: simulations/baseline_handoff_simulation.m
Enhanced Simulation: simulations/baseline_handoff_simulation_integrated.m
Data Files: data/ (hrc_datasets.csv, hrc_papers.csv, iso_safety_limits.csv)
Documentation: docs/ (all .md and .txt files)
Outputs: outputs/figures/ (all .png files)
Scripts: scripts/ (all .py files)
Test Files: simulations/ (test_*.m files)
Archive: archive/ (historical files and databases)
```

### **Getting Help**
1. **Check documentation** in `docs/` folder
2. **Review QUICK_START.txt** for basic usage
3. **Run verification scripts** in `simulations/`
4. **Check data integrity** in `data/` folder

---

## ✅ **Project Status**

**Simulation Engine**: `run_paper_benchmark.m` runs 50 seeds × 5 environments by default  
**A/B Benchmark**: ✅ `run_paper_benchmark.m` produces stable CSV + PNG artifacts  
**Sensitivity Analysis**: ✅ `run_sensitivity_analysis.m` sweeps 27 parameter combos  
**Research Paper**: 🔄 Draft in `RESEARCH_PAPER.md`  
**Scope Caveat**: 2-D kinematic model; hesitation classifier trained on synthetic data only — see paper Limitations section  
