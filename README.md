# BreakingAI
HPI-GCN Project Structure Analysis

This document provides a structured, indented overview of the HPI-GCN project layout. Every Python file (except the `discard` folder) is listed and the internal dataset folder structure is described.

---

1. Top-Level Entry Points & Scripts
   - These files live in the project root and serve as the main interfaces for training, inference, and visualization.
     - `main.py`  
       : Central entry point for the training framework (likely uses `argparse` to dispatch tasks).
     - `app.py`  
       : Gradio-based web app for interactive demos and visualization.
     - `audit_generator.py`  
       : Script to audit or validate the generator's outputs.
     - `audition_checkpoints.py`  
       : Utility to quick-check multiple model checkpoints (e.g., generate samples from each).
     - `check_progress_anchor.py`  
       : Tool to monitor training progress, possibly against a reference "anchor" set.
     - `debug_visualize.py`  
       : Lightweight script for visualization debugging without the full app.
     - `demo_inference_discriminator.py`  
       : Run inference specifically through the discriminator (classification / quality scoring).
     - `evaluate_brace_final.py`  
       : Final evaluation script for the BRACE dataset model.
     - `audio_brace_loader.py`  
       : Specialized dataloader / preprocessor for audio in the BRACE dataset.

2. Core Model Architecture (`model/`)
   - Contains network definitions (Generator, Discriminator, GCN building blocks).
     - `model/generator.py`  
       : Generator architecture (likely RNN/GRU-based).
     - `model/HPI_GCN_OP.py`  
       : Object-Pose GCN — a GCN for human pose, focusing on object interaction or absolute pose.
     - `model/HPI_GCN_RP.py`  
       : Relative-Pose GCN — focuses on relative joint positions/motion.
     - `model/REP_block/`  
       : Re-parameterization blocks for efficient GCN/TCN architectures.
         - `model/REP_block/REP_GCN.py`  
           : Reparameterizable Graph Convolution implementation.
         - `model/REP_block/REP_TCN.py`  
           : Reparameterizable Temporal Convolution implementation.
         - `model/REP_block/transforms.py`  
           : Tensor transformations for these blocks.

3. Data Feeders (`feeders/`)
   - Data loading, batching, and preprocessing for datasets.
     - `feeders/feeder_ntu.py`  
       : Data loader for NTU RGB+D dataset.
     - `feeders/feeder_ucla.py`  
       : Data loader for NW-UCLA dataset.
     - `feeders/bone_pairs.py`  
       : Definitions of bone connections (skeletal topology) used in preprocessing.
     - `feeders/tools.py`  
       : General data manipulation utilities (augmentation, normalization, etc.).

4. Graph Topologies (`graph/`)
   - Adjacency matrices and skeletal graphs for the GCNs.
     - `graph/coco_17.py`  
       : COCO 17-keypoint skeletal graph definition.
     - `graph/ntu_rgb_d.py`  
       : NTU RGB+D skeletal graph (25 joints) definition.
     - `graph/ucla.py`  
       : NW-UCLA skeletal graph definition.
     - `graph/tools.py`  
       : Graph utility functions (e.g., adjacency normalization).

5. The "SASAKI" Brain (`SASAKI/`)
   - High-level decision logic for the neuro-symbolic / improvisation system.
     - `SASAKI/ImprovisationDecisionEngine.py`  
       : Python logic engine for improvisation decisions.
     - `SASAKI/Improvisation_SLM.py`  
       : Integrates a Small Language Model or symbolic logic for improvisation.
     - YAML configs (control decision trees / behavior flows):
       - `ImprovisationDecision.yml`
       - `ImprovisationFlow.yml`
       - `coherence.yml`
       - `genesys.yml`
       - `motion.yml`
       - `stillness.yml`

6. Dataset Structures
   - data/brace/ (Processed Training Data)
     - `data/brace/`  
       : Processed, serialized NumPy files used as training input.
         - `BRACE_centered_aug.npz`  
           : Centered and augmented pose sequences — main training data.
         - `BRACE_fixed_topology_v3.npz`  
           : Version with standardized/corrected skeletal topology (v3).
   - brace/ (Source Raw Dataset)
     - `brace/`  
       : Raw multi-modal data before compilation into `.npz`.
         - `brace/dataset/`  
           : Individual sequence files.
         - `brace/audio_features/`  
           : Pre-extracted audio features organized by Year/VideoID (e.g., `2011/3rIk56dcBTM/`).
         - `brace/manual_keypoints/`  
           : Manually annotated / corrected keypoints.
         - `brace/annotations/`  
           : Label annotations (genre, move type, etc.).
         - `brace/videos_info.csv`  
           : Metadata linking video IDs to descriptions / labels.
   - data/ (Benchmark Datasets)
     - `data/nturgbd_raw/` : Raw skeleton files from NTU RGB+D.
     - `data/ntu120/`      : Processed NTU RGB+D 120 dataset.
     - `data/NW-UCLA/`     : NW-UCLA dataset.
     - `data/ntu/`         : NTU RGB+D (60 classes).

7. Model Checkpoints (`pretrained_models/`)
   - Trained weights and checkpoint history.
     - `pretrained_models/brace_final_model.pt`  
       : Final, fully trained model for BRACE.
     - `pretrained_models/brace_finetuned.pt`  
       : Fine-tuned variant.
     - `pretrained_models/ntu120_pretrained.pt`  
       : Base model pretrained on NTU-120 for transfer learning.
     - `pretrained_models/generator_checkpoints/`  
       : History of generator weights by epoch.
     - Example snapshots:
       - `generator_clean_ep5.pt`
       - `generator_clean_ep10.pt`
       - `generator_clean_ep15.pt`

8. Web / Mobile Application (`my-app/`)
   - User interface code and mobile starter.
     - `my-app/backend.py`  
       : Python backend logic.
     - `my-app/frontend.py`  
       : Python frontend (Streamlit / similar).
     - `my-app/my-app.py`  
       : Main app entry point.
     - `my-app/starter-for-react-native/`  
       : React Native starter code for mobile app.

9. Utilities & Libraries
   - `torchlight/`  
     : Custom / vendored PyTorch utilities.
       - `torchlight/torchlight/gpu.py` : GPU management utilities.
       - `torchlight/torchlight/util.py`: Generic I/O and logging utilities.
   - `config/`  
     : Configuration sets for training runs.
       - `config/nturgbd120-cross-subject/`
         - `HPI_GCN_OP.yaml` : Object-Pose model config.
         - `HPI_GCN_RP.yaml` : Relative-Pose model config.

10. Presentation & Output
    - `ppt/render_architecture.py`  
      : Script to generate architecture diagrams for papers / presentations.
    - `animation/`  
      : Generated GIFs and videos used for demos and validation (e.g., `Final_Footwork.gif`, `Scientific_Proof.gif`).

---
Notes
- This README excludes any "discard" folder contents (as originally requested).
- Filenames and folder names are shown with trailing slashes where appropriate to indicate directories.
- If you want this converted into a table-of-contents with direct links to files in the repository (or reorganized into a machine-readable manifest), tell me which format you prefer and I will produce it.
