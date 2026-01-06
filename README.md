# BreakingAI
HPI-GCN Project Structure Analysis
This document provides a deep analysis of the HPI-GCN project structure, detailing every Python file (excluding the discard folder) and defining the internal structure of the dataset folders.

1. Top-Level Entry Points & Scripts
These files reside in the project root and serve as the primary interfaces for training, inference, and visualization.

main.py
: The central entry point for the training framework (likely utilizing argparse to dispatch to specific processors).
app.py
: A Gradio-based web application for interactive demonstration and visualization of the model.
audit_generator.py
: Script for auditing or validating the generator's outputs.
audition_checkpoints.py
: Utility to quick-check multiple model checkpoints (e.g., generating samples from each to see progress).
check_progress_anchor.py
: A tool to monitor training progress, possibly against a reference "anchor" set.
debug_visualize.py
: A lightweight script for debugging visualization logic without the full app overhead.
demo_inference_discriminator.py
: Specialized script for running inference specifically through the discriminator (e.g., for classification or quality scoring).
evaluate_brace_final.py
: Final evaluation script for the BRACE dataset model.
audio_brace_loader.py
: A specialized dataloader or preprocessor for the audio component of the BRACE dataset.
2. Core Model Architecture (model/)
Contains the definitions of the neural networks (Generator, Discriminator, and building blocks).

model/generator.py
: The generative model architecture (likely RNN/GRU based).
model/HPI_GCN_OP.py
: Object-Pose GCN. A Graph Convolutional Network designed to handle human pose data, likely focusing on object interaction or absolute pose.
model/HPI_GCN_RP.py
: Relative-Pose GCN. A variation of the GCN focusing on relative joint positions or motions.
model/REP_block/: Contains re-parameterization blocks for optimizing the GCNs.
model/REP_block/REP_GCN.py: Reparameterizable Graph Convolution implementation.
model/REP_block/REP_TCN.py: Reparameterizable Temporal Convolution implementation.
model/REP_block/transforms.py: Tensor transformations specific to these blocks.
3. Data Feeders (feeders/)
Handles data loading, batching, and preprocessing for different datasets.

feeders/feeder_ntu.py: Data loader for the NTU RGB+D dataset.
feeders/feeder_ucla.py: Data loader for the NW-UCLA dataset.
feeders/bone_pairs.py: Definitions of bone connections (skeletal topology) used for preprocessing.
feeders/tools.py: General data manipulation tools (augmentation, normalization).
4. Graph Topologies (graph/)
Defines the adjacency matrices and skeletal graphs used by the GCNs.

graph/coco_17.py: The COCO 17-keypoint skeletal graph definition.
graph/ntu_rgb_d.py: The NTU RGB+D skeletal graph definition (25 joints).
graph/ucla.py: The NW-UCLA skeletal graph definition.
graph/tools.py: Graph utility functions (e.g., normalizing adjacency matrices).
5. The "SASAKI" Brain (SASAKI/)
This folder appears to contain the high-level decision logic for the Neuro-Symbolic aspect of the system (likely for the "Improvisation" or logic-driven generation).

SASAKI/ImprovisationDecisionEngine.py: The Python logic engine driving improvisation decisions.
SASAKI/Improvisation_SLM.py: Likely "Small Language Model" or "Symbolic Logic Model" integration for improvisation.
YAML Configs:
ImprovisationDecision.yml, ImprovisationFlow.yml: Configuration for decision trees/flow.
coherence.yml, genesys.yml, motion.yml, stillness.yml: Behavior profiles or state definitions.
6. Dataset Structures
data/brace/ (Processed Training Data)
This folder serves as the Training Input Source. It contains pre-processed, serialized numpy files ready for the data feeders.

BRACE_centered_aug.npz: The main training file, containing centered and augmented pose sequences.
BRACE_fixed_topology_v3.npz: A version with corrected or standardized skeletal topology (Version 3).
brace/ (Source Raw Dataset)
This folder contains the Raw Multi-Modal Data before it is compiled into .npz files.

brace/dataset/: The individual sequence files.
brace/audio_features/: Pre-extracted audio features organized by Year/VideoID (e.g., 2011/3rIk56dcBTM/).
brace/manual_keypoints/: Manually annotated or corrected keypoint data.
brace/annotations/: Label annotations (genre, move type, etc.).
brace/videos_info.csv: Metadata CSV linking video IDs to descriptions/labels.
data/ (Benchmark Datasets)
Standard academic datasets for action recognition/generation.

data/nturgbd_raw/: The raw skeleton files from NTU RGB+D.
data/ntu120/: Processed NTU RGB+D 120 dataset.
data/NW-UCLA/: The NW-UCLA dataset.
data/ntu/: Standard NTU RGB+D (60 classes) data.
7. Model Checkpoints (pretrained_models/)
Stores trained weights and checkpoint history.

brace_final_model.pt: The definitive, fully trained model for the BRACE dataset.
brace_finetuned.pt: A version of the model fine-tuned on specific data.
ntu120_pretrained.pt: Base model pretrained on NTU-120 (likely used for transfer learning).
generator_checkpoints/: Directory containing history of generator weights during training.
generator_clean_ep{5,10,15}.pt: Snapshots of the generator at specific epochs.
8. Web/Mobile Application (my-app/)
Contains source code for user interfaces.

my-app/backend.py: Python backend logic for the app.
my-app/frontend.py: Python-based frontend (possibly Streamlit or similar).
my-app/my-app.py: Main app entry point.
my-app/starter-for-react-native/: Source code for a React Native mobile application.
9. Utilities & Libraries
torchlight/
A custom utility library (possibly a submodule or vendored library) for PyTorch training loops.

torchlight/torchlight/gpu.py: GPU management utilities.
torchlight/torchlight/util.py: General I/O and logging utilities.
config/
Configuration files for training runs.

config/nturgbd120-cross-subject/:
HPI_GCN_OP.yaml: Config for Object-Pose model.
HPI_GCN_RP.yaml: Config for Relative-Pose model.
10. Presentation & Output
ppt/render_architecture.py: Script to generate professional architecture diagrams for the paper/presentation.
animation/: Stores generated GIFs and videos used for demos and validation (e.g., Final_Footwork.gif, Scientific_Proof.gif).
