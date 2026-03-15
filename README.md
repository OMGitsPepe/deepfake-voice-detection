# CRNN with Neural Denoising Front-End

## Architecture
The pipeline consists of three sequential modules designed to map noisy acoustic environments to temporal deepfake probabilities:

* **Module A - Neural Denoising (U-Net):** A lightweight U-Net pre-trained on noisy speech and frozen 
suppresses codec distortion. It is designed to allow the downstream classifier to focus strictly on 
vocal artifacts rather than environmental mismatches.
* **Module B - Spatial Feature Extractor (CNN):** A truncated ResNet extracts frame-level spatial 
features. Instead of pooling spatial dimensions into a single vector, it preserves the time dimension to 
extract a sequence of frame-level features.
* **Module C - Temporal Sequence Modeling (BiLSTM):** A BiLSTM models long-range temporal dependencies 
to capture phonetic pacing anomalies. The final forward and backward hidden states are concatenated and 
passed through a fully connected layer with a Sigmoid activation.

## Training
To prevent the denoiser from inadvertently washing out synthetic speech artifacts during initial optimization, 
training is conducted in two stages:

1. **Stage 1 (Enhancement):** The U-Net front-end is pre-trained on an auxiliary noisy speech corpus to learn 
robust speech separation, and then frozen.
2. **Stage 2 (Detection):** The CRNN is trained on the ASVspoof 5 training set. Sequences are processed 
dynamically using PyTorch's `pack_padded_sequence` to maintain computational efficiency without truncating critical temporal data.

## Results
[cite_start]The primary evaluation metric is to achieve a development-set Equal Error Rate (EER) competitive with 
published SOTA on ASVspoof 5; current best result: 3.07% EER. The model was evaluated on `ASVspoof5.dev.track_1.tsv` 
to ensure strict testing against unseen speakers and unknown attack types.

**Final Baseline Metrics (Epoch 10):**
* **Training Accuracy:** 99.90%
* **Equal Error Rate (EER):** 40.40%
* **Optimal Threshold:** 0.9665

## Analysis and Next Steps
The baseline results exhibit a classic open-set generalization failure, which this multi-branch project was explicitly 
designed to investigate. Current spoofing countermeasures fail to generalize to unknown attack types.

* **The Overfitting Chasm:** The pipeline achieved near-perfect accuracy (99.90%) on the training set but failed to 
generalize to the development set. The model successfully memorized the known attack signatures but failed 
to learn the stable distribution of bonafide speech.
* **The "Denoiser Washout" Hypothesis:** The extreme Optimal Threshold (0.9665) indicates the model is heavily biased 
toward predicting "Fake" (0.0). This suggests the frozen U-Net is actively scrubbing away the subtle phase discontinuities 
and temporal irregularities necessary for the BiLSTM to detect genuine human speech.

**Next Pipeline Iteration:** To mitigate Denoiser Washout, the next phase will implement End-to-End Fine-Tuning. 
By unfreezing the U-Net and applying a differential learning rate, the BCE loss will propagate through the entire network, 
forcing the denoiser to suppress noise while actively preserving biometric deepfake artifacts.