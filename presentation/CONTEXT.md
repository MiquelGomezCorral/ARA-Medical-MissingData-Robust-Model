# Presentation Context — Full Information Dossier

This document contains ALL context and information used to write `main.tex`.
Plain text extracted from every source: codebases, PDFs, logs, datasets, commands.

---

## 1. PRESENTATION GUIDELINES (from assignment)

Duration: 10 minutes presentation + 5 minutes for questions
Number of slides: ∼10 (1 minute per slide)

Recommended structure:
1. Introduction — problem description
2. Related work — comparison to previous works (architecture or results)
3. Task description — dataset size, number of classes, vocabulary size, task complexity
4. Feature extraction/description — input/output of the model
5. Model or system architecture — general description, intuitive ideas, no code
6. Experimental design — train/dev/test partitions, statistics per set
7. Results — error/accuracy measure, table (max 8 rows) and/or figures
8. Discussion — what is the model learning? where does it fail? positive/negative examples
9. Conclusions — short summary with highlights
10. Future work — what would we try with more time? promising research lines

---

## 2. STAGE 0 PRESENTATION — Human Head MRI.pdf

(Previous attempt at the presentation, used as reference. Extracted via pdftotext.)

### Slide 1: Portada
Multimodal Classification of Life Expectancy with Missing Modality Robustness

### Slide 2: Índice
1. Introducción al problema
2. Trabajos relacionados
3. Descripción de la tarea
4. Datos
5. Modelos y mejoras

### Slide 3: Introducción al problema
- Datos faltantes en medicina — no siempre se disponen de todos los datos al hacer un diagnóstico
- PREDICCIÓN SIN TODOS LOS DATOS
  - No hay registros / se ha perdido
  - Fallo en la captura
  - No se puede realizar a tiempo (urgencia)
  - No es viable dada la situación del paciente
  - Coste económico
  - ...
- Ver qué partes de verdad son necesarias → mejorar futuros diagnósticos
- Modelos que generalicen mejor ...

### Slide 4: Trabajos relacionados
- Dataset: UPENN-GBM
- A: Se generan dos vistas con cutout y patch-swapping. Contrastive Learning → ambas representaciones similares entre sí. Context Reconstruction → capacidad de reconstruir la imagen original.
- B: Cross-Attention entre las características fisiológicas y las características extraídas del ViT.

### Slide 5: Descripción de la tarea
- Predicción en entornos reales
  - Gestión de datos faltantes
  - Identificar información necesaria para mejorar futuros diagnósticos
- Datos multimodales
  - Imágenes
  - Clínicos
  - Radiómicos

### Slide 6: Datos
- 630 Muestras de 603. Segundas visitas descartadas.
- Se descarta la gran mayoría de lo que contiene el dataset
- Resonancias Magnéticas: T1, T1GD, T2, FLAIR
- Datos tabulares sobre el paciente
- Datos Radiómicos sobre las radiografías
- Corromper alguna muestra con transformación
- Clasificación: Más de un año de vida o Menos de un año de vida
- Esquema: Hombre 74 Sí ... / Mujer 45 - ...

### Slide 7: Modelos y mejoras
- Introducción de las Radiómicas → No hay MRI. Que se puedan hacer predicciones usando solo Radiómicas o solo MRI
- Concepto de 'Dropout' como principal foco para entrenar y hacer robusto el modelo ante datos faltantes
- Informar al modelo adecuadamente cuando un dato falta
- Aprendizaje y fusión: SSL, Fusión mediante atención, Predicción de supervivencia

### Slide 8: Baseline (Resultados iniciales)

| Dropout Train | Dropout Test | F1-Score % | F1-Score % (Radiómicos) |
|--------------|-------------|-----------|------------------------|
| No           | No          | 63.38%    | 58.55%                 |
| Sí           | No          | 59.21%    | 53.40%                 |
| No           | Sí          | 68.47%    | 59.56%                 |
| Sí           | Sí          | 63.04%    | 51.75%                 |

---

## 3. PROJECT PROPOSAL — template_ara.pdf

(Original ARF project proposal 2024-2025. Extracted via pdftotext.)

### Title
Multimodal Classification of IDH1 Mutation with Missing Modality Robustness

### Team
Miquel Gómez Corral, Andrés Pacheco Millán

### Brief description
This project focuses on the multimodal classification of the IDH1 genetic mutation in Glioblastoma patients using the UPenn-GBM dataset. In real-world clinical settings, acquiring all imaging modalities is often impossible due to time constraints, patient movement, or missing equipment. Therefore, our primary objective is to develop a robust multimodal neural network capable of handling missing inputs, known as "Modality Dropout".

The dataset contains four 3D NIfTI MRI volumes (T1, T1-Gd, T2, T2-FLAIR) and tabular clinical data per patient. We will train a model using all available MRI sequences fused with tabular patient data to establish a strong classification baseline. During testing, we will simulate missing modalities by dynamically dropping specific MRI sequences (e.g., replacing T1-Gd and T2-FLAIR with zeros) and evaluating the model's ability to maintain predictive accuracy using only the remaining scans and tabular features.

The project workflow will involve preprocessing 3D medical images, implementing a multimodal fusion architecture, integrating a modality dropout layer during training, and evaluating performance under varying degrees of missing inputs.

### List of objectives
1. To collect and prepare the UPenn-GBM dataset (NIfTI volumes and clinical CSV)
2. To define experimental design: preprocess 3D MRI volumes and tabular data, defining training, dev and test sets
3. To select and design a multimodal neural network architecture supporting modality dropout
4. To implement the selected model and run experiments establishing a baseline with all modalities
5. To run inference experiments simulating missing MRI sequences
6. To analyse results and evaluate the model's robustness against missing data

### Note
The original proposal targeted IDH1 mutation classification. The prediction goal was changed to life expectancy (>12 months or ≤12 months of expected survival after operation). Do not mention this change in the presentation — use only the current goal.

---

## 4. BASE PAPER — Gomaa et al. 2024 (paper_ara.pdf)

### Full Citation
"Comprehensive multimodal deep learning survival prediction enabled by a transformer architecture: A multicenter study in glioblastoma"
Ahmed Gomaa, Yixing Huang, et al.
Neuro-Oncology Advances, 6(1), vdae122, 2024
DOI: https://doi.org/10.1093/noajnl/vdae122

### Abstract (extracted text)
Background. This research aims to improve glioblastoma survival prediction by integrating MR images, clinical, and molecular-pathologic data in a transformer-based deep learning model, addressing data heterogeneity and performance generalizability.

Methods. We propose and evaluate a transformer-based nonlinear and nonproportional survival prediction model. The model employs self-supervised learning techniques to effectively encode the high-dimensional MRI input for integration with nonimaging data using cross-attention. To demonstrate model generalizability, the model is assessed with the time-dependent concordance index (Cdt) in 2 training setups using 3 independent public test sets: UPenn-GBM, UCSF-PDGM, and Rio Hortega University Hospital (RHUH)-GBM, each comprising 378, 366, and 36 cases, respectively.

Results. The proposed transformer model achieved a promising performance for imaging as well as nonimaging data, effectively integrating both modalities for enhanced performance (UCSF-PDGM test-set, imaging Cdt 0.578, multimodal Cdt 0.672) while outperforming state-of-the-art late-fusion 3D-CNN-based models. Consistent performance was observed across the 3 independent multicenter test sets with Cdt values of 0.707 (UPenn-GBM, internal test set), 0.672 (UCSF-PDGM, first external test set), and 0.618 (RHUH-GBM, second external test set). The model achieved significant discrimination between patients with favorable and unfavorable survival for all 3 datasets (log-rank P 1.9×10⁻⁸, 9.7×10⁻³, and 1.2×10⁻²). Comparable results were obtained in the second setup using UCSF-PDGM for training/internal testing and UPenn-GBM and RHUH-GBM for external testing (Cdt 0.670, 0.638, and 0.621).

Conclusions. The proposed transformer-based survival prediction model integrates complementary information from diverse input modalities, contributing to improved glioblastoma survival prediction compared to state-of-the-art methods. Consistent performance was observed across institutions supporting model generalizability.

### Key Contributions (from paper)
1. A fully deep learning-based method for survival function prediction using a transformer architecture. Novel technical contributions include a self-supervised Vision Transformer for effective encoding of local and global information from MR imaging input with reduced risk of overfitting in the survival prediction task. First work to use cross-attention to capture interactions between clinical and molecular-pathologic data with multimodal MR images for individual patient survival function prediction.
2. The proposed transformer architecture outperforms state-of-the-art late-fusion 3D-CNN-based approaches as well as conventional CoxPH regression models for multimodal survival prediction in glioblastoma patients.
3. Consistent model performance demonstrated for public datasets from 3 institutions in 2 training setups, indicating model generalizability.

### Materials and Methods — Neural Network Architecture (extracted text)
We present a 2-step multimodal learning framework, illustrated in Figure 1. In the initial step, a ViT, acting as an encoder, employs self-supervised learning to acquire a clinically relevant encoding from multimodal whole-volume MR input datasets. In the second step, the encoded MRI inputs undergo integration with the encoded clinical and molecular-pathologic input using cross-attention. To prevent overfitting, the trained ViT with frozen parameters is utilized in this step. The outputs of the cross-attention, capturing interactions between modalities, are then passed to 2 fully connected layers for final survival prediction using attention pooling.

### Feature Extraction (extracted text)
To bridge the feature space gap between input modalities, we employ specialized encoders for each modality. Clinical and molecular-pathologic data, with their structured nature, are encoded using a trainable fully connected module trained end-to-end. However, the more complex volumetric MR input images undergo self-supervised pretraining for effective encoding of the brain's anatomical information (Figure 1A). Inspired by the self-supervised training of Swin UNETR, the trained imaging encoder model can learn how to generate clinically relevant representations from the high-dimensional imaging input, which can be efficiently fused with nonimaging clinical data for various medical downstream tasks.

The self-supervised pretraining process is formulated as a multi-objective loss function wherein the ViT-based encoder learns 2 proxy tasks, namely contrastive learning and context restoration. The selection of a transformer-based encoder instead of its conventional CNN-based counterpart is attributed to its capability of capturing global context and long-range dependencies in high-dimensional inputs, owing to its attention mechanism.

During pretraining, the multimodal MR volumes (T2, T2-FLAIR, T1 pre- and post-contrast) for each subject are stacked in the channel dimension. Subsequently, the volumes are augmented twice with random patch swapping and cutout, resulting in 2 views for each subject. The augmented views are then employed in the proxy tasks: Context restoration, and contrastive learning. Context restoration aids the encoder in grasping the structural intricacies and anatomical context of different brain regions. In parallel, self-supervised contrastive learning endeavors to learn representations that encapsulate discriminative and pertinent information from the data.

### Cross-Attention (extracted text)
For the fusion approach, we adapt the cross-attention methodology, introduced by Lu et al. for visio-linguistic representations. Cross-attention is a type of attention mechanism used in machine learning models, particularly in natural language processing and computer vision tasks, to facilitate interactions between 2 different inputs or sequences. In this fusion method, each modality is mutually reinforced with relevant information from the other via multi-head attention. The process of cross-attention enables the model to selectively focus on relevant parts of one sequence while processing another sequence, capturing the relationships between the 2. Cross-attention can be extended to effectively model the interactions between 2 different modalities by attending one modality while processing the other.

Given 2 representation vectors from 2 modalities, α and β, we can utilize cross-attention to generate a new, enriched representation of α based on β. The updated, enriched representation vector, denoted as α̂, is given by:
    α̂ = softmax(Qα·Kβᵀ/√dk) · Vβ
    = softmax(αWQα · (βWKβ)ᵀ/√dk) · βWVβ

where WQα, WKβ, and WVβ represent learnable weights that are used to generate the queries Qα, keys Kβ, and values Vβ, while dk is a scaling factor. In our implementation, the cross-attention mechanism is employed twice. In each instance, one modality is projected onto the query vector Q, while the other is projected onto the key K and value V vectors. Consequently, this process yields an enriched representation of each modality, drawing information from the other.

### Fusion and Prediction (extracted text)
Upon obtaining the updated representations for each modality through the preceding cross-attention layers, a concatenation of both vectors occurs in the sequence dimension. Subsequently, this concatenated set of high-level representations is channeled into the final attention layer, where it assumes the roles of key and value vectors alongside the clinical data projection as the query vector. The rationale for this approach stems from an empirical observation indicating that the nonimaging modality carries a higher degree of prognostically relevant information in comparison to the imaging modality. Finally, attention pooling is performed on the resultant vector, which is then passed to 2 fully connected layers for output prediction.

### Discrete-Time Survival Prediction (extracted text)
Discrete-time survival models are statistical tools employed to analyze and predict the occurrence of events over discrete time intervals. The model's loss function has 2 key parts. The first part penalizes the misordering of event pairs, enhancing accurate event sequencing. The second part, the log-likelihood term, captures the joint distribution of event times, accounting for right-censored data.

Integral to understanding this model's operation is the estimation of the Cumulative Incidence Function (CIF), denoted as F̂k(s| x). This function quantifies the likelihood of the event k happening on or before a specific time point. The CIF plays an integral role in deriving the survival function of each patient:
    S(t) = 1 − Σ F̂k(s| x)

where K is set to 1, as the single event of death is considered for the prediction of the patient-specific overall survival function. Ultimately, the model predicts the survival probability at 5 specific time points predetermined by the percentiles of the survival time within the training cohort. Subsequently, to establish survival probabilities on a monthly basis, a linear interpolation is applied, facilitating the estimation of survival probabilities for intermediate time intervals between these predefined points. Thus, the final output of the model is the prediction of an individual patient's survival function throughout the follow-up period (Figure 1B).

### Patient Characteristics — Setup 1 (UPenn training)
The training set included 264 patients, characterized by a mean age of 63.6 ± 11.6 years, and a male predominance (159 males). The median overall survival within this set was 381 days, with an interquartile range spanning 182–558 days. The validation set, consisting of 57 patients, exhibited a mean age of 64.3 ± 12.1 years, and a male distribution of 42 individuals. Within this subset, the median survival was observed to be 360 days, with an interquartile range extending from 167 to 550 days. Similarly, the internal test set, also comprising 57 patients, featured a mean age of 63.9 ± 11.8 years, with 34 males. The median survival in this set was recorded as 370 days, and the interquartile range was reported to be 182–553 days.

### Data Preprocessing (extracted text)
The 4 main morphological MRI modalities in glioblastoma T2, T2-FLAIR, and T1 pre- and post-contrast are used. Preprocessing steps include spatial normalization, bias field correction, and stripping of any residual skull parts. We stack the 4 morphological volumes in the channel dimension and perform histogram standardization and z-normalization. For the clinical data, the non-categorical covariates are normalized to have values between −1 and 1. Categorical covariates, on the other hand, were converted to one-hot encoded vectors.

### Evaluation and Statistical Analysis (extracted text)
The proposed model is comprehensively evaluated using statistical metrics and survival analysis techniques. The time-dependent concordance index (Ctd), an extension to Harrell's C-index, assesses predictive accuracy over time, and the integrated Brier score measures differences between observed and predicted survival probabilities. Kaplan–Meier survival analyses with Log-rank tests and Schoenfeld criteria for the hazard proportionality assumption are conducted. Results show that, except for age and resection status, all features meet the hazard proportionality criterion (P ≥ .05).

### Table 1: Clinical and Molecular-Pathologic Variables in the 3 Multicenter Datasets

| Variable | UCSF (n=366) | UPenn (n=378) | RHUH (n=36) |
|----------|-------------|--------------|-------------|
| Age (years), mean ± SD | 61.8 ± 12.03 | 63.8 ± 11.7 | 63.7 ± 9.01 |
| Median overall survival, days | 359 | 373 | 341.5 |
| Male, n (%) | 218 (59.6%) | 232 (61.4%) | 24 (66.6%) |
| Female, n (%) | 148 (40.4%) | 146 (38.6%) | 12 (33.4%) |
| GTR [>90% resection], n (%) | 213 (58.2%) | 214 (56.6%) | 36 (100%) |
| NTR [<90% resection], n (%) | 0 (0%) | 146 (38.6%) | 0 (0%) |
| NA, n (%) | 153 (41.8%) | 18 (4.8%) | 0 (0%) |
| MGMT Unmethylated, n (%) | 103 (28.1%) | 139 (36.8%) | 0 (0%) |
| MGMT Methylated, n (%) | 246 (67.2%) | 72 (19.0%) | 0 (0%) |
| MGMT Not available, n (%) | 17 (4.7%) | 167 (44.2%) | 36 (100%) |
| KPS ≥80, n (%) | - | 52 (13.7%) | 20 (55.5%) |
| KPS <80, n (%) | - | 14 (3.7%) | 16 (44.5%) |
| KPS NA, n (%) | 366 (100%) | 312 (82.6%) | 0 (0%) |

### Table 2: Summary of Results for all Training Setups and Input Modalities

Setup 1 (Model training on UPenn dataset):

| Data | Model | Metric | UPenn (Internal) | UCSF (external 1) | RHUH (external 2) |
|------|-------|--------|-----------------|-------------------|-------------------|
| Clinical and molecular-pathologic data | CoxPH | Cdt | 0.667 ± 0.009 | 0.642 ± 0.016 | 0.552 ± 0.022 |
| | | IBS | 0.093 ± 0.008 | 0.092 ± 0.011 | 0.145 ± 0.007 |
| | DeepSurv | Cdt | 0.644 ± 0.011 | 0.613 ± 0.024 | 0.570 ± 0.021 |
| | | IBS | 0.129 ± 0.004 | 0.167 ± 0.014 | 0.145 ± 0.005 |
| | DeepHit | Cdt | 0.641 ± 0.007 | 0.567 ± 0.023 | 0.603 ± 0.019 |
| | | IBS | 0.242 ± 0.004 | 0.203 ± 0.010 | 0.241 ± 0.007 |
| | Proposed transformer | Cdt | 0.667 ± 0.011 | 0.669 ± 0.021 | 0.613 ± 0.021 |
| | | IBS | 0.111 ± 0.006 | 0.162 ± 0.012 | 0.172 ± 0.006 |
| Imaging data | ViT end-to-end | Cdt | 0.622 ± 0.012 | 0.578 ± 0.019 | 0.602 ± 0.023 |
| | | IBS | 0.173 ± 0.006 | 0.208 ± 0.014 | 0.192 ± 0.009 |
| | Proposed self-supervised | Cdt | 0.645 ± 0.014 | 0.615 ± 0.020 | 0.609 ± 0.021 |
| | | IBS | 0.171 ± 0.007 | 0.197 ± 0.019 | 0.186 ± 0.012 |
| Combined | 3D-CNN + CoxPH | Cdt | 0.671 ± 0.010 | 0.659 ± 0.021 | 0.607 ± 0.020 |
| | | IBS | 0.081 ± 0.006 | 0.151 ± 0.009 | 0.141 ± 0.008 |
| Combined | Proposed transformer | Cdt | 0.707 ± 0.012 | 0.672 ± 0.023 | 0.618 ± 0.014 |
| | | IBS | 0.157 ± 0.003 | 0.179 ± 0.021 | 0.180 ± 0.006 |

Setup 2 (Model training on UCSF dataset):

| Data | Model | Metric | UCSF (internal) | UPenn (external 1) | RHUH (external 2) |
|------|-------|--------|-----------------|-------------------|-------------------|
| Clinical and molecular-pathologic data | CoxPH | Cdt | 0.647 ± 0.014 | 0.638 ± 0.017 | 0.598 ± 0.020 |
| | | IBS | 0.095 ± 0.005 | 0.077 ± 0.019 | 0.142 ± 0.007 |
| | DeepSurv | Cdt | 0.611 ± 0.010 | 0.599 ± 0.022 | 0.523 ± 0.023 |
| | | IBS | 0.161 ± 0.009 | 0.109 ± 0.014 | 0.177 ± 0.006 |
| | DeepHit | Cdt | 0.645 ± 0.013 | 0.562 ± 0.019 | 0.539 ± 0.021 |
| | | IBS | 0.112 ± 0.008 | 0.184 ± 0.010 | 0.217 ± 0.013 |
| | Proposed transformer | Cdt | 0.664 ± 0.011 | 0.638 ± 0.020 | 0.614 ± 0.019 |
| | | IBS | 0.183 ± 0.008 | 0.174 ± 0.013 | 0.201 ± 0.011 |
| Imaging data | ViT end-to-end | Cdt | 0.602 ± 0.014 | 0.603 ± 0.021 | 0.591 ± 0.022 |
| | | IBS | 0.187 ± 0.010 | 0.188 ± 0.011 | 0.194 ± 0.009 |
| | Proposed self-supervised | Cdt | 0.610 ± 0.016 | 0.617 ± 0.019 | 0.609 ± 0.018 |
| | | IBS | 0.186 ± 0.009 | 0.185 ± 0.014 | 0.189 ± 0.012 |
| Combined | 3D-CNN + CoxPH | Cdt | 0.650 ± 0.010 | 0.528 ± 0.018 | 0.587 ± 0.020 |
| | | IBS | 0.115 ± 0.007 | 0.096 ± 0.009 | 0.161 ± 0.007 |
| Combined | Proposed transformer | Cdt | 0.670 ± 0.009 | 0.638 ± 0.022 | 0.621 ± 0.019 |
| | | IBS | 0.172 ± 0.006 | 0.191 ± 0.018 | 0.184 ± 0.009 |

### Table 3: Ablation Study Results

| Model | Metric | UCSF |
|-------|--------|------|
| Reconstruction loss + contrastive × reconstruction loss | Cdt | 0.662 |
| | IBS | 0.171 |
| Reconstruction + contrastive loss | Cdt | 0.647 |
| | IBS | 0.165 |
| Contrastive loss | Cdt | 0.639 |
| | IBS | 0.192 |
| Reconstruction loss | Cdt | 0.623 |
| | IBS | 0.188 |
| ViT encoder (end-to-end) | Cdt | 0.632 |
| | IBS | 0.210 |
| ResNet Encoder (ImageNet) | Cdt | 0.613 |
| | IBS | 0.198 |

### Discussion (extracted text)
Prognosis prediction has been a long-standing interest in glioma research. Histologic morphology, molecular-pathologic factors, imaging features, clinical and treatment-related factors have all been shown to predict the course of the disease and assist in decision-making in glioblastoma patients. Important molecular-pathologic factors like WHO classification and selected clinical prognostic factors like age and performance status inform treatment algorithms in current neuro-oncologic guidelines. However, for conventional methods, integration of the plethora of known prognostic factors into an improved comprehensive prediction has been challenging, especially considering incorporating high-dimensional imaging data together with structured clinical and molecular-pathologic parameters. Novel deep learning models could be a key for such a comprehensive prognosis prediction integrating all available multimodal data streams for an individual patient, which could allow for more informed decision-making and treatment selection than selected single parameters are able to.

In this work, involving 780 patients from 3 datasets, we developed a transformer-based deep learning model, which integrates the prognostically relevant information from imaging, clinical, and molecular-pathologic data for improved survival prediction. The model employs a specialized imaging encoder, trained in a self-supervised setup, to generate clinically relevant representation vectors from high-dimensional MRI input.

### Feature Representation Analysis (extracted text)
We implemented the t-distributed Stochastic Neighbor Embedding (t-SNE) methodology to visualize the learned representations derived from the amalgamated multimodal data. A clear clustering pattern is evident: Patients with mutated MGMT methylation status are predominantly grouped in one region of the embedding space, while those with wild-type MGMT methylation status occupy the opposite region. Similarly, patients who underwent a gross total resection of more than 90% cluster distinctly from those who did not. These spatial arrangements are consistent with the survival curves, which indicate that tumors with MGMT methylation mutations and higher extent of resection are associated with improved prognoses.

### Subgroup Analysis (extracted text)
We assessed the model predictions from setup 1 for differentiating favorable from unfavorable prognosis in the important subgroups of IDH1 wild-type glioblastoma patients ≤70 and >70 years of age to evaluate the model predictions for treatment selection. Within the cohort of patients under 70 years, the median survival time for the predicted short survivor group was observed to be 8.7 months, while the long survivor group demonstrated a median survival time of 14.8 months (log-rank P-value = 8.4 × 10⁻⁶, n = 261). In the cohort aged 70 years or older, the median survival time for the predicted short survivor group was 5.6 months, while the long survivor group exhibited a median survival time of 8.1 months (P-value = 1.5 × 10⁻², n = 117).

---

## 5. PROJECT STRUCTURE — ARA-Medical-MissingData-Robust-Model

### Directory Tree
```
ARA-Medical-MissingData-Robust-Model/
├── AGENTS.md                              # Dev environment instructions
├── EXPERIMENT_LOG.md                       # Experiment log (Phases 0-3c)
├── README.md                               # Project template readme
├── TUNING_LOG.md                           # Hyperparameter tuning log
├── pyproject.toml                          # Python project config
├── requirements.txt                        # Dependencies
├── app/
│   ├── main.py                             # CLI entry point (argparse, 40+ hyperparams)
│   ├── scripts/
│   │   ├── train_model.py                  # Training orchestration (3 stages)
│   │   └── convert_to_tensor.py            # NIfTI → .pt converter (4 channels)
│   └── src/
│       ├── config/config.py                # Configuration dataclass
│       ├── data/
│       │   ├── UPennGBMDataset.py          # UPenn-GBM dataset + dropout
│       │   ├── augmentations_3d.py         # 3D augmentation pipelines
│       │   ├── brats_ssl_datamodule.py     # BraTS SSL data module
│       │   ├── lightning_datamodule.py     # Base LightningDataModule
│       │   └── ssl_datamodule.py           # UPenn SSL data module
│       ├── models/
│       │   ├── vit_encoder_3d.py           # 3D ViT encoder with patches
│       │   ├── ssl_module.py               # SSL encoder (recon + contrastive)
│       │   ├── ssl_lightning.py            # PyTorch Lightning for SSL
│       │   ├── survival_predictor.py       # Multimodal survival predictor
│       │   ├── survival_lightning.py       # Lightning module for survival
│       │   ├── tabular_tokenizer.py        # Tabular tokenizer with missing_emb
│       │   ├── cross_attention.py          # Cross-attention block
│       │   └── radiomic_normalizer.py      # Radiomic feature normalization
│       ├── training/
│       │   ├── train_ssl.py                # SSL pretraining
│       │   ├── train_survival.py           # Survival training
│       │   └── test.py                     # Evaluation/testing
│       └── utils/seed.py                   # Seeding utilities
├── configs/config.yaml                     # Default YAML config
├── data/
│   ├── BraTS/images_tensors_96/            # 1251 .pt tensor files (4×96×96×96)
│   ├── MR_NIfTI/images_tensors_96/         # 630 .pkl dict files (T1,T1GD,T2,FLAIR each 96³)
│   ├── dropout_data.json                   # Static dropout masks per patient
│   ├── partitions_ids.json                 # Train/val/test patient ID splits
│   ├── radiomic_processed.json             # Normalized radiomic features
│   └── ... (other data JSONs)
├── logs/lightning_logs/                    # 226 version dirs with metrics.csv
├── models/                                 # Checkpoints (5 runs)
└── scripts/train_all.sh                    # Batch experiment runner (5 variants)
```

---

## 6. EXPERIMENT LOG — EXPERIMENT_LOG.md

(Full content, exactly as written in the project)

Parameters:
- deterministic: False (cudnn benchmark=True, allows variance)
- SSL loss: l_recon + l_contrast × l_recon (unchanged)
- Data split: 80/10/10 (UPenn)
- Tabular: Gender {-1,1}, Age (x/50)-1
- 3D positional embeddings
- BraTS: RobustIntensityStandardization + ZNormalization in SSL pipeline
- Mask wiring: missing_emb + image_mask_proj (zero-init)

Baseline runs: --ssl_epochs 100 --survival_epochs 50
Experiment runs: --ssl_epochs 30 --survival_epochs 15 (faster screening)
All runs: --ssl_num_workers 0 --ssl_batch_size 8 --num_workers 0 --survival-batch-size 4 (avoids DataLoader worker OOM)

### Phase 1: New Baselines

B1 — BraTS SSL, 50/25 epochs:
- SSL dataset: BraTS (1251)
- Command: train -mtr -mts --ssl_epochs 50 --survival_epochs 25 --ssl_dataset brats
- F1: 0.5653, Precision: 0.5773, Recall: 0.5738, Accuracy: 0.5738
- Workers: 0 (IPC overhead hurts more than helps)
- Notes: Regression from 0.6640 (20/20). 50 SSL epochs + contrastive loss may collapse.

B2 — UPenn SSL, 50/25 epochs:
- SSL dataset: UPenn (630)
- Command: train -mtr -mts --ssl_epochs 50 --survival_epochs 25 --ssl_dataset upenn
- F1: 0.6053, Precision: 0.6093, Recall: 0.6066, Accuracy: 0.6066
- Workers: 12 (fast, no OOM)
- Notes: Better than B1 BraTS SSL (0.5653) but below 0.6640. Early stop at surv epoch 13.

### Phase 2: Revert Mask Wiring

Reverted: missing_emb in TabularTokenizer, image_mask_proj in SurvivalPredictor, mask forwarding in LightningModule/test.py.

M1 — BraTS SSL, no mask wiring, 20/10 epochs:
- F1: 0.5888, Precision: 0.5904, Recall: 0.5902, Accuracy: 0.5902
- vs B1 Δ: +0.0235 (but B1 had 50/25 epochs)
- Workers/Batch: 12/8
- Notes: batch_size=8 suboptimal; contrastive loss needs larger batches.

M2 — UPenn SSL, no mask wiring, 20/10 epochs:
- F1: 0.6053, Precision: 0.6093, Recall: 0.6066, Accuracy: 0.6066
- vs B2 Δ: 0.0000 (identical!)
- Workers/Batch: 12/32
- Notes: Mask wiring has ZERO effect on UPenn SSL. Batch size 32 same as batch 8.

### Phase 3a: Dynamic Dropout

Re-added _generate_dynamic_dropout() for train-time random missingness.

D1 — BraTS SSL, no dynamic dropout, 20/10 epochs, batch 32:
- F1: 0.6364, Precision: 0.6461, Recall: 0.6393, Accuracy: 0.6393
- vs M1 Δ: +0.0476 (batch 32 + no dynamic dropout)
- Notes: Batch size 32 critical for contrastive SSL.

D2 — UPenn SSL, no dynamic dropout, 20/10 epochs, batch 32:
- F1: 0.6382, Precision: 0.6426, Recall: 0.6393, Accuracy: 0.6393
- vs M2 Δ: +0.0329 (batch 32 + no dynamic dropout)
- Notes: Dynamic dropout removal helps; batch 32 is key.

### Phase 3b: 1D Positional Embeddings (NOT RUN — templated, empty results)
### Phase 3c: Augmentation Cross-Pollination (NOT RUN — templated, empty results)

---

## 7. TUNING LOG — TUNING_LOG.md

(Full content, exactly as written in the project)

| Step | Change | F1 | Precision | Recall | Accuracy | Δ F1 |
|------|--------|-----|-----------|--------|----------|------|
| 0b | Baseline (UPenn SSL, no masks, no BraTS pretrain) | 0.6518 | 0.6657 | 0.6557 | 0.6557 | — |
| 1 | Mask wiring (missing_emb + image_mask_proj zero-init) | 0.6382 | 0.6426 | 0.6393 | 0.6393 | −0.0136 |
| 3 | Tabular [-1, 1] (Gender {-1,1}, Age/50−1) | 0.6407 | 0.6926 | 0.6557 | 0.6557 | −0.0111 |
| 4 | Config + CLI fixes (infra only, no model Δ) | 0.6407 | 0.6926 | 0.6557 | 0.6557 | ±0 |
| 2 | BraTS ViT→SSL pretraining (40 SSL epochs) | 0.5735 | 0.5745 | 0.5738 | 0.5738 | −0.0672 |
| 5 | 3D positional embeddings (learnable grid 6³) | 0.6601 | 0.7057 | 0.6721 | 0.6721 | +0.0083 |
| 6 | BraTS intensity augs (hist std + z-norm) | 0.6640 | 0.6946 | 0.6721 | 0.6721 | +0.0039 |

### Step Explanations (from TUNING_LOG.md):

Step 2 — BraTS pretraining (REVERTED):
- BraTS SSL was already active — prepare_ssl_data uses BraTSSSLDataModule.
- Enabling ViT→SSL doubled epochs to 40. The loss l_recon + l_contrast × l_recon collapses contrastive signal as l_recon → 0.
- Reverted. BraTS SSL at 20 epochs is the effective baseline.

Step 5 — 3D positional embeddings:
- Replaced 1D learnable pos_embed (1, L+1, D) with learnable 3D grid (1, 6, 6, 6, D) + separate cls_pos.
- Mathematically: grid flattened yields same structure, but 3D initialization preserves spatial locality.
- KEPT. +0.0083 F1 over baseline.

Step 6 — BraTS intensity augmentations:
- Added RobustIntensityStandardization3D + ZNormalization3D before patch swap/cutout in BraTSSSLAugmentPipeline.
- Normalizes each SSL view independently before corruption, increasing augmentation diversity for contrastive learning.
- KEPT. +0.0039 F1 over Step 5, cumulative +0.0122 over baseline.

### Final Cumulative Result

| Metric | Baseline | Final | Δ |
|--------|----------|-------|---|
| F1 | 0.6518 | 0.6640 | +0.0122 |
| Precision | 0.6657 | 0.6946 | +0.0289 |
| Recall | 0.6557 | 0.6721 | +0.0164 |
| Accuracy | 0.6557 | 0.6721 | +0.0164 |

### Active Changes Summary (files modified)

| File | Change |
|------|--------|
| config.py | Session isolation (timestamped dirs), survival_batch_size, survival_num_workers |
| tabular_tokenizer.py | missing_emb param + mask-aware forward(x, mask) |
| survival_predictor.py | image_mask_proj (zero-init) injects MRI channel presence into sigmoid gate; tabular_mask routed to tokenizer |
| survival_lightning.py | Passes tabular_mask + image_mask to model |
| UPennGBMDataset.py | Gender {-1,1}, Age (x/50)-1 ⇒ [-1,1] range |
| vit_encoder_3d.py | 3D learnable positional embeddings (grid 6³) |
| augmentations_3d.py | RobustIntensityStandardization3D + ZNormalization3D in BraTS SSL pipeline |
| train_model.py | Uses config survival_batch_size/survival_num_workers |
| main.py | Fixed --batch-size typo, help text for --masked-* |
| test.py | Passes tabular_mask + image_mask to model |

### Step Details (from TUNING_LOG.md):

Step 0 — Session isolation: config.py __post_init__ creates models/{exp_name}_{timestamp}/ and logs/{exp_name}_{timestamp}/ per run.

Step 0b — Baseline: Current code with SSL on UPenn (no BraTS, no ViT pretraining, no mask wiring). Reference: 0.6518.

Step 1 — Mask wiring:
- tabular_tokenizer.py: missing_emb learnable param replaces zeroed features with learned values when mask passed.
- survival_predictor.py: image_mask_proj (zero-init last layer) injects MRI channel presence into sigmoid gate.
- survival_lightning.py: passes tabular_mask + image_mask to model.
- test.py: passes tabular_mask + image_mask to model.
- Result: F1 dropped 2.1%. Small regression — likely extra learnable params need more epochs or the zero-init projection converges slowly. Acceptable.

---

## 8. DATASET STATISTICS — UPenn-GBM

(Extracted via Python from the actual data files.)

### Source
https://www.cancerimagingarchive.net/collection/upenn-gbm/#citations

### Clinical Data (UPENN-GBM_clinical_info_v2.1.csv)
- Total rows: 671 (including both visit_11 and visit_21)
- Unique patients: 630
- Visit distribution: _11 = 611, _21 = 60
- After deduplication (visit_11 preferred): 630 samples from 603 patients

### Survival Statistics
- Survival days (from surgery): mean 496.3, std 508.2, min 3, 25%=167, 50%=376, 75%=618, max 6109
- >365 days: 339 patients (class 1 — "long survival")
- ≤365 days: 305 patients (class 0 — "short survival")
- Binary classification: 339 vs 305 (fairly balanced)

### Gender
- M: 378 (60%), F: 252 (40%)

### Age
- Mean: 62.7 ± 12.5 years, range: 18.6–88.5

### IDH1 Mutation Status
- Wildtype: 514 (81.6%)
- NOS/NEC: 97 (15.4%)
- Mutated: 19 (3.0%)

### MGMT Methylation Status
- Not Available: 327 (51.9%)
- Unmethylated: 160 (25.4%)
- Methylated: 115 (18.3%)
- Indeterminate: 28 (4.4%)

### KPS (Karnofsky Performance Status)
- Not Available: 555 (88.1%)
- 90: 37, 80: 18, 60: 6, 70: 6, 100: 5, 40: 2, 30: 1

### GTR (Gross Total Resection >90%)
- Yes: 362 (57.5%)
- No: 211 (33.5%)
- Not Available: 38
- Not Applicable: 19

### Survival Status
- Deceased: 603
- Alive: 17
- Lost to Follow-up: 7
- Deceased - uncertain date of death: 3

### MRI Tensor Files
- Location: data/MR_NIfTI/images_tensors_96/
- 630 .pkl files (dict format)
- Each .pkl is a dict with keys: ['T1', 'T1GD', 'T2', 'FLAIR']
- Each modality is a torch.Tensor of shape (96, 96, 96), float32
- ALL 630 patients have all 4 modalities present (no real missingness — we simulate it)

### BraTS Tensor Files
- Location: data/BraTS/images_tensors_96/
- 1251 .pt files
- Each is a torch.Tensor of shape (4, 96, 96, 96), float32
- Used only for SSL pretraining (not for survival prediction directly)

### Data Partitions (partitions_ids.json)
- train: 482 patients (80%)
- val: 60 patients (10%)
- test: 61 patients (10%)

### Dropout Data (dropout_data.json)
- 603 patients with dropout masks
- Reference probabilities:
  - T1: 0.05
  - T1GD: 0.05
  - T2: 0.05
  - FLAIR: 0.05
  - RADIOMIC: ET: 0.1, ED: 0.1, NC: 0.1
  - TABULAR: Gender: 0.1, Age: 0.1, KPS: 0.1, IDH1: 0.1, MGMT: 0.1, GTR: 0.1
- Sample patient dropout decision (UPENN-GBM-00010):
  ```
  {'T1': False, 'T1GD': True, 'T2': False, 'FLAIR': False,
   'RADIOMIC': {'ET': False, 'ED': False, 'NC': False},
   'TABULAR': {'Gender': True, 'Age': False, 'KPS': True, 'IDH1': False, 'MGMT': False, 'GTR': True}}
  ```

- Actual dropout counts across 603 patients:
  - T1: 26 patients (4.3%)
  - T1GD: 29 patients (4.8%)
  - T2: 25 patients (4.1%)
  - FLAIR: 23 patients (3.8%)
  - RADIOMIC_any: 155 patients (25.7%)
  - TABULAR_any: 290 patients (48.1%)
  - TABULAR_gender: 56 patients (9.3%)
  - TABULAR_age: 57 patients (9.5%)

### Radiomic Data (radiomic_processed.json)
- 585 patients with radiomic data
- 574 complete (all 4 modalities × 3 regions), 11 incomplete
- Structure per patient:
  - Top-level keys: ['FLAIR', 'T1', 'T1GD', 'T2']
  - Each modality has 3 regions: ['ED', 'ET', 'NC'] (Enhancing Tumor, Edema, Necrotic Core)
  - Each (modality, region) pair: 144 normalized radiomic features (float32 list)
  - Missing modality/region: None (replaced with zero vector + mask=0)
- Sample (UPENN-GBM-00010):
  - FLAIR/ED: 144 features, e.g., [-1.0755, -0.1579, -0.5]
  - FLAIR/ET: 144 features, e.g., [-1.0623, -0.2982, -0.8571]
  - FLAIR/NC: 144 features, e.g., [-1.0225, -0.2772, -0.6364]
  - T1/ED: 144 features, e.g., [-0.6740, -0.2238, -0.5714]
  - T1/ET: 144 features, e.g., [0.0091, -0.4484, 0.3333]
  - T1/NC: 144 features, e.g., [-0.8138, -0.2823, -0.5]
  - T1GD/ED: 144 features, e.g., [-1.0133, -0.1018, -0.5]
  - T1GD/ET: 144 features, e.g., [-1.0633, -0.3331, -0.3125]
  - T1GD/NC: 144 features, e.g., [-1.4400, -0.1618, -0.7778]
  - T2/ED: 144 features, e.g., [-0.9321, -0.4052, -0.9846]
  - T2/ET: 144 features, e.g., [-1.0835, -0.5066, -1.2727]
  - T2/NC: 144 features, e.g., [-0.9272, -0.4212, -0.9583]

---

## 9. MODEL ARCHITECTURE — Code Analysis

### 9.1 ViTEncoder3D (vit_encoder_3d.py)

3D Vision Transformer encoder.

Components:
- PatchEmbed3D: Conv3d(in_channels=4, embed_dim=256, kernel_size=16, stride=16)
  - Input: (B, 4, 96, 96, 96)
  - Grid: 96/16 = 6³ = 216 patches
  - Output: (B, 216, 256) patch tokens

- cls_token: learnable parameter (1, 1, 256) prepended to patch tokens
  - Total sequence: 217 tokens

- Positional embeddings:
  - Option "1d": pos_embed (1, 217, 256) — standard 1D position encoding
  - Option "3d": pos_embed (1, 6, 6, 6, 256) + cls_pos (1, 1, 256)
    - 3D grid flattened to (1, 216, 256) preserving spatial locality
    - All initialized with truncated normal (std=0.02)

- Transformer blocks: 4 layers
  - Each: LayerNorm → MultiheadAttention (8 heads, batch_first) → residual
          → LayerNorm → MLP (256→1024→256) with GELU + Dropout(0.1) → residual

- Final LayerNorm
- Output: (B, 217, 256) normalized tokens

- _resize_input: uses trilinear interpolation if input doesn't match vol_size

### 9.2 SSL Pretraining (ssl_module.py)

SSLPretraining module wraps the ViTEncoder3D.

Components:
- ContrastiveHead:
  - Linear(256→256) → ReLU → Linear(256→128)
  - Takes [CLS] token → 128-dim normalized projection
  - F.normalize output (L2 norm)

- ReconstructionHead:
  - Linear(256 → 4×16³ = 16384)
  - Takes patch tokens (skip CLS), reconstructs original pixel patches

Forward pass (x_i, x_j, target):
1. Encode both views: tokens_i = encoder(x_i), tokens_j = encoder(x_j)
2. Contrastive: z_i = contrastive_head(tokens_i[:,0]), z_j = contrastive_head(tokens_j[:,0])
   - NT-Xent loss: sim = z @ zᵀ / temperature, masked diagonal, cross-entropy
   - temperature = 0.5
3. Reconstruction: recon_i = reconstruction_head(tokens_i[:,1:])
   - target_patches = _extract_patches(target) → (B, 216, 16384)
   - MSE loss on patch reconstructions
4. Total loss: L_ssl = L_recon + L_contrast × L_recon
   - The contrastive loss is weighted by the reconstruction loss magnitude
   - As L_recon → 0, the contrastive term collapses (observed problem)

### 9.3 TabularTokenizer (tabular_tokenizer.py)

Maps 14 clinical features → 8 tokens of embed_dim 256.

Components:
- Sequential network:
  - Linear(14 → 128) → GELU → Dropout(0.1)
  - Linear(128 → 8×256) → Dropout(0.1)
  - Reshape to (B, 8, 256)
  - LayerNorm(256)

- missing_emb: learnable parameter of shape (14,)
  - Initialized with truncated normal (std=0.02)
  - Applied when mask is passed: x = x + (1 - mask) × missing_emb
  - Effect: replaces zeroed features with learned values when feature is masked as absent
  - The model learns what value to insert for missing features

### 9.4 RadiomicTokenMLP (survival_predictor.py)

Shared MLP projecting each radiomic channel-token independently.

Input: x (B, N_tok, 144), mask (B, N_tok) bool True=valid
Output: proj (B, N_tok, 256) — invalid tokens zeroed out

Architecture: 3 layers
- Linear(144 → 512) → GELU → Dropout(0.1)
- Linear(512 → 512) → GELU → Dropout(0.1)
- Linear(512 → 256)
- LayerNorm(256)
- Zero masking: proj = proj × mask.unsqueeze(-1).float()

N_tok = 12 (3 regions × 4 modalities)

### 9.5 TokenWiseSigmoidGate (survival_predictor.py)

Fuses radiomic summary with image tokens.

Input: img_tokens (B, S, 256), rad_summary (B, 256)
Output: fused (B, S, 256)

Mechanism:
1. rad_exp = rad_summary.unsqueeze(1).expand(-1, S, -1)  → (B, S, 256)
2. gate_in = cat([img_tokens, rad_exp], dim=-1)          → (B, S, 512)
3. sigma = sigmoid(Linear(512 → 256)(gate_in))           → (B, S, 256)
4. fused = img_tokens × sigma + rad_exp × (1 - sigma)    → (B, S, 256)

Key property: per-token, per-dimension gating
- No valid radiomic tokens → rad_summary ≈ 0 → sigma → 1 → pure image
- Full radiomics → sigma learns how much radiomic info to blend per token
- No explicit if-branch needed — the mask system handles all missingness patterns

### 9.6 CrossAttentionBlock (cross_attention.py)

Standard pre-norm cross-attention block.

Components:
- norm_q: LayerNorm(256) on query
- norm_kv: LayerNorm(256) on key/value
- MultiheadAttention(256, 8 heads, dropout=0.1, batch_first=True)
- Residual connection for query
- norm2: LayerNorm(256)
- FFN: Linear(256→1024)→GELU→Dropout→Linear(1024→256)→Dropout
- Residual connection for FFN output

Forward: q = norm_q(query), kv = norm_kv(context)
  → attn_out = MHA(q, kv, kv)
  → query = query + attn_out
  → query = query + FFN(norm2(query))
  → returns updated query only (context unchanged)

### 9.7 MultimodalSurvivalPredictor (survival_predictor.py) — Complete Architecture

Parameters:
- num_classes: 2 (binary survival)
- embed_dim: 256
- num_heads: 8
- dropout: 0.1
- in_channels: 4 (MRI modalities)
- patch_size: 16
- vit_depth: 4
- vol_size: 96
- tabular_in: 14
- tabular_tokens: 8
- tabular_hidden: 128
- radiomic_n_features: 144
- radiomic_mlp_hidden: None → 2×embed_dim = 512
- radiomic_mlp_layers: 3
- pos_embed: "1d"
- use_radiomics: True/False

Forward pass (complete flow):

1. Image encoding: image → ViTEncoder3D → img_tokens (B, 217, 256)

2. Radiomic pipeline (if use_radiomics=True):
   - Concatenate 3 regions: rad_x (B, 12, 144)
   - Concatenate masks: rad_mask (B, 12) bool
   - RadiomicTokenMLP: rad_tokens (B, 12, 256)
   - Masked mean pool: rad_summary (B, 256)
     - masked_mean_pool sums valid tokens, divides by count (clamped >=1)
     - All-masked → rad_summary ≈ 0 vector
   - If image_mask is not None:
     - image_mask_proj: Linear(4→256)→GELU→Linear(256→256), zero-initialized last layer
     - rad_summary = rad_summary + image_mask_proj(image_mask)
     - Injects knowledge of which MRI channels are present into the gate
   - TokenWiseSigmoidGate: fused_tokens = gate(img_tokens, rad_summary) (B, 217, 256)

3. Tabular tokenization: tabular → TabularTokenizer(tabular, mask=tabular_mask) → tab_tokens (B, 8, 256)

4. Two-way cross-attention:
   - ca_fused_on_tab: Q = fused_tokens, KV = tab_tokens → fused_out (B, 217, 256)
   - ca_tab_on_fused: Q = tab_tokens, KV = fused_tokens → tab_out (B, 8, 256)

5. Mean pool both streams:
   - fused_pool = fused_out.mean(dim=1) → (B, 256)
   - tab_pool = tab_out.mean(dim=1) → (B, 256)

6. Final cross-attention:
   - combined = stack([fused_pool, tab_pool], dim=1) → (B, 2, 256)
   - final_out = ca_final(Q=combined, KV=combined) → (B, 2, 256)
   - pooled = final_out.mean(dim=1) → (B, 256)

7. Survival head:
   - classifier: Linear(256 → 2)
   - logits = classifier(dropout(pooled)) → (B, 2)

Missing modality philosophy (from docstring):
There are NO explicit if-branches for missing modalities. Missingness is handled entirely through masks:
- invalid radiomic tokens are zeroed before and after MLP
- masked mean pool ignores them for the summary vector
- a fully-absent sample produces rad_summary≈0, gate learns σ→1 (pure image)
- the model generalises to any missingness pattern at inference time

### 9.8 MultimodalSurvivalLightningModule (survival_lightning.py)

Lightning wrapper around MultimodalSurvivalPredictor.

- Criterion: CrossEntropyLoss(label_smoothing=0.1)
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR(T_max=epochs)
- Metrics logged: loss, accuracy, balanced_accuracy, F1 (macro)
- Handles forwarding of tabular_mask and image_mask to the model

### 9.9 RadiomicTokenizer (radiomic_normalizer.py) — Alternative simple version

Simple projection alternative (not used in current predictor):
- Input: x (B, 4, N_feat), mask (B, 4) — 1=valid, 0=null
- proj = Linear(N_feat → embed_dim)
- norm + dropout
- tokens = tokens × mask.unsqueeze(-1) — zero out null modalities
- Returns (B, 4, embed_dim)

This simpler version is in a separate file and represents an earlier design iteration.

---

## 10. AUGMENTATIONS — augmentations_3d.py

### SSL Augmentations (MRIAugmentPipeline)
Applied twice independently to create positive pairs (TwoViewTransform).
Order: Normalize → Spatial → Intensity → Corruption

Pipeline:
1. RobustIntensityStandardization3D(nonzero=True) — 1st/99th percentile clipping, scale [0,1]
2. ZNormalization3D(nonzero=True) — z-score on brain tissue only
3. RandomFlip3D(p=0.5) — flip on any spatial axes
4. RandomRotate90_3D(p=0.5) — 0/90/180/270° rotation in random plane
5. RandomIntensityScaling(lo=0.9, hi=1.1) — multiply by random scalar
6. RandomGaussianNoise(max_std=0.05) — additive zero-mean noise
7. RandomCrop3D(target_size=96, min_scale=0.85) — crop + resize back
8. RandomPatchSwap3D(patch_size=12, p=0.8, swaps=1) — swap random cubic patches
9. RandomCutout3D(min_ratio=0.1, max_ratio=0.25, p=0.8) — mask random cubic region

### BraTS-Specific SSL Augmentations (BraTSSSLAugmentPipeline)
Same pipeline with additional augmentations for BraTS.

Pipeline:
1. RobustIntensityStandardization3D(nonzero=True)
2. ZNormalization3D(nonzero=True)
3. RandomFlip3D(p=0.5) — Phase 3c addition
4. RandomRotate90_3D(p=0.5) — Phase 3c addition
5. RandomIntensityScaling(lo=0.9, hi=1.1) — Phase 3c addition
6. RandomGaussianNoise(max_std=0.05) — Phase 3c addition
7. RandomCrop3D(target_size=96, min_scale=0.85) — Phase 3c addition
8. RandomPatchSwap3D(patch_size=12, p=0.8, swaps=1)
9. RandomCutout3D(min_ratio=0.1, max_ratio=0.25, p=0.8)

### Survival Augmentations (SurvivalAugmentPipeline)
Conservative augmentations — preserves tumor morphology and anatomical orientation.

Order rationale:
1. Normalize first (consistent space for intensity augs)
2. Spatial transforms (geometry before intensity corruption)
3. Intensity augs (scanner variability simulation)
4. Mild blur last

Pipeline:
1. RobustIntensityStandardization3D(nonzero=True)
2. ZNormalization3D(nonzero=True)
3. SurvivalSafeFlip3D(p=0.5) — left-right flip only (H axis)
4. RandomRotate3D(max_angle=10°, p=0.5) — small rotations, trilinear interpolation
5. RandomTranslate3D(max_shift=0.05, p=0.5) — small translations
6. RandomBiasField(max_strength=0.3, p=0.5) — smooth multiplicative field
7. RandomGammaCorrection(gamma_range=(0.7, 1.5), p=0.5) — scanner contrast variability
8. RandomIntensityScaling(lo=0.9, hi=1.1)
9. RandomGaussianNoise(max_std=0.05)
10. RandomGaussianBlur3D(max_sigma=1.0, p=0.3) — low-SNR simulation

NOT included vs SSL (survival-safe):
- No RandomPatchSwap3D (breaks local anatomy/label signal)
- No RandomCutout3D (destroys tumor regions)
- No RandomCrop+resize (changes apparent tumor size)
- No RandomRotate90_3D (90° rotations break anatomical orientation)

### Survival Inference Pipeline
Normalization only, no augmentation:
1. RobustIntensityStandardization3D(nonzero=True)
2. ZNormalization3D(nonzero=True)

### Additional Augmentation Classes
- RandomBiasField: simulates MRI bias field via linear gradients on each axis
- RandomGammaCorrection: out = in^gamma, simulates scanner contrast, nonzero voxels only
- RandomGaussianBlur3D: depthwise 3D conv with Gaussian kernel

---

## 11. DATA LOADING — Data Flow

### UPennGBMDataset (UPennGBMDataset.py)

Per-sample return:
```
{
    'label':         tensor (scalar, long),           # 0 or 1 (survival class)
    'image':         tensor (4, 96, 96, 96),          # stacked MRI channels
    'image_mask':    tensor (4,),                      # 1.0=present, 0.0=missing per channel
    'tabular':       tensor (14,),                     # clinical features
    'tabular_mask':  tensor (14,),                     # 1.0=present, 0.0=missing per feature
    'radiomic_ED':   tensor (4, 144),                  # ED region radiomic features
    'radiomic_ET':   tensor (4, 144),                  # ET region
    'radiomic_NC':   tensor (4, 144),                  # NC region
    'radiomic_mask_ED': tensor (4,),                   # validity masks
    'radiomic_mask_ET': tensor (4,),
    'radiomic_mask_NC': tensor (4,),
}
```

Tabular columns (14 features):
- 'Gender': -1 (F) or 1 (M)
- 'Age_at_scan_years': (age/50) - 1  → range [-1, 1]
- 'KPS_High', 'KPS_Low', 'KPS_Unk': one-hot dummies
- 'IDH1_Mutated', 'IDH1_Wildtype', 'IDH1_Unk': one-hot dummies
- 'MGMT_Methylated', 'MGMT_Unmethylated', 'MGMT_Unk': one-hot dummies
- 'GTR_Y', 'GTR_N', 'GTR_Unk': one-hot dummies

Missing data handling:
- MRI: channel set to zeros + image_mask = 0
- Tabular: feature set to 0 + tabular_mask = 0 (then missing_emb replaces in tokenizer)
- Radiomic: feature vector set to zeros + mask = 0

Dynamic dropout (train only, if enabled):
- _generate_dynamic_dropout(pid): samples dropout decisions per epoch based on reference probabilities
- Static dropout (test, default): uses precomputed dropout_data.json masks

Data split:
- partition_ids.json: {"train": [482 IDs], "val": [60 IDs], "test": [61 IDs]}
- Intersection with clinical CSV: matched by patient root ID

### BraTSSSLDataModule (brats_ssl_datamodule.py)

BraTSTensorDataset:
- Loads .pt tensors from data/BraTS/images_tensors_96/
- Excludes overlapping IDs with UPenn test set (overlap_ucsf_test_ids.json — currently empty [])
- Per-channel z-normalization on load: (channel - mean) / std.clamp(1e-5)

BraTSSSLDataset:
- Wraps base dataset with TwoViewTransform(BraTSSSLAugmentPipeline)
- Returns: (x_i, x_j, target_image)

BraTSSSLDataModule:
- Single train dataloader (no val/test for SSL)
- Batch size: 32, workers: 12, pin_memory, drop_last, persistent_workers

### SSLDataModule — UPenn SSL (ssl_datamodule.py)

SSLImageDataset:
- Wraps UPennGBMDataset with TwoViewTransform(MRIAugmentPipeline)
- Returns: (x_i, x_j) — two augmented views, no target

SSLDataModule:
- Forces masked_train=False, masked_test=False (SSL needs full images)
- Concatenates train + val + test bases for SSL (603 total)
- Single train dataloader

---

## 12. TRAINING PIPELINE — train_model.py

### prepare_ssl_data(CONFIG)
- Branch on CONFIG.ssl_dataset: "brats" → BraTSSSLDataModule, "upenn" → SSLDataModule
- SSL Train samples: BraTS = 1251, UPenn = 603

### prepare_survival_data(CONFIG)
- UPennGBMDataModule with train/val/test splits
- Train: 482, Val: 60, Test: 61
- DataLoaders with batch_size, shuffle, pin_memory

### train_3d_vit(CONFIG) — Main pipeline
1. set_all_seeds(seed)
2. prepare_data → ssl_train_loader + survival dataloaders
3. train_stage_ssl → ssl_checkpoint_path
4. train_stage_survival → survival_module
5. test_model → results dict (saved as results.json)

---

## 13. CONFIGURATION — config.py

### Key Configuration Parameters

Paths:
- mr_data: data/MR/metadata/UPENN-GBM_clinical_info_v2.1.csv
- mr_nf_tensors_96: data/MR_NIfTI/images_tensors_96/
- brats_tensors_96: data/BraTS/images_tensors_96/
- partition_ids_path: data/partitions_ids.json
- dropout_data_path: data/dropout_data.json
- radiomic_data_path: data/radiomic_processed.json

Model Parameters:
- bins: [0, 365, inf]  (binary survival at 12 months)
- labels: [0, 1]  (short vs long)
- ssl_dataset: "brats" (default) or "upenn"
- ssl_epochs: 30, survival_epochs: 20
- pos_embed: "1d" or "3d"
- use_radiomics: False (default)
- dynamic_dropout: False (default)
- freeze_encoder: False

SSL Parameters:
- ssl_embed_dim: 256, ssl_vit_depth: 4, ssl_patch_size: 16
- ssl_proj_dim: 128, ssl_temperature: 0.5
- ssl_learning_rate: 1e-4, ssl_weight_decay: 1e-4
- ssl_batch_size: 32, ssl_num_workers: 12
- ssl_aug_patch_size: 12, ssl_cutout_min_ratio: 0.10, ssl_cutout_max_ratio: 0.25

Survival Parameters:
- survival_batch_size: 16, survival_num_workers: 12

### Session Management
- __post_init__ creates timestamped directories:
  - models/{exp_name}_{timestamp}/
  - logs/{exp_name}_{timestamp}/

---

## 14. CLI ARGUMENTS — main.py

### Subcommand: train

Key arguments:
- --base_name: experiment name (default: "default")
- -mtr / --masked_train: enable missing data masking during training
- -mts / --masked_test: enable missing data masking during test
- --ssl_dataset: "brats" or "upenn"
- --pos_embed: "1d" or "3d"
- --use_radiomics: enable radiomic features + sigmoid gate
- --dynamic_dropout: enable train-time random dropout
- --ssl_epochs: number of SSL epochs (default: 30)
- --ssl_batch_size: SSL batch size (default: 32)
- --survival_epochs: number of survival training epochs (default: 20)
- --survival_batch_size: survival batch size (default: 16)
- --learning_rate: survival learning rate (default: 1e-4)
- --label_smoothing: (default: 0.1)
- --embed_dim: shared token dimension (default: 256)
- --num_heads: attention heads (default: 8)
- --dropout: dropout rate (default: 0.1)
- --vit_depth: ViT depth (default: 4)
- --vol_size: input volume size (default: 96)
- --tabular_tokens: number of tabular tokens (default: 8)
- --freeze_encoder: freeze image encoder during survival training
- --no_early_stopping: disable early stopping
- --early_stopping_patience: (default: 10)

---

## 15. BATCH EXPERIMENT SCRIPT — train_all.sh

Runs 5 experiments sequentially:

1. "No Masks - Emb D1 - No Radiomics - No D Dropout"
   - python main.py train --base_name 'No_Masks'
   - Test F1: 0.6695, Precision: 0.6801, Recall: 0.6721, Accuracy: 0.6721

2. "All masks - Emb D1 - No Radiomics - No D Dropout"
   - python main.py train -mts -mtr --base_name 'All_masks'

3. "All masks - Emb D3 - No Radiomics - No D Dropout"
   - python main.py train -mts -mtr --pos_embed 3d --base_name 'All_masks-Emb_D3'

4. "All masks - Emb D1 - Radiomics - No D Dropout"
   - python main.py train -mts -mtr --use_radiomics --base_name 'All_masks-Radiomics'

5. "All masks - Emb D1 - No Radiomics - D Dropout"
   - python main.py train -mts -mtr --dynamic_dropout --base_name 'All_masks-D_Dropout'

---

## 16. GIF TO PNG CONVERSION

The MRI visualization GIF (UPENN-GBM-00465.gif) was converted to PNG because PDFLaTeX does not support GIF format natively.

Command: convert "UPENN-GBM-00465.gif[0]" UPENN-GBM-00465.png
Result: 960×240 grayscale PNG (first frame of the animated GIF)
Original GIF: 7.2MB, 960×240, multi-frame animated MRI visualization

For animated presentations, use the `animate` package with `\animategraphics` after splitting GIF into frames.

---

## 17. PRESENTATION STRUCTURE — Final Slides

| Slide | Section | Content |
|-------|---------|---------|
| 1 | Portada | Título, autores, UPV |
| 2 | Índice | Tabla de contenidos |
| 3 | Introducción | Problema de datos faltantes + causas + GIF cerebral |
| 4 | Trabajos Relacionados | Paper Gomaa et al. 2024 + Tabla 2 resumida |
| 5 | Descripción de la Tarea | Dataset UPenn-GBM + tabla de estadísticas |
| 6 | Extracción de Características | 3 columnas: MRI, tabular, radiómicos + shapes I/O |
| 7 | Arquitectura (Visión General) | Placeholder diagrama + SSL + predicción |
| 8 | Arquitectura (Detalles) | Sigmoid gate + cross-attention + mask wiring |
| 9 | Nuestras Contribuciones | 6 cambios respecto al paper + principio de diseño |
| 10 | Diseño Experimental | Split 482/60/61 + SSL + hiperparámetros |
| 11 | Resultados (Ablación) | 6 pasos TUNING_LOG + tabla acumulada |
| 12 | Resultados (Baseline) | Comparativa template_ara.pdf (4 configs) |
| 13 | Resultados (Experimentos) | EXPERIMENT_LOG fases B1-D2 + lecciones |
| 14 | Discusión | Fortalezas, debilidades, hallazgo principal |
| 15 | Conclusiones + Trabajo Futuro | 5 conclusiones, 6 líneas futuras |
