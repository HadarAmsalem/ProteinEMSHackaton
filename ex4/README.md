# Peptide Analysis with ESM-2 Embeddings

## Overview

This project implements a complete pipeline for analyzing and classifying peptide sequences using ESM-2 (Evolutionary Scale Modeling) embeddings. The system supports both sequence-based and structure-based classification using deep learning, distance-based heuristics, clustering, and ROC evaluation. Visualizations are generated to interpret the quality of predictions and clustering.

---

## Project Structure

```
.
├── ex4.py                    # Main script that orchestrates the entire pipeline
├── esm_embeddings.py         # ESM-2 model wrapper and embedding generation
├── neural_net.py             # Simple neural network classifier and utilities
├── pep_utils.py              # Peptide utilities: loading data, splitting, distances
├── plot.py                   # ROC, boxplot and embedding visualizations
├── clustering.py             # KMeans and dimensionality reduction (t-SNE)
├── structure_analysis.py     # Structure-based analysis using COM and pLDDT
├── requirements.txt          # List of required Python packages
```

---

## What Does the User Gain from Each Module?

### `ex4.py` — Main Execution Script

* Provides the **entire analytical workflow** from data loading to visualization.
* Serves as the starting point for understanding and customizing the pipeline.
* Allows you to experiment with embedding sizes, neural network parameters, and evaluation techniques.

### `esm_embeddings.py`

* Abstracts away the complexity of loading and interacting with various **pretrained ESM-2 models**.
* Provides a flexible API for obtaining **sequence-level or residue-level embeddings**, enabling various downstream analyses.

### `neural_net.py`

* Allows training of a **simple yet effective classifier** for distinguishing between positive and negative peptides.
* Easily replaceable or extendable with more complex architectures if desired.

### `pep_utils.py`

* Simplifies **data handling**, including loading, splitting, and computing meaningful distance metrics between peptides.
* Provides the building blocks for any distance-based method or pre-processing pipeline.

### `plot.py`

* Offers ready-to-use **visualizations for model evaluation and interpretation**.
* Users can immediately see how well their classifier separates classes via ROC and boxplots.

### `clustering.py`

* Enables **unsupervised analysis** through clustering and dimensionality reduction.
* Useful for detecting natural structure or patterns in embedding space.

### `structure_analysis.py`

* Provides tools for **3D structure-based evaluation** of peptides using structural files (PDB/mmCIF).
* Complements sequence-based models with structural metrics like COM distance and pLDDT.

---

## Key Differences Between the Modules

| Module                  | Primary Focus                    | Input Type              | Output Type                    |
| ----------------------- | -------------------------------- | ----------------------- | ------------------------------ |
| `ex4.py`                | Pipeline orchestration           | Peptide sequence files  | Plots, classification results  |
| `esm_embeddings.py`     | Embedding generation (ESM-2)     | Sequence list           | Embedding vectors              |
| `neural_net.py`         | Supervised learning (MLP)        | Embeddings + labels     | Trained model, scores          |
| `pep_utils.py`          | Data preparation and metrics     | Raw peptides/embeddings | Clean splits, distance vectors |
| `plot.py`               | Visualization                    | Scores, labels          | PNG images                     |
| `clustering.py`         | Unsupervised structure discovery | Embedding vectors       | Cluster labels, 2D coords      |
| `structure_analysis.py` | Structure-based classification   | 3D structure files      | COM/pLDDT scores, plots        |

Each module is loosely coupled and can be **reused independently** or extended for custom workflows.

---

## Program Flow Summary

1. **Data Preparation**

   * Load peptide sequences and assign labels.

2. **Sequence Embedding**

   * Generate fixed-size embeddings for all peptides using ESM-2.

3. **Exploratory Visualization**

   * Visualize amino acid-level embeddings.
   * Cluster peptide embeddings and visualize them in 2D.

4. **Distance-Based Classification**

   * For each test peptide, calculate average distances to positive and negative training embeddings.
   * Compute log-fold difference and evaluate with ROC.

5. **Neural Network Classification**

   * Train an MLP on training embeddings and evaluate on test set.

6. **Structure-Based Evaluation**

   * Calculate 3D structural properties (COM distance to reference, average pLDDT).
   * Use these as classification scores and evaluate with ROC.

---

## Requirements

Install using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Ensure all data files (e.g., peptide sequences, structure files) are placed in the correct folders:

   * `structures/`
   * `structures/af_positives/`
   * `structures/af_negatives/`

2. Run the main script:

```bash
python ex4.py
```





