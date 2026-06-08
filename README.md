# ⚡ Smart Grid Machine Learning: Electrical Fault Detection & Classification

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Domain-Electrical_Engineering-00599C?style=for-the-badge" alt="Electrical Engineering" />
  <img src="https://img.shields.io/badge/Conference-NSES_2024-8B0000?style=for-the-badge" alt="NSES 2024" />
</div>

## 📌 Executive Summary
This repository contains the research and codebase for an applied machine learning project intersecting **Data Science** and **Power Systems Engineering**. 

The objective was to evaluate various machine learning architectures for the automated detection and classification of faults in electrical transmission lines using raw current and voltage data. The findings of this project were formally presented at the **National Student Energy Summit (NSES) in May 2024**.

---

## 🔬 The Cross-Disciplinary Challenge (EE + ML)
A dependable electrical grid relies on the immediate detection of transmission line faults to prevent cascading power outages and hardware damage. Traditionally, this requires complex, threshold-based physical relays. 

This project proves that **Machine Learning can act as a digital relay system**. By treating three-phase current and voltage readings as feature vectors, we can train algorithmic classifiers to not only detect if a fault has occurred, but diagnose the specific *type* of fault instantly.

---

## 🛠️ Methodology & Technical Architecture

The project was divided into a two-tier pipeline to evaluate model readiness for real-time grid deployment:

### Phase 1: Binary Classification (Fault Detection)
* **Goal:** A rapid-response trigger to answer: *"Is there an anomaly in the power system?"*
* **Models Evaluated:** Random Forest, K-Nearest Neighbors (KNN), AdaBoost.
* **Result:** Achieved **near-perfect accuracy (1.0 F1-Score)**. The algorithms successfully mapped the threshold variances in current/voltage data to identify an active fault state with zero false positives.

### Phase 2: Multi-Class Classification (Fault Diagnosis)
* **Goal:** Diagnosing the specific physical nature of the fault (e.g., Line-to-Ground, Line-to-Line) to dispatch exact repair protocols.
* **Evaluation:** Utilized `LabelEncoder` to categorize fault states and tested algorithmic ability to separate the classes in high-dimensional space.
* **Result:** **Random Forest Classifier** significantly outperformed both KNN and AdaBoost. While certain fault types were predicted with 99.5% accuracy, complex multi-phase faults yielded lower precision boundaries (~56-58% F1-Scores on minority classes).

---

## 📊 Engineering Conclusions & Pragmatic Insights

Unlike many ML projects that claim 100% accuracy in a vacuum, this research yielded a pragmatic, industry-ready conclusion:
1. **Binary deployment is ready:** The binary classification models are sufficiently robust and reliable to be deployed as primary anomaly detection triggers in real-world grid SCADA systems today.
2. **Multi-class requires advanced architecture:** The multi-class classification drop-off proves that simple tree-based algorithms struggle with the non-linear overlapping of multi-phase electrical faults. 

**Future Scope:** Based on these results, the next iteration of this research will deprecate traditional ML classifiers in favor of **Convolutional Neural Networks (CNNs)** and **Transformer Neural Networks** to leverage attention mechanisms on time-series waveform data.

---

## 🚀 How to Run the Project

**1. Clone the repository**
```bash
git clone https://github.com/aansikkaw/electrical-fault-prediction.git
cd electrical-fault-prediction
