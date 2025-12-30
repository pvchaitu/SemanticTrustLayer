This consolidated README provides a complete roadmap for the **Semantic Trust Layer (STL)** research, covering all 18 experiments from foundational black-box monitoring to advanced white-box mechanistic interpretability.

# ---

**Semantic Trust Layer (STL) Research Repository**

## **1\. Research Overview & Core Novelty**

The Semantic Trust Layer (STL) is a security framework designed to preserve **behavioral invariants** in LLM outputs. Traditional security (RLHF, keyword filtering) often fails against jailbreaks because it focuses on the *syntax* of the prompt. STL instead monitors the **semantic manifold** and **internal cognitive states** of the model to detect when it is being forced out of its "trusted" behavioral zone.

### **Key Innovations:**

* **Unsupervised Discovery**: Bootstrapping trust manifolds without labeled datasets using DBSCAN.  
* **Cognitive Turbulence**: Measuring the Euclidean "delta" between middle layers (intent) and final layers (output) to detect internal model conflict.  
* **Adaptive Criticality**: Dynamically adjusting security thresholds based on the model's internal energy baseline.  
* **Manifold Freezing**: Preventing "Adversarial Drift" by locking the trusted semantic space after a calibration period.

## ---

**2\. Environment Setup**

### **Software Requirements**

* **Python 3.10+**  
* **Ollama**: For black-box experiments (1–5).  
* **PyTorch & Transformers**: For white-box experiments (6–18).

### **Installation**

Bash

pip install torch transformers sentence-transformers scikit-learn numpy requests matplotlib

## ---

**3\. Experiment Roadmap**

### **Phase I: Foundational Manifold Discovery (Black-Box)**

* **experiment1.py**: Reference implementation using a static "trusted" output set.  
* **experiment2.py**: First unsupervised discovery using DBSCAN to find dense "safe" semantic regions.  
* **experiment3.py**: **Two-Manifold Fix**: Separates output into *Explanatory* (Safe) vs. *Procedural* (Risky) manifolds.  
* **experiment4.py**: Adds temporal accumulation and "Review Escalation" logic.  
* **experiment5.py**: **Adaptive-EPS**: Dynamically estimates clustering density thresholds.

### **Phase II: White-Box Signal Integration (Internal Metrics)**

* **experiment6.py**: Introduction of Hybrid STL using **Hidden State Variance** and **Attention Entropy**.  
* **experiment7.py**: Calibration of internal signals with research logging.  
* **experiment8.py**: Fixed attention "spikiness" saturation and linearized stability scaling.  
* **experiment9.py**: Final calibration of multi-layer attention sampling.  
* **experiment10.py**: **Manifold Freezing**: Implementing a "calibration window" to prevent poisoning.

### **Phase III: Cognitive Turbulence & Contrastive Analysis**

* **experiment11.py**: **Adaptive-Contrast**: Introduction of the Mid-vs-Final layer delta.  
* **experiment12.py**: Cross-architectural calibration (Qwen vs. Llama vs. Mistral settings).  
* **experiment13.py**: Multi-Layer Contrastive v4: Stability improvements.

### **Phase IV: Adversarial Suppression & Refinement**

* **experiment14.py**: **Aggressive Suppression**: Increasing PENALTY\_MULT to 80.0 to crush malicious scores.  
* **experiment15.py**: **Global Suppression**: Moving to PENALTY\_MULT 90.0 for near-zero false negatives.  
* **experiment16.py**: **Caution-Zone Scaling**: Gradient-based penalties for prompts near the trust boundary.  
* **experiment17.py**: **Malice Anchoring**: Weighting the manifold toward known "Bad Intent" anchors.  
* **experiment18.py**: **Collapse Detection**: Detecting "Safety Collapse" when internal variance drops to near zero under stress.

## ---

**4\. Execution Guide**

### **Running Black-Box Tests (1-5)**

Ensure Ollama is running (ollama serve) and the model is pulled (ollama pull llama3).

Bash

python experiment3.py

### **Running White-Box Tests (6-18)**

These require a GPU or high system RAM (16GB+) as they load the Qwen2.5-1.5B-Instruct model directly into memory to extract tensors.

Bash

python experiment18.py

## ---

**5\. Architectural Calibration (Tweaking Guide)**

If you switch the base model for experiments 11–18, update the constants in the script headers:

| Model | MID\_LAYER | TURB\_THRESHOLD | PENALTY\_MULT |
| :---- | :---- | :---- | :---- |
| **Qwen-2.5-1.5B** | 12 | 11,000 | 120.0 |
| **Llama-3-8B** | 16 | 23,000 | 145.0 |
| **Mistral-7B** | 14 | 15,000 | 125.0 |
| **Phi-3-Mini** | 10 | 5,000 | 80.0 |

## ---

**6\. Interpreting Results**

* **STL Score \> 0.60**: **TRUSTED**. The prompt's internal processing is stable and semantically familiar.  
* **STL Score 0.40 \- 0.60**: **CAUTION**. The system applies a gradient penalty; the model is acting unusual.  
* **STL Score \< 0.40**: **BLOCK**. High "Cognitive Turbulence" or semantic outlier detected. Likely an adversarial attempt or jailbreak.