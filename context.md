You are an expert ML Research Engineer, here is a spec for an experiment I want
you to implement. Ask me for any clarifications and use your expertise to push
back on anything that doesn't make sense.

# Research Spec: Continuous Signal Resampling for Network Scaling

**Target Hardware:** Single H100 Node
**Framework:** PyTorch, uv for dependency management
**Logging:** TensorBoard

## 1. Abstract & Objective

The objective of this project is to validate a new method for scaling neural network width called **Continuous Signal Resampling**.

Current methods for expanding network width (e.g., Net2Net) treat neurons as discrete, unordered entities, often using duplication and noise to expand capacity. This project posits that neural layers have an underlying continuous manifold structure. By discovering the "canonical topological order" of neurons (via Spectral Seriation) and treating the weight vector as a discretized continuous signal, we can scale up the network using **Linear Resampling** (interpolation).

**Hypothesis:** Expanding a network via sorted resampling will result in a smaller loss spike and faster convergence to the lower loss basin compared to:

1. Training the larger model from scratch.
2. Expanding via standard Net2Net (random mapping/duplication).

---

## 2. Theoretical Background

### 2.1 The Manifold View vs. The Discrete View

Standard deep learning optimization (Gradient Descent) is the "Ant's Eye View"—local traversal on a loss manifold. We aim to take the "Bird's Eye View" by exploiting the global structure of the parameter space.

* **Premise:** The permutation symmetry of neurons implies that for any layer of size , there are  equivalent representations.
* **Insight:** Among these permutations, there exists a "canonical" sorting where functionally similar neurons are adjacent. In this sorted state, the layer weights approximate a smooth function  rather than a jagged discrete set.
* **Implication:** If the layer is smooth, we can increase its resolution (scale width ) via signal processing techniques (interpolation) rather than discrete graph operations.

### 2.2 Prior Work

**A. Git Re-Basin (Ainsworth et al., 2022)**

* **Core Concept:** Models trained from different seeds can be linearly interpolated if their weights are permuted to align. This implies a single global loss basin modulo permutation.
* **Algorithm (Matching):** Uses linear assignment (Hungarian algorithm) to maximize cosine similarity between units of two models.
* **Relevance:** We adapt this not to match *two* models, but to match a model *to itself* (Self-Organization). We use **Spectral Seriation** to sort neurons such that adjacent neurons have maximal cosine similarity.

**B. Net2Net (Chen et al., 2016)**

* **Core Concept:** Function-Preserving Transformations. To widen a layer, Net2Net selects a neuron , copies it to create a new neuron , and splits the outgoing weight magnitude (e.g., ).
* **Limitation:** The splitting is discrete and local. It does not respect the global topology of the feature space. It creates "cliffs" in the loss landscape unless the copied neurons are treated as identical (which stalls feature diversity).

---

## 3. The Proposed Algorithm: Continuous Signal Resampling

Our method consists of three distinct phases performed during the scaling step.

### Phase 1: Canonical Sorting (Spectral Seriation)

We must first recover the smooth manifold structure from the unordered "bag of neurons."

* **Input:** Weight matrix  (or strictly the hidden dimension weights).
* **Similarity Matrix:** Compute the cosine similarity matrix  between all column vectors of .


* **Laplacian:** Compute the graph Laplacian , where  is the degree matrix.
* **Fiedler Vector:** Compute the eigenvector  corresponding to the second smallest eigenvalue of .
* **Sort:** Permute the columns of  (and rows of ) according to the indices that sort .
* *Result:* Functionally similar neurons are now indices  and .



### Phase 2: Signal Resampling (Interpolation)

Treat the sorted weights as a signal sampled at  points and resample to .

* **Operation:** Use Linear Interpolation.


* **Why `align_corners=True`?** This ensures the first and last neurons (the "boundary conditions" of our manifold) are preserved exactly, mapping the interval  to  without phase shift.
* **Handling Outgoing Weights:** The corresponding rows in the next layer's weight matrix () must also be resampled similarly to align inputs.

### Phase 3: Energy Preservation (Scaling)

Because we have doubled the number of inputs to the next layer ( neurons), the pre-activation sum will double if weights are unchanged.

* **Adjustment:** Scale the *outgoing* weights () by a factor of .

---

## 4. Experimental Design

### 4.1 Architecture & Task

* **Task:** MNIST Classification (Digit recognition).
* **Base Model (Small):** MLP.
* Input: 784 (28x28 flattened)
* Hidden 1: **128 units** (ReLU)
* Hidden 2: **128 units** (ReLU)
* Output: 10


* **Target Model (Large):** MLP.
* Hidden 1: **256 units**
* Hidden 2: **256 units**



### 4.2 The Baselines

We will run three distinct training protocols. All models use the same optimizer (AdamW), Learning Rate, and Batch Size.

**Baseline A: Train Large from Scratch**

1. Initialize MLP with width 256.
2. Train for  epochs.
3. Log validation accuracy/loss.

**Baseline B: Net2Net (Standard Expansion)**

1. Train MLP (width 128) for  epochs.
2. **Apply Net2Net:**
* Randomly select 128 indices to duplicate.
* Add small noise to break symmetry ().
* Scale outgoing weights by 0.5.


3. Train MLP (width 256) for remaining  epochs.

**Experiment C: Continuous Resampling (Ours)**

1. Train MLP (width 128) for  epochs.
2. **Apply Continuous Resampling:**
* Sort Hidden 1 and Hidden 2 via Fiedler vector.
* Resample weights to width 256 using `linear` interpolation.
* Scale outgoing weights by 0.5.


3. Train MLP (width 256) for remaining  epochs.

---

## 5. Implementation Details

### 5.1 Environment

* **Container:** Pytorch environment using uv for dependency management.
* **Compute:** 1x NVIDIA H100 (Note: This is overkill for MNIST, but validates the environment for future LLM scaling).

### 5.2 TensorBoard Metrics

We need high-resolution logging around the expansion event.

1. **`Loss/Train` & `Loss/Val**`: Logged every step.
2. **`Accuracy/Val`**: Logged every epoch.
3. **`Metric/Expansion_Shock`**: specifically measure the immediate loss increase at step .
4. **`Weight/Spectrum`**: (Optional) Visualize the cosine similarity matrix of the weights before and after sorting to verify the "diagonalization" of the matrix.

### 5.3 Code Structure (Python/PyTorch)

* `models.py`: Defines the MLP. Needs to handle arbitrary width.
* `ops.py`:
* `def spectral_sort(weight_matrix):` Returns permutation indices.
* `def resample_layer(weight_matrix, new_size):` Implements `torch.nn.functional.interpolate`.
* `def expand_model(model, method='continuous'):` The core logic switching between Net2Net and Our Method.


* `train.py`: Standard loop.

### 5.4 Logging

In addition to TensorBoard we should have some basic logging to the console. This should include a progress bar using tqdm that also shows loss and printing out metrics at the end of each epoch. Especially important are metrics after scale up.

---

## 6. Success Metrics

The experiment is a success if **Experiment C** demonstrates:

1. **Lower "Shock":** The jump in validation loss immediately after expansion is lower than Baseline B.
2. **Faster Recovery:** The number of steps required to return to pre-expansion accuracy is lower than Baseline B.
3. **Final Performance:** The final accuracy at epoch  is  Baseline A (From Scratch) and  Baseline B.

---

## 7. Clarifications & Open Questions

* **When to expand:** One interesting question is when we should expand the network. We should start by doing it at a fixed interval, but we'd also like to try doing it when loss reaches a certain threshold. This allows us to exploit the advantage of the "ant's eye view" and expand when we are in a "valley" of the loss landscape, which should make the transition smoother.
* **Layer Depth:** This spec assumes we only scale width. Scaling depth (inserting layers) is a separate algebraic problem (Identity initialization) and is out of scope for this specific test.
* **Activation Functions:** We assume ReLU. If using GeLU or Swish, the scaling factor () generally remains correct for linear regimes, but variance analysis might suggest slight tweaks. We will stick to  for simplicity.
* **Optimizer State:** When expanding, we handle the Adam optimizer state (momentum  and variance ) by resampling and exactly using the same interpolation method as the weights. This allows the "momentum" of the "ant" to be preserved across the manifold transformation.

Read this spec thoroughly, ask for any clarifications and use your expertise to push back on anything that doesn't make sense.

Once we agree on the design we'll move on to implementation. I have already setup an empty uv environment with PyTorch, TensorBoard and datasets dependencies installed. Install anything else you need.
