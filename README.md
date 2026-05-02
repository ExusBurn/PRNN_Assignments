# Pattern Recognition and Neural Networks — PRNN 2026
### IISc Bangalore | Department of Electrical Communication Engineering

Solutions to all three programming assignments for the PRNN course (2026), covering classical machine learning, deep learning with PyTorch, and generative/ensemble/RL methods. All core algorithms are implemented from scratch using NumPy, SciPy, and PyTorch — no high-level wrappers like scikit-learn ensembles, PyTorch Lightning, or pre-trained models (except where explicitly authorized).

---

## Repository Structure

```
.
├── Assignment 1/          # Classical ML from scratch (numpy/scipy)
│   ├── q1.ipynb           # OLS, Gradient Descent, Hard-margin SVM
│   ├── q2.ipynb           # Ridge regression, Lasso via Coordinate Descent
│   ├── q3.ipynb           # Logistic Regression, GMM, Soft-margin SVM, RBF Kernel
│   ├── q4.ipynb           # OvR Logistic Regression, KNN, Naive Bayes
│   └── q5.ipynb           # Bias-Variance decomposition, Bayesian MAP
│
├── Assignment 2/          # Deep Learning with PyTorch
│   └── AllCodes.ipynb     # MLP, CNN, RNN, LSTM, GRU, Transformer, ViT, Transfer Learning
│
└── Assignment 3/          # Ensembles, Generative Models, RL
    ├── Phase1_2.ipynb     # Decision Trees, Random Forest, AdaBoost, Gradient Boosting, PCA, GMM
    └── Phase3_4_5.ipynb   # VAE, DCGAN, FID, SimCLR, REINFORCE
```

---

## Assignment 1 — Classical Machine Learning from Scratch

**Datasets:** Synthetic bio-optic sensor data, oncology genomic arrays, ICU telemetry, and emergency room EHR records.

### Phase 1: Bio-Optic Sensor Calibration (`q1.ipynb`)
- **Q1.1 — OLS: Closed-form vs Gradient Descent:** Implemented Ordinary Least Squares via both the normal equations (X^T X)^{-1} X^T y and vectorized batch gradient descent. Reported the L2 norm of the difference between the two weight vectors, confirming both converge to the same solution.
- **Q1.2 — Hard-Margin SVM Dual:** Converted the regression target to binary classification by thresholding at the median. Formulated the SVM dual problem and solved for Lagrange multipliers using `scipy.optimize.minimize`. Extracted exact support vector indices from the training set.

### Phase 2: Oncology Genomic Array (`q2.ipynb`)
- **Q2.3 — The Singularity Trap:** Attempted closed-form OLS on a fat matrix (N=500, D=1000) and documented the exact LinAlgError from the singular X^T X. Implemented L2 (Ridge) regression to rescue the inversion. Plotted the condition number of (X^T X + λI) over a logarithmic grid from 10⁻⁵ to 10².
- **Q2.4 — Lasso via Coordinate Descent:** Implemented L1 regularization using Coordinate Descent from scratch. Generated a full regularization path plot (weight values vs. log λ). Programmatically identified the exact λ at which exactly 50% of feature weights are zeroed out.

### Phase 3: ICU Telemetry (`q3.ipynb`)
- **Q3.5 — Logistic Regression with L2:** Binary logistic regression (Class 1 vs Class 2) with L2 regularization via gradient descent. Plotted learning curves and demonstrated empirically how the loss diverges when the learning rate violates the Lipschitz smoothness condition.
- **Q3.6 — GMM E-Step & M-Step:** Implemented the EM algorithm for a Gaussian Mixture Model with K=4. Reported the initial marginal log-likelihood and the exact log-likelihood after one E-step and one M-step.
- **Q3.7 — EM Convergence & Crashes:** Ran GMM for K=2,4,6 and plotted log-likelihood over iterations to verify monotonic convergence. Forced a covariance matrix to zero variance mid-training and documented the exact Python error breaking the Multivariate Normal PDF.
- **Q3.8 — Bayes Optimal Decision Boundaries:** Trained a GMM (K=3) on the first two dimensions of Dataset 3. Plotted a dense decision boundary grid overlaid with Gaussian covariance contours on the scatter plot.
- **Q3.9 — Soft-Margin SVM KKT Verification:** Implemented the soft-margin SVM dual with C=1.0. Programmatically verified KKT conditions and printed the index and µ value for: a point strictly inside the margin, a point on the margin boundary, and a point safely outside.
- **Q3.10 — Mercer's Condition:** Implemented the RBF (Gaussian) kernel. Computed the full N×N kernel matrix for a 1000-sample subset. Verified positive semi-definiteness by computing all eigenvalues and showing none are negative.
- **Q3.11 — Hyperparameter Topography:** Grid search over C ∈ {0.1, 1, 10, 100} and σ ∈ {0.001, 0.01, 0.1, 1, 10}. Generated a 2D heatmap of validation accuracy and identified the (C, σ) coordinate of maximum overfitting.

### Phase 4: EHR Records (`q4.ipynb`)
- **Q4.12 — One-vs-Rest Logistic Regression + StandardScaler:** Extended logistic regression to multiclass via OvR. Implemented StandardScaler from scratch (zero-mean, unit-variance). Compared the number of iterations to convergence on raw vs. scaled features — scaling reduced average iterations from 1000 to 316.
- **Q4.13 — KNN Vectorization Benchmark:** Implemented KNN twice — once with nested Python for loops and once using fully vectorized matrix operations (||a-b||² = ||a||² + ||b||² - 2aᵀb). Timed both on a 2000-sample test split. Vectorized implementation achieved a **94.41× speedup**.
- **Q4.14 — Curse of Scale:** Evaluated vectorized KNN (K=5) on raw vs. standardized data. Standardization improved accuracy from 0.9045 to 0.9210, a difference of **0.0165**.
- **Q4.15 — Naive Bayes & Underflow:** Implemented Gaussian Naive Bayes with log-probabilities to prevent underflow (accuracy: 91.33%). Separately multiplied raw probabilities and found the exact feature dimension D at which the joint probability underflows to exactly 0.0 in Python.

### Phase 5: Bias-Variance Decomposition (`q5.ipynb`)
- **Q5.16 — Empirical Bias-Variance via Bootstrapping:** Implemented polynomial feature extraction. Trained d=1 and d=15 polynomial regressors across B=100 bootstrap samples. Computed and reported empirical squared Bias and Variance for both degrees on a held-out test set — demonstrating the classic underfitting vs. overfitting tradeoff.
- **Q5.17 — Frequentist vs. Bayesian Uncertainty:** Computed the frequentist variance of the slope parameter across 100 bootstrapped d=1 models. Implemented a Bayesian MAP estimator with a Gaussian prior and computed the analytical posterior variance. Plotted both distributions and documented the fundamental difference in what "expectation" means in each paradigm.

---

## Assignment 2 — Deep Learning with PyTorch (`AllCodes.ipynb`)

**Datasets:** Delhi Air Quality Time-Series (Kaggle), PlantVillage Crop Disease (Kaggle), APTOS 2019 Blindness Detection (Kaggle). All splits: 70% train / 15% val / 15% test. Air quality split is strictly chronological to prevent temporal leakage.

### Phase 1: MLP & Gradient Flow
- **Q1.1 — Temporal MLP Regression:** Flattened 72-hour meteorological windows into a 3-layer MLP to predict t+24 PM2.5. Reported test MSE.
- **Q1.2 — Vanishing Gradient Proof:** Replaced ReLU with Sigmoid and attached PyTorch backward hooks to capture gradient L2 norms at each layer during the first epoch. Plotted the decay and explained it mathematically via the chain rule and saturating sigmoid derivatives.
- **Q1.3 — Imbalanced Classification:** Converted target to binary (PM2.5 > 200). Compared standard BCELoss against BCEWithLogitsLoss with pos_weight scaled to the class imbalance ratio. Overlaid Precision-Recall curves for both.

### Phase 2: CNNs & Spatial Hierarchies
- **Q2.4 — CNN from Scratch:** Built a CNN using nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d. Programmatically printed tensor shape at every layer for a 128×128 input and calculated the theoretical receptive field of the final feature map.
- **Q2.5 — CNN Regression Head:** Modified the CNN to regress a continuous severity percentage (synthesized from necrotic tissue bounding box ratio). Reported MAE on the test set.
- **Q2.6 — Translation Invariance:** Evaluated the CNN on 100 test images, then re-evaluated after shifting all pixels 5px right and 5px down. Documented the accuracy drop and explained why max-pooling does not provide exact mathematical translation invariance.

### Phase 3: RNNs & Sequential Memory
- **Q3.7 — Vanilla RNN from Scratch:** Implemented a full RNN cell using only nn.Linear and tanh inside a Python for loop — no nn.RNN. Trained on 72-hour sequences for PM2.5 regression.
- **Q3.8 — BPTT Decay:** Passed a 100-step sequence through the untrained RNN, called .backward(), and extracted gradient magnitudes at t=100, t=50, t=0. Plotted the exponential decay empirically demonstrating vanishing gradients.
- **Q3.9 — Exploding Gradients:** Increased learning rate and removed gradient clipping until loss hit NaN. Extracted weight matrices one step before the crash and computed their maximum singular value to mathematically explain the explosion.

### Phase 4: LSTMs, GRUs & Context
- **Q4.10 — LSTM Gradient Rescue:** Replaced vanilla RNN with nn.LSTM and repeated the t=100,50,0 gradient extraction experiment. Plotted alongside RNN gradients and explained how the cell state's additive gradient pathway prevents vanishing.
- **Q4.11 — GRU vs. LSTM Efficiency:** Counted exact trainable parameters for both architectures by iterating state_dict(). Compared final validation MSE and training time per epoch.
- **Q4.12 — Seq2Seq Forecasting:** Built an LSTM Encoder-Decoder that encodes 72-hour history into a context vector and autoregressively decodes the full 24-hour PM2.5 trajectory. Plotted ground truth vs. predicted sequences on test samples.

### Phase 5: Transformers & Attention
- **Q5.13 — Scaled Dot-Product Attention:** Implemented a single attention head from scratch using raw tensor operations. Extracted and plotted the 72×72 attention weight heatmap for a specific test sample.
- **Q5.14 — Positional Encoding Necessity:** Trained a Transformer Encoder without positional encodings and reported the validation loss. Added sinusoidal encodings and retrained. Explained mathematically why the permutation-invariant attention mechanism fundamentally cannot capture temporal order without positional information.
- **Q5.15 — Vision Transformer (ViT):** Sliced 128×128 images into 16×16 non-overlapping patches, flattened each into a 1D sequence, and passed through a Transformer Encoder for disease classification. Compared parameter count vs. the CNN from Phase 2.

### Phase 6: Advanced Vision & Imbalance
- **Q6.16 — Transfer Learning & Freezing:** Loaded pretrained ResNet18, froze all convolutional layers, replaced the FC head with a 5-class output. Reported exact trainable vs. frozen parameter counts and plotted the validation loss curve.
- **Q6.17 — Focal Loss & Ordinal Imbalance:** Implemented Focal Loss from scratch to handle severe class imbalance on APTOS (skewed toward Class 0). Evaluated with Cohen's Quadratic Weighted Kappa (QWK) and explained why cross-entropy accuracy is a misleading metric for ordinal imbalanced data.

---

## Assignment 3 — Ensembles, Generative Models & RL

**Datasets:** Same splits as Assignment 2 — Delhi Air Quality, PlantVillage, APTOS 2019.

### Phase 1: Tree Ensembles & Boosting (`Phase1_2.ipynb`)
- **Q1.1 — Gini Impurity from Scratch:** Wrote a function to compute Gini impurity reduction for every continuous threshold on the Temperature feature. Reported the exact optimal split value maximizing Gini gain at the root node.
- **Q1.2 — Random Forest:** Implemented a Random Forest classifier with depth-limited trees and feature bagging (random subspaces) — no sklearn.
- **Q1.3 — AdaBoost Reweighting:** Implemented AdaBoost.M1 with shallow trees. Tracked the sample weights of the 5 most consistently misclassified training samples across 50 iterations. Plotted their exponential weight growth and mathematically explained the update rule.
- **Q1.4 — Gradient Boosting Residuals:** Implemented a Gradient Boosted Regressor by sequentially fitting shallow trees to pseudo-residuals. Plotted the variance of training residuals after 1, 5, 10, and 50 boosting stages.

### Phase 2: Subspace & Clustering (`Phase1_2.ipynb`)
- **Q2.5 — PCA vs. Linear Autoencoder:** Implemented PCA via SVD and computed the number of components for 95% variance retention. Trained a purely linear Autoencoder with matching bottleneck dimension. Proved empirically via reconstruction MSE that both span the same subspace.
- **Q2.6 — GMM Latent Clustering:** Extracted dense feature embeddings from the Assignment 2 CNN (head stripped). Fit a GMM and computed the Adjusted Rand Index (ARI) between GMM clusters and true disease labels.

### Phase 3: Deep Generative Models (`Phase3_4_5.ipynb`)
- **Q3.7 — VAE ELBO:** Built a Convolutional VAE for PlantVillage. Plotted reconstruction loss and KL regularization term separately over training epochs.
- **Q3.8 — VAE vs. DCGAN:** Trained a DCGAN on PlantVillage alongside the VAE. Generated 4×4 grids of synthetic leaf images from both and compared visual sharpness. Explained mathematically why the ELBO objective induces blurry outputs while the GAN minimax objective does not.
- **Q3.9 — FID Score:** Filtered APTOS to Class 4 images and trained a DCGAN. Passed 500 real and 500 generated images through frozen Inception V3, extracted 2048-dimensional features, and implemented the FID formula from scratch using SciPy matrix operations.
- **Q3.10 — Batch Norm & Mode Collapse:** Removed all BatchNorm layers from the APTOS generator and retrained. Provided visual output grids and explained mathematically how the absence of batch statistics exacerbates mode collapse.

### Phase 4: Self-Supervised Learning — SimCLR (`Phase3_4_5.ipynb`)
- **Q4.11 — InfoNCE Loss:** Implemented the contrastive augmentation pipeline (random crops, flips, color jitter) and the Normalized Temperature-scaled InfoNCE loss from scratch. Trained a ResNet18 encoder with an MLP projection head on PlantVillage without any labels.
- **Q4.12 — Linear Evaluation Protocol:** Froze the SimCLR encoder and attached a single nn.Linear head. Trained only the head with true labels. Compared self-supervised vs. fully supervised CNN accuracy from Assignment 2.

### Phase 5: Policy Gradient / RL (`Phase3_4_5.ipynb`)
- **Q5.13 — REINFORCE:** Formulated an Air Quality MDP (state: 24-hour meteorological history; action: purifier ON/OFF; reward: energy cost + health penalty). Implemented the REINFORCE algorithm from scratch using the log-derivative trick. Simulated 500 training episodes and plotted total episodic return over time.

---

## Key Learnings

- **From-scratch implementations reveal what libraries hide.** Implementing Coordinate Descent for Lasso, the SVM dual, shared-memory reductions, and the EM algorithm by hand forces a genuine understanding of convergence conditions, numerical edge cases, and the mathematics underneath.
- **Numerical stability matters in practice.** The Naive Bayes underflow experiment, the singularity in fat OLS matrices, and the NaN loss in exploding-gradient RNNs are all real failure modes — log-probability tricks, regularization, and gradient clipping exist for concrete reasons.
- **Vectorization is transformative.** The 94× KNN speedup (nested loops vs. matrix operations) and the statevector simulation work in PP are both reminders that the gap between a correct implementation and an efficient one is enormous.
- **Real datasets are messy.** Working across bio-sensor calibration, oncology microarrays, ICU telemetry, retinal fundus images, and air quality time-series meant confronting class imbalance, missing data, unscaled features, temporal leakage, and extreme multicollinearity — not just clean toy problems.
- **Evaluation metrics must match the problem.** Cross-entropy accuracy is deeply misleading for ordinal imbalanced data like APTOS — QWK and precision-recall curves tell a very different story.
- **Classical and deep methods are complementary.** Bootstrap bias-variance decomposition, GMM clustering on CNN embeddings, and the PCA-autoencoder equivalence proof all sit at the intersection of classical and modern ML.

---

## Dependencies

**Assignment 1:** `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`

**Assignment 2:** `torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, `PIL`

**Assignment 3:** `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `pandas`

---

## Course

**Pattern Recognition and Neural Networks (PRNN), 2026**
Instructor: Prathosh A P
Department of Electrical Communication Engineering, IISc Bangalore
