# Cross-Domain Hyperspectral Image Classification Using Semantic Tokenization, Mixture of Experts, Bidirectional Coupled Transformers, and HMA Teacher Distillation

## 1. Overview

This document describes a proposed architecture for **unsupervised cross-domain hyperspectral image classification**.

The method is inspired by the bidirectional domain adaptation concept used in the reference paper, but introduces several substantial architectural modifications:

1. A **3D/2D CNN semantic tokenizer**.
2. Exactly **5 semantic tokens per hyperspectral patch**.
3. Independent source and target **Transformer encoders**.
4. **Mixture-of-Experts (MoE)** layers replacing the standard Transformer FFN.
5. **Top-2 expert routing**.
6. A **bidirectional coupled Transformer** based on cross-attention.
7. Source queries target and target queries source simultaneously.
8. **Bidirectional teacher-student representation distillation**.
9. A teacher representation updated using a **Hull Moving Average (HMA)** strategy.
10. MoE load-balancing regularization.

The overall objective is to learn representations that preserve useful domain-specific information while allowing explicit bidirectional information exchange between source and target domains.

---

# 2. Problem Definition

We consider an unsupervised domain adaptation problem.

We have a labeled source domain:

$$
\mathcal{D}_s =
\{(X_i^s,y_i^s)\}_{i=1}^{N_s}
$$

where:

* \(X_i^s\) is a source hyperspectral patch.
* \(y_i^s\) is its class label.
* \(N_s\) is the number of source samples.

We also have an unlabeled target domain:

$$
\mathcal{D}_t =
\{X_i^t\}_{i=1}^{N_t}
$$

where:

* \(X_i^t\) is a target hyperspectral patch.
* \(N_t\) is the number of target samples.

The objective is to train a model using:

* labeled source samples;
* unlabeled target samples;

and then correctly classify target-domain samples.

---

# 3. Input Data Format

Each input sample is a hyperspectral patch:

$$
X \in \mathbb{R}^{13\times13\times13}
$$

where:

* height = 13;
* width = 13;
* spectral channels = 13.

The central pixel of the patch is the pixel being classified.

During training, the model receives:

$$
X_s \in \mathbb{R}^{B\times13\times13\times13}
$$

and:

$$
X_t \in \mathbb{R}^{B\times13\times13\times13}
$$

where \(B\) is the batch size.

For PyTorch 3D convolution, the recommended internal arrangement is:

$$
[B,C,D,H,W]
$$

Therefore, the input can be rearranged into:

```text
[B, 1, Spectral, Height, Width]
```

For this model:

```text
[B, 1, 13, 13, 13]
```

---

# 4. High-Level Architecture

The model contains four major components:

1. Semantic tokenizer.
2. Independent domain Transformer encoders.
3. Bidirectional coupled Transformer.
4. Classification and adaptation losses.

The high-level pipeline is:

```text
                    SOURCE DOMAIN
                         │
                         ▼
                 Semantic Tokenizer
                         │
                         ▼
                  5 Semantic Tokens
                         │
                  Positional Encoding
                         │
                         ▼
                Source Transformer
                     + Top-2 MoE
                         │
                         ▼
              Source Representation ───────► Classifier
                         │                       │
                         │                       ▼
                         │               Classification Loss
                         │
                         ▼
                    HMA Teacher
                         │
                    Stop Gradient
                         │
                         ▼
                  Distillation Loss
                         ▲
                         │
              Source Queries Target
                         │
                         ▼
                Coupled Transformer
                     + Top-2 MoE
                         ▲
                         │
              Target Queries Source
                         │
                         ▼
                  Distillation Loss
                         ▲
                         │
                    Stop Gradient
                         │
                    HMA Teacher
                         ▲
                         │
              Target Representation
                         ▲
                         │
                Target Transformer
                     + Top-2 MoE
                         ▲
                         │
                  Positional Encoding
                         ▲
                         │
                  5 Semantic Tokens
                         ▲
                         │
                 Semantic Tokenizer
                         ▲
                         │
                    TARGET DOMAIN
```

---

# 5. Step 1 — Semantic Tokenizer

## 5.1 Objective

The semantic tokenizer converts the raw hyperspectral patch:

$$
X \in \mathbb{R}^{13\times13\times13}
$$

into:

$$
Z \in \mathbb{R}^{5\times D}
$$

where:

* 5 is the number of semantic tokens;
* \(D\) is the token embedding dimension.

The tokenizer should extract:

* local spatial information;
* spectral information;
* spectral-spatial relationships.

---

## 5.2 First 3D Convolution

The first layer is:

$$
\text{Conv3D}(2\times2\times2)
$$

followed by:

$$
\text{BatchNorm}
$$

and:

$$
\text{LeakyReLU}.
$$

The operation is:

$$
F_1=
\text{LeakyReLU}
(
\text{BN}
(
\text{Conv3D}_1(X)
)
).
$$

Recommended conceptual implementation:

```text
Input
[B, 1, 13, 13, 13]

↓

Conv3D
kernel = (2, 2, 2)

↓

BatchNorm3D

↓

LeakyReLU
```

This layer begins extracting local spectral-spatial features.

---

## 5.3 Second 3D Convolution

The second layer repeats the same structure:

$$
F_2=
\text{LeakyReLU}
(
\text{BN}
(
\text{Conv3D}_2(F_1)
)
).
$$

Pipeline:

```text
Conv3D (2×2×2)
↓
BatchNorm3D
↓
LeakyReLU
```

This layer further combines local spatial and spectral information.

---

## 5.4 Transition from 3D to 2D

After the second 3D convolution, the representation still contains a spectral/depth dimension.

Before applying Conv2D, convert the spectral feature dimension into channels.

For example:

```text
[B, C, Spectral, H, W]
```

can be reshaped into:

```text
[B, C × Spectral, H, W]
```

Then apply:

$$
\text{Conv2D}(2\times2).
$$

The operation is:

$$
F_3=
\text{LeakyReLU}
(
\text{BN}
(
\text{Conv2D}(F_2)
)
).
$$

The purpose of this stage is to convert the extracted spectral-spatial representation into a compact spatial semantic representation.

---

# 6. Step 2 — Generate Five Semantic Tokens

The tokenizer must output exactly:

$$
Z\in\mathbb{R}^{5\times D}.
$$

The recommended approach is learned attention-based semantic token pooling.

Assume the CNN produces:

$$
F\in\mathbb{R}^{C\times H'\times W'}.
$$

Flatten the spatial dimensions:

$$
F\rightarrow
F'\in\mathbb{R}^{N\times C}
$$

where:

$$
N=H'W'.
$$

Next, calculate five learned attention maps:

$$
A=
\text{Softmax}
(
F'W_A
)
$$

where:

$$
A\in\mathbb{R}^{N\times5}.
$$

Each of the five attention maps determines which spatial features belong to one semantic token.

The token embeddings are then:

$$
Z=A^TF'.
$$

Therefore:

$$
\boxed{
Z\in\mathbb{R}^{5\times D}
}
$$

Each token can learn to represent a different semantic concept or spectral-spatial pattern.

Examples may include:

* vegetation-like structures;
* road-like structures;
* building structures;
* background material;
* mixed or boundary regions.

These meanings should not be manually assigned. They should emerge during training.

---

# 7. Step 3 — Positional Encoding

The five semantic tokens receive positional embeddings:

$$
\tilde Z=Z+P.
$$

Where:

$$
P\in\mathbb{R}^{5\times D}.
$$

Recommended implementation:

```text
nn.Parameter(torch.randn(1, 5, D))
```

The same positional encoding structure can be used for both source and target branches.

---

# 8. Step 4 — Source Transformer Encoder

The source branch processes:

$$
\tilde Z_s=Z_s+P.
$$

The source Transformer produces:

$$
H_s=E_s(\tilde Z_s).
$$

The Transformer consists of multiple identical blocks.

Each block contains:

```text
Input
  │
  ▼
Multi-Head Self Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Mixture of Experts
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Output
```

Mathematically:

$$
H'=
\text{LN}
(
H+\text{MHSA}(H)
)
$$

and:

$$
H_{out}
=
\text{LN}
(
H'+\text{MoE}(H')
).
$$

---

# 9. Step 5 — Target Transformer Encoder

The target branch has the same conceptual structure:

$$
H_t=E_t(Z_t+P).
$$

A key implementation decision is whether the source and target encoders share weights.

Recommended initial design:

```text
Source Transformer: independent parameters

Target Transformer: independent parameters
```

This allows each domain to maintain domain-specific feature extraction.

Later experiments can compare:

1. fully independent encoders;
2. fully shared encoders;
3. partially shared encoders.

The independent configuration should be used as the main initial model.

---

# 10. Step 6 — Mixture of Experts

The standard Transformer FFN is replaced by an MoE layer.

Assume \(K\) experts:

$$
E_1,E_2,\ldots,E_K.
$$

Each expert is a small neural network:

$$
E_k(x)
=
W_{k,2}
\sigma
(
W_{k,1}x+b_{k,1}
)
+b_{k,2}.
$$

---

# 11. Step 7 — Top-2 Routing

A router produces expert probabilities:

$$
g(x)
=
\text{Softmax}
(
W_gx
).
$$

The model selects the two largest probabilities.

If experts \(i\) and \(j\) are selected:

$$
\text{MoE}(x)
=
\hat g_iE_i(x)
+
\hat g_jE_j(x).
$$

The selected probabilities are renormalized:

$$
\hat g_i=
\frac{g_i}{g_i+g_j}
$$

and:

$$
\hat g_j=
\frac{g_j}{g_i+g_j}.
$$

Recommended initial experiment:

```text
Number of experts: 4
Routing: Top-2
Expert hidden dimension: 2D to 4D
```

Because hyperspectral datasets can be relatively small, using too many experts may cause overfitting.

---

# 12. Step 8 — MoE Load Balancing

MoE routing can collapse if most tokens select the same expert.

Let:

$$
p_k
$$

be the fraction of tokens assigned to expert \(k\).

A simple balancing loss is:

$$
\mathcal{L}_{balance}
=
\sum_{k=1}^{K}
\left(
p_k-\frac{1}{K}
\right)^2.
$$

This encourages all experts to receive training samples.

The exact balancing strategy can later be replaced with a standard sparse-MoE auxiliary loss.

---

# 13. Step 9 — Source Representation

The source Transformer output is:

$$
H_s\in\mathbb{R}^{5\times D}.
$$

A single patch representation must be generated.

Recommended approach:

$$
h_s=
\frac{1}{5}
\sum_{i=1}^{5}
H_{s,i}.
$$

Therefore:

$$
h_s\in\mathbb{R}^{D}.
$$

Alternative experiments can use:

* attention pooling;
* learnable classification token;
* maximum pooling.

For the first implementation, mean pooling is recommended because it is simple and stable.

---

# 14. Step 10 — Source Classification

The source representation is classified:

$$
\hat y_s=
C(h_s).
$$

The classifier can be:

```text
Linear(D → D/2)
↓
LeakyReLU
↓
Dropout
↓
Linear(D/2 → Number of Classes)
```

The source classification loss is:

$$
\mathcal{L}_{cls}
=
-\frac{1}{N_s}
\sum_{i=1}^{N_s}
\sum_{c=1}^{C}
y_{i,c}
\log
p_{i,c}.
$$

Only source labels are used.

---

# 15. Step 11 — Coupled Transformer

The coupled Transformer receives both source and target tokens.

Inputs:

$$
H_s
$$

and:

$$
H_t.
$$

It performs bidirectional cross-domain attention.

There are two directions:

1. Source queries target.
2. Target queries source.

---

# 16. Step 12 — Source Queries Target

Source queries are:

$$
Q_s=H_sW_Q^s.
$$

Target keys and values are:

$$
K_t=H_tW_K^t
$$

and:

$$
V_t=H_tW_V^t.
$$

Cross-attention is:

$$
A_{s\leftarrow t}
=
\text{Attention}
(
Q_s,
K_t,
V_t
).
$$

This means:

> Source-domain tokens query the target-domain tokens and retrieve target-domain information.

The residual connection should use the source representation:

$$
C'_{s\leftarrow t}
=
\text{LN}
(
H_s+A_{s\leftarrow t}
).
$$

Then apply the Top-2 MoE:

$$
C_{s\leftarrow t}
=
\text{LN}
(
C'_{s\leftarrow t}
+
\text{MoE}(C'_{s\leftarrow t})
).
$$

The final result is:

$$
\boxed{
C_{s\leftarrow t}
}
$$

which is:

> A source representation enriched using target-domain information.

---

# 17. Step 13 — Target Queries Source

Similarly:

$$
Q_t=H_tW_Q^t.
$$

Source keys and values are:

$$
K_s=H_sW_K^s
$$

and:

$$
V_s=H_sW_V^s.
$$

The cross-attention output is:

$$
A_{t\leftarrow s}
=
\text{Attention}
(
Q_t,
K_s,
V_s
).
$$

Residual connection:

$$
C'_{t\leftarrow s}
=
\text{LN}
(
H_t+A_{t\leftarrow s}
).
$$

MoE:

$$
C_{t\leftarrow s}
=
\text{LN}
(
C'_{t\leftarrow s}
+
\text{MoE}(C'_{t\leftarrow s})
).
$$

The result is:

$$
\boxed{
C_{t\leftarrow s}
}
$$

which is:

> A target representation enriched using source-domain information.

---

# 18. Step 14 — Teacher-Student Distillation

The independent branches act as teachers.

The coupled branch acts as the student.

The teacher does not receive gradients from the distillation loss.

---

## Source Teacher

The source teacher representation is:

$$
h_s^{teacher}.
$$

The student representation is:

$$
h_{s\leftarrow t}^{student}.
$$

The distillation loss is:

$$
\mathcal{L}_{s\leftarrow t}
=
D
(
\text{StopGrad}
(
h_s^{teacher}
),
h_{s\leftarrow t}^{student}
).
$$

---

## Target Teacher

The target teacher representation is:

$$
h_t^{teacher}.
$$

The student representation is:

$$
h_{t\leftarrow s}^{student}.
$$

The loss is:

$$
\mathcal{L}_{t\leftarrow s}
=
D
(
\text{StopGrad}
(
h_t^{teacher}
),
h_{t\leftarrow s}^{student}
).
$$

---

# 19. Step 15 — Representation Distance

Recommended initial implementation:

$$
D(a,b)
=
1-
\frac{
a\cdot b
}{
||a||||b||
}.
$$

This is cosine distance.

Therefore:

$$
\mathcal{L}_{s\leftarrow t}
=
1-\cos
(
h_s^{teacher},
h_{s\leftarrow t}^{student}
).
$$

And:

$$
\mathcal{L}_{t\leftarrow s}
=
1-\cos
(
h_t^{teacher},
h_{t\leftarrow s}^{student}
).
$$

The total distillation loss is:

$$
\boxed{
\mathcal{L}_{dist}
=
\mathcal{L}_{s\leftarrow t}
+
\mathcal{L}_{t\leftarrow s}
}
$$

---

# 20. Step 16 — Hull Moving Average Teacher

This component must be implemented carefully.

The teacher is not updated directly using gradients from the distillation loss.

Instead, the teacher parameters are calculated using a temporal smoothing process based on a Hull Moving Average.

For a parameter history:

$$
\theta_{t-L+1},
\dots,
\theta_t,
$$

the HMA can conceptually be expressed as:

$$
\text{HMA}(n)
=
\text{WMA}
\left(
2\cdot
\text{WMA}
\left(
\frac{n}{2}
\right)
-
\text{WMA}(n),
\sqrt{n}
\right).
$$

The teacher parameters are:

$$
\theta_{teacher}^{(t)}
=
\text{HMA}
(
\theta^{(t-L+1)},
\dots,
\theta^{(t)}
).
$$

---

## Recommended Implementation Strategy

Maintain a parameter history:

```text
Student parameter history:

θ(t-L+1)
θ(t-L+2)
...
θ(t)
```

After each training update:

1. Store the current student encoder parameters.
2. Remove parameters older than the HMA window.
3. Compute the HMA-smoothed parameters.
4. Update the teacher encoder using the HMA result.
5. Use the teacher only for forward inference.
6. Disable gradients for the teacher.

Important:

```text
Teacher:
requires_grad = False

Teacher output:
detach()
```

---

# 21. Step 17 — Distillation Warm-Up

The teacher and student representations may be unstable during the first training iterations.

Therefore, do not apply full distillation strength immediately.

Define:

$$
\lambda_{dist}(e)
=
\lambda_{max}
\min
\left(
1,
\frac{e}{E_{warmup}}
\right).
$$

Where:

* \(e\) is the current epoch;
* \(E_{warmup}\) is the warm-up duration.

Example:

```text
Epoch 0:
λ_dist = 0

Epoch 10:
λ_dist = 0.5 λ_max

Epoch 20:
λ_dist = λ_max
```

The exact schedule should be selected experimentally.

---

# 22. Step 18 — Complete Training Loss

The complete loss is:

$$
\boxed{
\mathcal{L}_{total}
=
\mathcal{L}_{cls}
+
\lambda_{dist}
\mathcal{L}_{dist}
+
\lambda_{balance}
\mathcal{L}_{balance}
}
$$

where:

### Classification loss

$$
\mathcal{L}_{cls}
=
CE
(
C(h_s),
y_s
).
$$

### Bidirectional distillation

$$
\mathcal{L}_{dist}
=
\mathcal{L}_{s\leftarrow t}
+
\mathcal{L}_{t\leftarrow s}.
$$

### MoE balancing

$$
\mathcal{L}_{balance}
=
\sum_k
\left(
p_k-\frac{1}{K}
\right)^2.
$$

---

# 23. Step 19 — Training Algorithm

## Initialization

Initialize:

```text
Source Tokenizer
Target Tokenizer

Source Transformer
Target Transformer

Coupled Transformer

MoE Experts

Source Classifier

Source Teacher
Target Teacher
```

Initially:

```text
Teacher parameters = corresponding student parameters
```

---

## Training Loop

For every training iteration:

### Step 1

Sample:

```text
Source batch:
(X_s, y_s)

Target batch:
X_t
```

---

### Step 2

Generate source semantic tokens:

$$
Z_s=
Tokenizer_s(X_s).
$$

---

### Step 3

Generate target semantic tokens:

$$
Z_t=
Tokenizer_t(X_t).
$$

---

### Step 4

Add positional encoding:

$$
\tilde Z_s=Z_s+P
$$

$$
\tilde Z_t=Z_t+P.
$$

---

### Step 5

Run independent source encoder:

$$
H_s=
E_s(\tilde Z_s).
$$

---

### Step 6

Run independent target encoder:

$$
H_t=
E_t(\tilde Z_t).
$$

---

### Step 7

Generate source classification prediction:

$$
\hat y_s=
C(h_s).
$$

---

### Step 8

Calculate source classification loss:

$$
\mathcal{L}_{cls}.
$$

---

### Step 9

Run coupled cross-attention.

Source queries target:

$$
C_{s\leftarrow t}.
$$

Target queries source:

$$
C_{t\leftarrow s}.
$$

---

### Step 10

Run HMA teacher encoders without gradients.

Generate:

$$
h_s^{teacher}
$$

and:

$$
h_t^{teacher}.
$$

---

### Step 11

Calculate bidirectional distillation:

$$
\mathcal{L}_{dist}.
$$

---

### Step 12

Calculate MoE load-balancing loss:

$$
\mathcal{L}_{balance}.
$$

---

### Step 13

Calculate total loss:

$$
\mathcal{L}_{total}.
$$

---

### Step 14

Backpropagate:

```text
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Teacher parameters must not receive gradients.

---

### Step 15

Update HMA parameter history.

Store the updated student parameters.

Compute the HMA teacher parameters.

Update the teacher.

---

# 24. Step 20 — Inference

During target-domain inference:

```text
Target Patch
     │
     ▼
Semantic Tokenizer
     │
     ▼
5 Semantic Tokens
     │
     ▼
Target Transformer
     │
     ▼
Target Representation
     │
     ▼
Classifier
     │
     ▼
Predicted Class
```

However, one important architectural decision remains:

> Which representation is used for final target classification?

There are two possible approaches.

---

## Option A — Independent Target Representation

Use:

$$
h_t
$$

directly:

$$
\hat y_t=C(h_t).
$$

Advantages:

* no source sample is required during inference;
* simple deployment;
* computationally efficient.

This is the recommended initial implementation.

---

## Option B — Coupled Target Representation

Use:

$$
h_{t\leftarrow s}.
$$

This requires source-domain tokens during inference.

It may provide stronger adaptation but creates a dependency on source samples.

For the first implementation, use Option A.

---

# 25. Recommended Initial Hyperparameters

The following should be considered starting points rather than final values.

```text
Patch size:
13 × 13 × 13

Semantic tokens:
5

Embedding dimension:
64 or 128

Transformer layers:
2 to 4

Attention heads:
4

MoE experts:
4

Routing:
Top-2

Expert hidden dimension:
2 × embedding dimension

Dropout:
0.1

Activation:
LeakyReLU

Optimizer:
AdamW

Learning rate:
1e-4 to 3e-4

Weight decay:
1e-4

Batch size:
Depends on GPU memory

Distillation warm-up:
10 to 30 epochs
```

---

# 26. Recommended Project Structure

```text
project/
│
├── data/
│   ├── source_dataset.py
│   ├── target_dataset.py
│   └── preprocessing.py
│
├── models/
│   ├── semantic_tokenizer.py
│   ├── positional_encoding.py
│   ├── moe.py
│   ├── transformer_block.py
│   ├── transformer_encoder.py
│   ├── coupled_attention.py
│   ├── coupled_transformer.py
│   ├── hma_teacher.py
│   ├── classifier.py
│   └── cross_domain_model.py
│
├── losses/
│   ├── classification_loss.py
│   ├── distillation_loss.py
│   └── moe_balance_loss.py
│
├── training/
│   ├── trainer.py
│   └── scheduler.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
│
├── configs/
│   └── default.yaml
│
├── train.py
├── test.py
└── requirements.txt
```

---

# 27. Recommended Implementation Order

The model should not be implemented all at once.

Use the following sequence.

## Phase 1

Implement data loading.

Verify:

```text
Input patch:
[B, 1, 13, 13, 13]
```

---

## Phase 2

Implement the semantic tokenizer.

Verify:

```text
Input:
[B, 1, 13, 13, 13]

Output:
[B, 5, D]
```

---

## Phase 3

Implement a normal Transformer without MoE.

Verify:

```text
[B, 5, D]
→
[B, 5, D]
```

Train source-only classification.

This establishes a baseline.

---

## Phase 4

Replace FFN with Top-2 MoE.

Train source-only classification again.

Compare:

```text
Normal Transformer
vs
MoE Transformer
```

---

## Phase 5

Add the independent target branch.

Verify that source and target forward passes work independently.

---

## Phase 6

Implement bidirectional coupled cross-attention.

Verify:

```text
Source queries Target:
[B, 5, D]
```

and:

```text
Target queries Source:
[B, 5, D]
```

---

## Phase 7

Add MoE to the coupled Transformer.

Add and verify the balancing loss.

---

## Phase 8

Implement teacher-student distillation using a simple frozen teacher first.

Before implementing HMA, verify:

```text
Stop-gradient works correctly.

Teacher receives no gradients.

Student receives gradients.
```

---

## Phase 9

Implement the HMA teacher.

Maintain parameter history and verify that the teacher parameters evolve smoothly.

---

## Phase 10

Add distillation warm-up.

Train the complete model.

---

# 28. Recommended Ablation Study

The final research should include at least the following experiments.

| Experiment           | Tokenizer | MoE | Coupled Attention | Distillation | HMA |
| -------------------- | --------- | --- | ----------------- | ------------ | --- |
| Source Only          | CNN       | No  | No                | No           | No  |
| Transformer Baseline | CNN       | No  | No                | No           | No  |
| MoE                  | CNN       | Yes | No                | No           | No  |
| Coupled              | CNN       | Yes | Yes               | No           | No  |
| Distillation         | CNN       | Yes | Yes               | Yes          | No  |
| Full Model           | CNN       | Yes | Yes               | Yes          | Yes |

Also compare:

```text
Top-1 routing
vs
Top-2 routing
```

and:

```text
2 experts
vs
4 experts
vs
8 experts
```

and:

```text
1 semantic token
vs
5 semantic tokens
vs
10 semantic tokens
```

---

# 29. Important Research Questions

The following questions should be answered experimentally.

## Q1

Does reducing each patch to five semantic tokens preserve sufficient hyperspectral information?

## Q2

Does Top-2 MoE improve cross-domain generalization compared with a standard FFN?

## Q3

Do experts specialize differently for source and target domains?

## Q4

Does bidirectional cross-attention improve adaptation?

## Q5

Does representation-level distillation outperform logit-level distillation?

## Q6

Does HMA teacher smoothing improve stability compared with:

* no teacher;
* frozen teacher;
* EMA teacher?

The final comparison against an EMA teacher is particularly important because EMA teachers are much more established and computationally simpler.

---

# 30. Final Model Summary

The complete model can be summarized as:

$$
\boxed{
X_s
\rightarrow
\text{3D/2D CNN Tokenizer}
\rightarrow
5\text{ Semantic Tokens}
\rightarrow
\text{Transformer + Top-2 MoE}
\rightarrow
H_s
}
$$

and:

$$
\boxed{
X_t
\rightarrow
\text{3D/2D CNN Tokenizer}
\rightarrow
5\text{ Semantic Tokens}
\rightarrow
\text{Transformer + Top-2 MoE}
\rightarrow
H_t
}
$$

The coupled branch performs:

$$
\boxed{
H_s
\overset{\text{query target}}{\longrightarrow}
H_{s\leftarrow t}
}
$$

and:

$$
\boxed{
H_t
\overset{\text{query source}}{\longrightarrow}
H_{t\leftarrow s}
}
$$

The coupled representations are trained through HMA teacher-student distillation:

$$
\boxed{
\mathcal{L}_{dist}
=
D
(
\text{sg}(H_s^{teacher}),
H_{s\leftarrow t}
)
+
D
(
\text{sg}(H_t^{teacher}),
H_{t\leftarrow s}
)
}
$$

The final objective is:

$$
\boxed{
\mathcal{L}_{total}
=
\mathcal{L}_{classification}
+
\lambda_{dist}\mathcal{L}_{distillation}
+
\lambda_{balance}\mathcal{L}_{MoE}
}
$$

The implementation should begin with the semantic tokenizer and source-only baseline, then progressively introduce MoE, coupled attention, distillation, and finally the HMA teacher.

This staged implementation is important because it allows every proposed contribution to be independently verified and evaluated.
