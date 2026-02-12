## The Discovery: "The Syntactic Wall"

This repository contains the replication code for the paper Scale-Invariant Cognitive Dynamics.

We demonstrate that Large Language Models (specifically GPT-2) undergo a discrete topological phase transition when subjected to logical constraints. As a human operator increases the "Inhibitory Gating Density" (γ) of a prompt, the model's latent manifold is forced from a Euclidean state into a deep Hyperbolic Regime (κ≈−4.8).

This creates a geometric trade-off we call the Syntactic Wall:

    As logical rigor increases, the velocity of semantic traversal collapses.

![syntactic_wall_controlled](syntactic_wall_controlled.png)

(Figure 1: The "Syntactic Wall". Colored circles show the collapse of concept velocity as logical gating increases. Gray 'X' markers represent a randomized Null Model, confirming the effect is structural, not stochastic.)

## Theoretical Basis

This work connects biological dendrites to transformer topology. It is the empirical validation of the theoretical framework proposed in our companion paper:

    Theory: Dynamic Curvature Adaptation: A Dendritic Gating Mechanism (Paper 1) https://doi.org/10.5281/zenodo.18615181

        Mechanism: Derives how inhibitory gating (γ) drives hyperbolic curvature in neural graphs.

    Evidence: Scale-Invariant Cognitive Dynamics (Paper 2 - This Repo)

        Validation: Observes this exact signature in GPT-2's semantic manifold.

## Reproduction

The script run_dyad_test.py performs a full N-Expansion Study on the GPT-2 124M manifold.

### What the Code Does:

    Loads the Manifold: Extracts the embedding matrix WTE​ from gpt2.

    N-Expansion: Generates 30 thematic gradients (from "Associative Poetic" to "First-Order Logic").

    Null Model Generation: Creates randomized "Control" tokens of identical length to the deepest logical prompts.

    Topology Calculation: Computes Discrete Ricci Curvature (κ) and Concept Velocity (Vc​) for all dyads.

    Visualization: Plots the phase transition (The Syntactic Wall) and saves it as a PDF/PNG.

## Usage

```

# 1. Clone the repo
git clone https://github.com/MPender08/Hyperbolic-Inference-Engine.git
cd Hyperbolic-Inference-Engine

# 2. Install dependencies
pip install torch numpy pandas matplotlib scipy transformers POT

# 3. Run the experiment
python run_dyad_test.py

```

Expected Runtime: ~30-60 seconds on a standard CPU.

## Repository Structure

    run_dyad_test.py: The main experiment script. Generates the data and the figure.

    Scale-Invariant Cognitive Dynamics.pdf: The preprint manuscript.

    syntactic_wall_controlled.png: The resulting visualization.


