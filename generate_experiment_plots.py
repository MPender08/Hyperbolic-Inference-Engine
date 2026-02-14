import torch
import numpy as np
import ot
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist, cosine
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- MATHEMATICAL FRAMEWORK ---
class SemanticTopology:
    def __init__(self, model_name="gpt2"):
        print(f"Loading Manifold: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        raw_embeds = self.model.transformer.wte.weight.detach()
        # L2 Normalize the embedding matrix for consistent cosine metrics
        self.embed_matrix = torch.nn.functional.normalize(raw_embeds, p=2, dim=1).numpy()

    def get_latent_state(self, text, top_k=50):
        """Extracts the distribution and center of mass for a given prompt."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = (top_probs / top_probs.sum()).numpy()
            
            indices = top_indices.numpy()
            center_of_mass = np.average(self.embed_matrix[indices], axis=0, weights=top_probs)
            
            return {
                "indices": indices,
                "probs": top_probs,
                "center": center_of_mass,
                "token_count": inputs['input_ids'].shape[1]
            }

    def compute_kappa(self, state_curr, state_prev):
        """Calculates local Ollivier-Ricci curvature and distance between adjacent states."""
        # 1-Wasserstein (EMD) Cost Matrix based on cosine distance
        cost_matrix = cdist(self.embed_matrix[state_curr['indices']], 
                            self.embed_matrix[state_prev['indices']], 
                            metric='cosine')
        
        w1 = ot.emd2(state_curr['probs'], state_prev['probs'], cost_matrix)
        d_xy = cosine(state_curr['center'], state_prev['center'])
        
        if d_xy < 1e-9: 
            return 0.0, 0.0
        
        kappa = 1 - (w1 / d_xy)
        return kappa, d_xy

# --- EXPERIMENT EXECUTION ---
topo = SemanticTopology()
origin_state = topo.get_latent_state("The")

results = []
for g_idx in tqdm(range(30), desc="Processing Gradients"):
    prompts = [
        "The cat is on the mat.",
        "The cat is on the mat, implying a spatial relationship.",
        "The cat is on the mat, implying a spatial relationship constrained by gravity.",
        "The cat is on the mat, implying a spatial relationship constrained by gravity and classical mechanics.",
        "The cat is on the mat, implying a spatial relationship constrained by gravity and classical mechanics within a formal logical lens.",
        "The cat is on the mat, implying a spatial relationship constrained by gravity and classical mechanics within a formal logical lens requiring absolute proof."
    ]
    
    prev_state = origin_state
    for level, prompt in enumerate(prompts):
        curr_state = topo.get_latent_state(prompt)
        kappa, dist = topo.compute_kappa(curr_state, prev_state)
        
        tokens_added = curr_state['token_count'] - prev_state['token_count']
        v_inc = dist / max(tokens_added, 1)
        
        results.append({
            "Level": level, 
            "Type": "Experiment", 
            "Kappa": kappa, 
            "Step_Distance": dist, 
            "Velocity": v_inc
        })
        prev_state = curr_state

# --- NULL MODEL (CONTROL) ---
for c_idx in tqdm(range(30), desc="Processing Control"):
    prev_state = origin_state
    for level in range(6):
        random_tokens = "".join([chr(random.randint(97, 122)) for _ in range(random.randint(5, 20))])
        curr_state = topo.get_latent_state(random_tokens)
        kappa, dist = topo.compute_kappa(curr_state, prev_state)
        v_inc = dist / 1 
        results.append({
            "Level": level, "Type": "Control", "Kappa": kappa, "Step_Distance": dist, "Velocity": v_inc
        })
        prev_state = curr_state

df = pd.DataFrame(results)
df_exp = df[df['Type'] == 'Experiment']
df_ctrl = df[df['Type'] == 'Control']

# --- FINAL METRICS ---
avg_step_dist = df_exp['Step_Distance'].mean()
print(f"\n--- Analysis Complete ---")
print(f"Average Step Distance (Experiment): {avg_step_dist:.4f}")

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# Scatter for Experiment
scatter = plt.scatter(df_exp['Kappa'], df_exp['Velocity'], c=df_exp['Level'], 
                      cmap='plasma', s=100, alpha=0.8, edgecolors='black', label='Semantic Gradient')

# Scatter for Control
plt.scatter(df_ctrl['Kappa'], df_ctrl['Velocity'], c='gray', marker='x', 
            s=100, linewidths=2, alpha=0.6, label='Null Model (Random)')

# Visual elements
plt.colorbar(scatter).set_label('Gating Density (Constraint Level)', fontsize=12)
plt.title("Manifold Dynamics: Local Curvature vs. Incremental Velocity", fontsize=14, fontweight='bold')
plt.xlabel(r"Local Curvature ($\kappa$)", fontsize=12)
plt.ylabel(r"Incremental Velocity ($V_{inc}$)", fontsize=12)

# Complexity Trendline (Geodesic Efficiency Check)
z = np.polyfit(df_exp['Kappa'], df_exp['Velocity'], 1)
p = np.poly1d(z)
plt.plot(df_exp['Kappa'], p(df_exp['Kappa']), "r--", alpha=0.8, label=f"Trendline (r={np.corrcoef(df_exp['Kappa'], df_exp['Velocity'])[0,1]:.4f})")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# SAVE OUTPUT
plt.savefig("manifold_corrected_results.png", dpi=300)
print("Plot saved as manifold_corrected_results.png")