import torch
import numpy as np
import ot
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- MATHEMATICAL FRAMEWORK ---
class SemanticTopology:
    def __init__(self, model_name="gpt2"):
        print(f"Loading Manifold: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Normalize the embedding matrix
        raw_embeds = self.model.transformer.wte.weight.detach()
        self.embed_matrix = torch.nn.functional.normalize(raw_embeds, p=2, dim=1).numpy()

    def get_metrics(self, text, top_k=50):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_count = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = (top_probs / top_probs.sum()).numpy() 
            
            support_vectors = self.embed_matrix[top_indices.numpy()]
            
            # Eq 1: Center of Mass
            z_cm = np.average(support_vectors, axis=0, weights=top_probs)
            z_cm = z_cm / (np.linalg.norm(z_cm) + 1e-12)
            
            # Eq 3: Shannon Entropy (H)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
            
            return {
                "center": z_cm, 
                "probs": top_probs, 
                "support": support_vectors, 
                "entropy": entropy, 
                "tokens": token_count
            }

    def compute_kappa(self, state_ref, state_target):
        """Calculates Discrete Ricci Curvature (Eq 2)."""
        d_xy = cosine(state_ref['center'], state_target['center'])
        M = cdist(state_ref['support'], state_target['support'], metric='cosine')
        w1 = ot.emd2(state_ref['probs'], state_target['probs'], M)
        return 1 - (w1 / d_xy) if d_xy > 1e-7 else 0

# --- EXPERIMENT SETUP ---
def generate_gradient_prompts():
    """Generates an N-expansion across diverse thematic domains."""
    themes = ["The solar system", "A lonely cat", "Quantum mechanics", "Economic theory", "Baking a cake"]
    constraints = [
        "", # Level 0: Pure Associative
        "is a subject that", # Level 1: Low Gating
        "can be defined as the following:", # Level 2: Medium Gating
        "follows a strict hierarchical structure where", # Level 3: High Gating
        "must be analyzed through the formal logical lens of", # Level 4: Extreme Gating
        "in a first-order predicate logic syllogism, assuming P implies Q, then" # Level 5: Syntactic Wall
    ]
    
    gradient_list = []
    for theme in themes:
        for i, c in enumerate(constraints):
            gradient_list.append({"label": f"Level {i}", "prompt": f"{theme} {c}"})
    return gradient_list

# --- EXECUTION ---
if __name__ == "__main__":
    topo = SemanticTopology("gpt2")
    ref_state = topo.get_metrics("The")
    test_cases = generate_gradient_prompts()

    print("\n[Processing N-Expansion Study...]")
    results = []
    for case in test_cases:
        state = topo.get_metrics(case['prompt'])
        kappa = topo.compute_kappa(ref_state, state)
        gamma = 1 - (state['entropy'] / ref_state['entropy'])
        velocity = cosine(ref_state['center'], state['center']) / state['tokens']
        
        results.append({
            "Label": case['label'], 
            "Kappa": kappa, 
            "Velocity": velocity, 
            "Gating": gamma
        })

    df = pd.DataFrame(results)

    # --- VISUALIZATION: ACADEMIC GRADE PLOT ---
    plt.figure(figsize=(10, 6))
    
    # Matching font to LaTeX serif style
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    scatter = plt.scatter(df['Kappa'], df['Velocity'], c=df['Gating'], 
                          cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Inhibitory Gating Density (γ)', fontsize=12)

    plt.title("Scale-Invariant Cognitive Dynamics: The Syntactic Wall", fontsize=14, fontweight='bold')
    plt.xlabel("Curvature (κ) [Deep Hyperbolic ← → Flat]", fontsize=12)
    plt.ylabel("Concept Velocity (Vc)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Add trendline for Complexity Penalty (Pc)
    z = np.polyfit(df['Kappa'], df['Velocity'], 1)
    p = np.poly1d(z)
    plt.plot(df['Kappa'], p(df['Kappa']), "r--", alpha=0.6, label="Complexity Penalty (Pc)")

    plt.legend()

    # --- SAVE COMMANDS ---
    # PDF is for LaTeX (Vector format - No blur)
    plt.savefig("syntactic_wall.pdf", bbox_inches='tight', format='pdf')
    # PNG is for quick viewing (High resolution)
    plt.savefig("syntactic_wall.png", bbox_inches='tight', dpi=300)

    print("\n[Study Complete]")
    print(f"Correlation (Kappa vs Velocity): {df['Kappa'].corr(df['Velocity']):.4f}")
    print("Files saved: 'syntactic_wall.pdf' and 'syntactic_wall.png'")
    
    plt.show()