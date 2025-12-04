import os
import math
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import OntologyGraphDataset
from model import OntoAlignEncoder
from torch_geometric.data import Data
import difflib
from tqdm import tqdm
from alignment_utils import AlignmentEvaluator
from typing import List, Dict, Tuple, Set
from collections import defaultdict

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_property_vocab(graphs_dir: str) -> Dict[str, int]:
    vocab_path = os.path.join(graphs_dir, "edge_property_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()}
    return {"__NONE__": 0}

# === Mixed Domain Prompt Adapter ===
class MixedDomainPromptAdapter(nn.Module):
    def __init__(self, pretrained_domain_embs: torch.Tensor):
        super().__init__()
        self.num_domains, self.hidden_dim = pretrained_domain_embs.shape
        self.register_buffer("pretrained_embs", pretrained_domain_embs.detach().clone())
        # Store normalized keys for similarity lookup
        self.register_buffer(
            "domain_keys",
            F.normalize(pretrained_domain_embs.detach().clone(), dim=-1)
        )
        
        # Learnable components
        self.delta = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.bias_logits = nn.Parameter(torch.zeros(self.num_domains))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def get_mixing_weights(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        graph_embedding: [1, hidden_dim] aggregated representation of the current graph
        Returns: [num_domains] softmax weights
        """
        if graph_embedding is None:
            # fallback to uniform if no context is provided
            return torch.full(
                (self.num_domains,),
                1.0 / self.num_domains,
                device=self.pretrained_embs.device
            )
        
        query = F.normalize(self.query_proj(graph_embedding), dim=-1)  # [1, d]
        keys = F.normalize(self.key_proj(self.domain_keys), dim=-1)    # [num_domains, d]
        sim = torch.matmul(query, keys.transpose(0, 1)).squeeze(0)     # [num_domains]
        sim = sim / self.temperature.clamp(min=1e-3)
        sim = sim + self.bias_logits
        weights = F.softmax(sim, dim=-1)
        return weights
    
    def forward(self, graph_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (p_uni, p_mix)
        """
        p_uni = self.delta
        weights = self.get_mixing_weights(graph_embedding)
        mixed_prompt = torch.matmul(weights, self.pretrained_embs).unsqueeze(0)
        return p_uni, mixed_prompt

    def get_top_weights(self, graph_embedding: torch.Tensor, top_k=5):
        weights = self.get_mixing_weights(graph_embedding)
        values, indices = torch.topk(weights, k=min(top_k, self.num_domains))
        return values.detach().cpu().tolist(), indices.detach().cpu().tolist()


def resolve_graph_domain_id(graph) -> int:
    value = getattr(graph, "domain_id", None)
    if isinstance(value, torch.Tensor):
        return int(value.view(-1)[0].item())
    if value is None:
        return 0
    return int(value)

# --- Helper Functions for Lexical Matching ---
def get_lexical_similarity(s1, s2):
    if s1 == s2: return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def normalize_lexical_term(term: str) -> str:
    return term.strip().lower()

def get_lexical_lists(graph: Data) -> List[List[str]]:
    lexical = getattr(graph, 'node_lexical', None)
    labels = getattr(graph, 'node_labels', None)
    texts = getattr(graph, 'node_text', None)

    def fallback(idx: int) -> str:
        if labels and idx < len(labels) and isinstance(labels[idx], str) and labels[idx].strip():
            return labels[idx]
        if texts and idx < len(texts) and isinstance(texts[idx], str) and texts[idx].strip():
            return texts[idx]
        return ""

    if lexical:
        result: List[List[str]] = []
        for idx, terms in enumerate(lexical):
            filtered = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
            if not filtered:
                fb = fallback(idx)
                result.append([fb] if fb else [])
            else:
                result.append(filtered)
        return result
    if labels:
        return [[lbl.strip()] if isinstance(lbl, str) and lbl.strip() else [fallback(i)] for i, lbl in enumerate(labels)]
    if texts:
        return [[txt] if isinstance(txt, str) else [] for txt in texts]
    raise ValueError("Graph is missing lexical metadata.")

def generate_lexical_seeds(src_graph, tgt_graph) -> List[Tuple[int, int]]:
    print("Generating lexical seeds...")
    src_terms = get_lexical_lists(src_graph)
    tgt_terms = get_lexical_lists(tgt_graph)
    
    src_map = {}
    for idx, terms in enumerate(src_terms):
        for t in terms:
            norm = normalize_lexical_term(t)
            if norm: src_map.setdefault(norm, []).append(idx)
            
    tgt_map = {}
    for idx, terms in enumerate(tgt_terms):
        for t in terms:
            norm = normalize_lexical_term(t)
            if norm: tgt_map.setdefault(norm, []).append(idx)
            
    pairs = set()
    src_matched = set()
    tgt_matched = set()
    
    # 1. Exact Match
    common_terms = set(src_map.keys()) & set(tgt_map.keys())
    for term in common_terms:
        s_indices = src_map[term]
        t_indices = tgt_map[term]
        
        if len(s_indices) == 1 and len(t_indices) == 1:
            s, t = s_indices[0], t_indices[0]
            if s not in src_matched and t not in tgt_matched:
                pairs.add((s, t))
                src_matched.add(s)
                tgt_matched.add(t)

    print(f"  Exact Matches: {len(pairs)}")

    # 2. Strict Fuzzy Match
    unmatched_src = [i for i in range(len(src_terms)) if i not in src_matched]
    unmatched_tgt = [j for j in range(len(tgt_terms)) if j not in tgt_matched]
    
    src_labels = [(i, src_terms[i][0]) for i in unmatched_src if src_terms[i]]
    tgt_labels = [(j, tgt_terms[j][0]) for j in unmatched_tgt if tgt_terms[j]]
    
    print(f"  Running fuzzy match on {len(src_labels)} src and {len(tgt_labels)} tgt unmatched nodes...")
    
    tgt_buckets = defaultdict(list)
    for j, txt in tgt_labels:
        if txt:
            tgt_buckets[txt[0].lower()].append((j, txt))
        
    fuzzy_count = 0
    for s_idx, s_txt in tqdm(src_labels, desc="Fuzzy Matching"):
        if not s_txt: continue
        candidates = tgt_buckets.get(s_txt[0].lower(), [])
        best_score = 0
        best_j = -1
        
        for j_idx, t_txt in candidates:
            if abs(len(s_txt) - len(t_txt)) > 3: continue
            score = get_lexical_similarity(s_txt.lower(), t_txt.lower())
            if score > best_score:
                best_score = score
                best_j = j_idx
                
        if best_score > 0.9:
            pairs.add((s_idx, best_j))
            src_matched.add(s_idx)
            tgt_matched.add(best_j)
            fuzzy_count += 1
            
    print(f"  Fuzzy Matches: {fuzzy_count}")
    print(f"  Total Seeds: {len(pairs)}")
    return list(pairs)

def enforce_bijection(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    unique: List[Tuple[int, int]] = []
    seen_src: Set[int] = set()
    seen_tgt: Set[int] = set()
    for src_idx, tgt_idx in pairs:
        if src_idx in seen_src or tgt_idx in seen_tgt:
            continue
        unique.append((src_idx, tgt_idx))
        seen_src.add(src_idx)
        seen_tgt.add(tgt_idx)
    return unique

# --- Helper Functions for Structural Boosting (Jaccard) ---
def get_adj_set(graph):
    """
    Builds adjacency set from graph edge_index.
    Returns: dict {node_idx: set(neighbor_indices)}
    """
    adj = defaultdict(set)
    if hasattr(graph, 'edge_index'):
        src, dst = graph.edge_index
        src_list = src.tolist()
        dst_list = dst.tolist()
        for u, v in zip(src_list, dst_list):
            adj[u].add(v)
            # Undirected for structural context
            adj[v].add(u)
    return adj

def compute_jaccard_similarity(candidates: List[Tuple[int, int]], 
                               adj_src: Dict[int, Set[int]], 
                               adj_tgt: Dict[int, Set[int]], 
                               anchor_map_s2t: Dict[int, int]) -> torch.Tensor:
    """
    Computes Jaccard Similarity based on 1-hop structural overlap via anchors.
    Jaccard = |Mapped_N(s) ∩ N(t)| / |Mapped_N(s) ∪ N(t)|
    """
    scores = []
    for s, t in candidates:
        # Neighbors of s
        n_s = adj_src.get(s, set())
        # Map s-neighbors to target side using known anchors
        mapped_n_s = set()
        for neighbor in n_s:
            if neighbor in anchor_map_s2t:
                mapped_n_s.add(anchor_map_s2t[neighbor])
        
        # Neighbors of t
        n_t = adj_tgt.get(t, set())
        
        if not mapped_n_s and not n_t:
            scores.append(0.0)
            continue
            
        intersection = len(mapped_n_s.intersection(n_t))
        union = len(mapped_n_s.union(n_t))
        
        if union == 0:
            scores.append(0.0)
        else:
            scores.append(intersection / union)
            
    return torch.tensor(scores)


# --- Advanced Loss Function (InfoNCE) ---
class InfoNCELoss(nn.Module):
    """
    Bi-directional InfoNCE Loss for Alignment.
    Optimizes (src -> tgt) and (tgt -> src) retrieval simultaneously.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, h_src, h_tgt, src_indices, tgt_indices):
        # Gather active embeddings
        anchor = h_src[src_indices]  # [B, d]
        positive = h_tgt[tgt_indices] # [B, d]
        
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Similarity matrix: [B, B]
        logits = torch.mm(anchor, positive.t()) / self.temperature
        
        # Labels: diagonal (0..B-1)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Bi-directional loss
        loss_s2t = self.cross_entropy(logits, labels)
        loss_t2s = self.cross_entropy(logits.t(), labels)
        
        return (loss_s2t + loss_t2s) / 2

# --- CSLS Utility (Structural Calibration) ---
def compute_csls_sim(h_src, h_tgt, k=10):
    """
    Compute CSLS similarity matrix.
    r(x, y) = 2*cos(x, y) - avg_sim_src(x) - avg_sim_tgt(y)
    """
    h_src = F.normalize(h_src, p=2, dim=1)
    h_tgt = F.normalize(h_tgt, p=2, dim=1)
    
    sim = torch.mm(h_src, h_tgt.t())
    
    # K-Nearest Neighbors for penalization
    topk_src, _ = torch.topk(sim, k=k, dim=1)
    avg_sim_src = topk_src.mean(dim=1) # [N]
    
    topk_tgt, _ = torch.topk(sim, k=k, dim=0)
    avg_sim_tgt = topk_tgt.mean(dim=0) # [M]
    
    # Broadcast: CSLS = 2*S - A - B
    csls_sim = 2 * sim - avg_sim_src.unsqueeze(1) - avg_sim_tgt.unsqueeze(0)
    return csls_sim

def get_csls_mnn(
    h_src,
    h_tgt,
    threshold=0.5,
    k=10,
    adaptive=False,
    std_multiplier=1.5,
    min_threshold=None,
    min_pairs=20
):
    """
    Get Mutual Nearest Neighbors based on CSLS score.
    Returns tuples (src_idx, tgt_idx, score).
    """
    csls_sim = compute_csls_sim(h_src, h_tgt, k=k)

    stats_max = csls_sim.max().item()
    stats_mean = csls_sim.mean().item()
    stats_min = csls_sim.min().item()
    stats_std = csls_sim.std().item()
    used_threshold = threshold

    if adaptive:
        dynamic_thr = stats_mean + std_multiplier * stats_std
        if min_threshold is not None:
            dynamic_thr = max(dynamic_thr, min_threshold)
        max_allowed = stats_max - 1e-6 if math.isfinite(stats_max) else threshold
        if not math.isfinite(dynamic_thr):
            used_threshold = max_allowed
        else:
            used_threshold = min(dynamic_thr, max_allowed)
        if not math.isfinite(used_threshold):
            used_threshold = threshold

    val_s2t, idx_s2t = csls_sim.max(dim=1)
    val_t2s, idx_t2s = csls_sim.max(dim=0)

    print(f"  [DEBUG] CSLS Stats: Max={stats_max:.4f}, Mean={stats_mean:.4f}, Min={stats_min:.4f}, Std={stats_std:.4f}")
    print(f"  [DEBUG] Max Score Sample: {val_s2t[:5].tolist()}")
    print(f"  [DEBUG] Using CSLS threshold={used_threshold:.4f}")

    pairs = []
    all_mnn = []
    for i in range(h_src.size(0)):
        j = idx_s2t[i].item()
        score = val_s2t[i].item()

        # Check Mutual Nearest
        if idx_t2s[j].item() == i:
            all_mnn.append((i, j, score))
            if score > used_threshold:
                pairs.append((i, j, score))

    if min_pairs and len(pairs) < min_pairs and all_mnn:
        needed = min_pairs - len(pairs)
        seen = {(i, j) for i, j, _ in pairs}
        for i, j, score in sorted(all_mnn, key=lambda x: x[2], reverse=True):
            if (i, j) in seen:
                continue
            pairs.append((i, j, score))
            seen.add((i, j))
            needed -= 1
            if needed <= 0:
                break

    return pairs

# --- MNN Candidate Generation ---
def get_mnn_candidates(h_src, h_tgt, threshold=0.5):
    """
    Return pairs (i, j) that are mutual nearest neighbors with sim > threshold.
    """
    src_emb = F.normalize(h_src, p=2, dim=1)
    tgt_emb = F.normalize(h_tgt, p=2, dim=1)
    sim_matrix = torch.mm(src_emb, tgt_emb.t())

    # Forward & Backward NN
    val_s, idx_s = sim_matrix.max(dim=1)
    val_t, idx_t = sim_matrix.max(dim=0)

    pairs = []
    for i in range(sim_matrix.size(0)):
        j = idx_s[i].item()
        if idx_t[j].item() == i and val_s[i].item() > threshold:
            pairs.append((i, j))
    return pairs


def prune_noisy_anchors(pairs, h_src, h_tgt, lexical_seed_set, min_score):
    if min_score is None or min_score <= 0:
        return pairs, []
    src_norm = F.normalize(h_src, p=2, dim=1)
    tgt_norm = F.normalize(h_tgt, p=2, dim=1)
    cleaned = []
    dropped = 0
    removed_pairs = []
    for s, t in pairs:
        if (s, t) in lexical_seed_set:
            cleaned.append((s, t))
            continue
        score = float(torch.dot(src_norm[s], tgt_norm[t]).item())
        if score >= min_score:
            cleaned.append((s, t))
        else:
            dropped += 1
            removed_pairs.append((s, t))
    if dropped > 0:
        print(f"  Pruned {dropped} low-sim anchors (<{min_score:.2f})")
    return cleaned, removed_pairs

# --- High-Confidence Promotion ---
def get_embedding_topk_candidates(h_src, h_tgt, topk=50, min_score=0.5, margin=0.1):
    if topk <= 0 or min_score <= 0:
        return []
    src_emb = F.normalize(h_src, p=2, dim=1)
    tgt_emb = F.normalize(h_tgt, p=2, dim=1)
    sim_matrix = torch.mm(src_emb, tgt_emb.t())
    if sim_matrix.size(1) == 0:
        return []
    k = min(2, sim_matrix.size(1))
    vals, idx = torch.topk(sim_matrix, k=k, dim=1)
    best = vals[:, 0]
    if k > 1:
        second = vals[:, 1]
    else:
        second = torch.full_like(best, -1.0)
    margin_mask = (best - second) >= margin
    score_mask = best >= min_score
    val_t, idx_t = sim_matrix.max(dim=0)
    rows = torch.nonzero(margin_mask & score_mask, as_tuple=True)[0]

    candidates = []
    for r in rows.tolist():
        j = int(idx[r, 0].item())
        if idx_t[j].item() != r:
            continue
        candidates.append((r, j, float(best[r].item())))

    candidates.sort(key=lambda x: x[2], reverse=True)
    if topk < len(candidates):
        candidates = candidates[:topk]
    return candidates

def get_high_conf_candidates(h_src, h_tgt, topk=50, min_score=0.3, margin=0.05):
    """
    Get high-confidence candidates using top-1 similarity with a margin over the runner-up.
    Returns tuples (src_idx, tgt_idx, score) sorted by score.
    """
    src_emb = F.normalize(h_src, p=2, dim=1)
    tgt_emb = F.normalize(h_tgt, p=2, dim=1)
    sim_matrix = torch.mm(src_emb, tgt_emb.t())
    if sim_matrix.size(1) == 0:
        return []

    k = min(max(2, topk), sim_matrix.size(1))
    val, idx = torch.topk(sim_matrix, k=k, dim=1)

    best = val[:, 0]
    if k > 1:
        second = val[:, 1]
        margin_mask = (best - second) >= margin
    else:
        second = torch.full_like(best, -1.0)
        margin_mask = torch.ones_like(best, dtype=torch.bool)

    score_mask = best >= min_score
    mask = margin_mask & score_mask
    rows = torch.nonzero(mask, as_tuple=True)[0]

    candidates = []
    for r in rows.tolist():
        candidates.append((int(r), int(idx[r, 0].item()), float(best[r].item())))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates

# --- Hard Negative Mining ---
def get_hard_negatives(z_s, h_t, known_tgt_indices, device, beta=20):
    """
    Sample hard negatives: Close to s, but not the true t.
    """
    # Sim: [Batch, N_all]
    sim = torch.mm(F.normalize(z_s, p=2, dim=1), F.normalize(h_t, p=2, dim=1).t())

    # Get top-beta candidates for each source
    _, indices = torch.sort(sim, dim=1, descending=True)

    # For each source, select a hard negative from top-beta (excluding known positives)
    neg_indices = []
    for i in range(z_s.size(0)):
        known_t = known_tgt_indices[i].item()
        # Get top-beta indices for this source
        top_beta = indices[i, :beta].tolist()

        # Remove known positive if present
        if known_t in top_beta:
            top_beta.remove(known_t)

        # Select a random hard negative from remaining candidates
        if top_beta:
            neg_idx = top_beta[torch.randint(0, len(top_beta), (1,), device=device).item()]
        else:
            # Fallback: random selection if no good candidates
            neg_idx = torch.randint(0, h_t.size(0), (1,), device=device).item()

        neg_indices.append(neg_idx)

    return h_t[torch.tensor(neg_indices, device=device)]

# --- Hybrid Inference & Evaluation ---
def evaluate_hybrid(src_graph, tgt_graph, h_src, h_tgt, lexical_seeds, evaluator):
    if not evaluator: return
    
    # 1. Trust Lexical Seeds but allow confident embedding overrides
    pred_pairs = set(lexical_seeds)
    lexical_assignments = {s: t for s, t in lexical_seeds}
    lexical_tgt = set(lexical_assignments.values())
    lexical_override_margin = 0.05
    
    # 2. Retrieve missing using MNN (Strict for Eval)
    with torch.no_grad():
        src_emb = F.normalize(h_src, p=2, dim=1)
        tgt_emb = F.normalize(h_tgt, p=2, dim=1)
        sim_matrix = torch.mm(src_emb, tgt_emb.t())
        
        val_s, idx_s = sim_matrix.max(dim=1)
        val_t, idx_t = sim_matrix.max(dim=0)
        
        for i in range(sim_matrix.size(0)):
            j = idx_s[i].item()
            candidate_is_mnn = idx_t[j].item() == i
            current_t = lexical_assignments.get(i)

            if current_t is not None:
                if j == current_t or not candidate_is_mnn:
                    continue
                lexical_score = sim_matrix[i, current_t].item()
                candidate_score = val_s[i].item()
                if candidate_score <= lexical_score + lexical_override_margin:
                    continue
                if j in lexical_tgt:
                    continue
                pred_pairs.discard((i, current_t))
                lexical_tgt.discard(current_t)
                pred_pairs.add((i, j))
                lexical_assignments[i] = j
                lexical_tgt.add(j)
                continue

            if not candidate_is_mnn or j in lexical_tgt:
                continue

            pred_pairs.add((i, j))
            lexical_assignments[i] = j
            lexical_tgt.add(j)
    
    pred_pairs_iri = set()
    for s, t in pred_pairs:
        pred_pairs_iri.add((src_graph.node_iris[s], tgt_graph.node_iris[t]))
        
    metrics = evaluator.evaluate(pred_pairs_iri)
    print(f"  [Eval] P={metrics['Precision']:.4f}, R={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
    return metrics


def evaluate_embeddings_only(src_graph, tgt_graph, h_src, h_tgt, evaluator, label="Embedding-only"):
    if not evaluator:
        return None
    with torch.no_grad():
        src_emb = F.normalize(h_src, p=2, dim=1)
        tgt_emb = F.normalize(h_tgt, p=2, dim=1)
        sim_matrix = torch.mm(src_emb, tgt_emb.t())
        val_s, idx_s = sim_matrix.max(dim=1)
        val_t, idx_t = sim_matrix.max(dim=0)
        pred_pairs = []
        for i in range(sim_matrix.size(0)):
            j = idx_s[i].item()
            if idx_t[j].item() == i:
                pred_pairs.append((i, j))
    pred_pairs_iri = {(src_graph.node_iris[s], tgt_graph.node_iris[t]) for s, t in pred_pairs}
    metrics = evaluator.evaluate(pred_pairs_iri)
    print(f"  [{label}] P={metrics['Precision']:.4f}, R={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
    return metrics

# --- Graph-level utilities ---
def compute_graph_embedding(graph, text_proj, device):
    x = text_proj(graph.x_text.to(device))
    if x.dim() == 1:
        x = x.unsqueeze(0)
    graph_embedding = x.mean(dim=0, keepdim=True)
    return graph_embedding, x


# --- Model Runner ---
def run_model(model, text_proj, graph, device, adapter=None):
    graph_embedding, x_in = compute_graph_embedding(graph, text_proj, device)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    # Use a dummy domain ID for the forward pass since we override domain_emb_input
    d_id = torch.zeros(1, dtype=torch.long, device=device) 
    
    edge_prop = getattr(graph, "edge_property_id", None)
    if edge_prop is not None:
        edge_prop = edge_prop.to(device)

    if adapter:
        # Dual Prompt Mode: Sum of Outputs
        p_uni, p_mix = adapter(graph_embedding)
        
        # Pass 1: Unified Prompt
        h_uni = model(
            x_in,
            graph.edge_index.to(device),
            graph.edge_type.to(device),
            batch,
            d_id,
            domain_emb_input=p_uni,
            edge_property_id=edge_prop
        )
        
        # Pass 2: Mixed Prompt
        h_mix = model(
            x_in,
            graph.edge_index.to(device),
            graph.edge_type.to(device),
            batch,
            d_id,
            domain_emb_input=p_mix,
            edge_property_id=edge_prop
        )
        
        return (h_uni + h_mix) / 2.0

    else:
        # No adapter case
        # Here we use the real domain ID
        domain_id = resolve_graph_domain_id(graph)
        d_id_real = torch.tensor([domain_id], dtype=torch.long, device=device)
        return model(
            x_in,
            graph.edge_index.to(device),
            graph.edge_type.to(device),
            batch,
            d_id_real,
            edge_property_id=edge_prop
        )


def get_domain_id_mapping(graphs_dir: str) -> Dict[int, str]:
    mapping = {}
    if not os.path.exists(graphs_dir):
        return mapping
    
    files = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
    for f in files:
        path = os.path.join(graphs_dir, f)
        try:
            # Only load metadata to speed up
            # Torch load can be slow for big graphs, but we need domain_id
            g = torch.load(path, map_location='cpu')
            if isinstance(g, dict):
                d_id = g.get('domain_id')
                if isinstance(d_id, torch.Tensor):
                    d_id = int(d_id.item())
                elif isinstance(d_id, (int, float)):
                     d_id = int(d_id)
                
                # Extract name from filename if not in dict
                name = g.get('name', f.replace('.pt', ''))
                if d_id is not None:
                    mapping[d_id] = name
        except Exception:
            pass
    return mapping


# --- Main Finetune Loop ---
def finetune(config_path="config/finetune.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config.get('train', {})
    csls_threshold = train_cfg.get('csls_threshold', 0.5)
    structure_threshold = train_cfg.get('structure_threshold', 0.05)
    high_conf_topk = train_cfg.get('high_conf_topk', 100)
    max_new_pairs = train_cfg.get('max_new_pairs', 200)
    min_precision_to_expand = train_cfg.get('min_precision_to_expand', 0.95)
    csls_adaptive = train_cfg.get('csls_adaptive', True)
    csls_std_multiplier = train_cfg.get('csls_std_multiplier', 1.5)
    csls_min_threshold = train_cfg.get('csls_min_threshold', -0.05)
    csls_max_pairs = train_cfg.get('csls_max_pairs_per_round', 2000)
    csls_min_pairs = train_cfg.get('csls_min_pairs_per_round', 25)
    high_conf_min_score = train_cfg.get('high_conf_min_score', 0.3)
    high_conf_margin = train_cfg.get('high_conf_margin', 0.05)
    high_conf_max_pairs = train_cfg.get('high_conf_max_pairs_per_round', 1000)
    warmup_rounds = train_cfg.get('warmup_rounds', 3)
    embedding_topk = train_cfg.get('embedding_topk', 0)
    embedding_score_threshold = train_cfg.get('embedding_score_threshold', 0.5)
    embedding_max_pairs = train_cfg.get('embedding_max_pairs_per_round', 0)
    embedding_margin = train_cfg.get('embedding_margin', 0.1)
    anchor_prune_threshold = train_cfg.get('anchor_prune_threshold', 0.6)
    anchor_replace_margin = train_cfg.get('anchor_replace_margin', 0.02)
    lexical_override_threshold = train_cfg.get('lexical_override_threshold', 0.75)
    lexical_guard_rounds = train_cfg.get('lexical_guard_rounds', 2)
    lexical_min_hits = train_cfg.get('lexical_min_hits', 10)
    lexical_min_precision = train_cfg.get('lexical_min_precision', 0.95)

    
    # 1. Load Data
    print("Loading graphs...")
    src_graph = torch.load(config['data']['src_graph'])
    tgt_graph = torch.load(config['data']['tgt_graph'])
    
    if isinstance(src_graph, dict): src_graph = Data(**src_graph)
    if isinstance(tgt_graph, dict): tgt_graph = Data(**tgt_graph)
    
    if getattr(src_graph, "num_nodes", None) is None: src_graph.num_nodes = src_graph.x_text.size(0)
    if getattr(tgt_graph, "num_nodes", None) is None: tgt_graph.num_nodes = tgt_graph.x_text.size(0)
    
    src_domain_id = resolve_graph_domain_id(src_graph)
    tgt_domain_id = resolve_graph_domain_id(tgt_graph)
    property_vocab = load_property_vocab(config['data']['output_dir'])
    num_properties = max(property_vocab.values()) + 1 if property_vocab else 1
    none_prop_id = property_vocab.get("__NONE__", 0)

    def ensure_edge_property_ids(graph):
        edge_prop = getattr(graph, "edge_property_id", None)
        if edge_prop is not None:
            if not torch.is_tensor(edge_prop):
                edge_prop = torch.tensor(edge_prop, dtype=torch.long)
        else:
            raw_props = getattr(graph, "edge_property", None)
            if raw_props is None:
                num_edges = graph.edge_index.shape[1]
                edge_prop = torch.zeros(num_edges, dtype=torch.long)
            else:
                mapped = [property_vocab.get(prop, none_prop_id) if prop else none_prop_id for prop in raw_props]
                edge_prop = torch.tensor(mapped, dtype=torch.long)
        graph.edge_property_id = edge_prop

    ensure_edge_property_ids(src_graph)
    ensure_edge_property_ids(tgt_graph)
    
    # Pre-compute Adjacency (for Jaccard/Structural Boosting)
    adj_src = get_adj_set(src_graph)
    adj_tgt = get_adj_set(tgt_graph)
    
    # 2. Load Model
    print("Loading Pretrained Model with LoRA...")
    hidden_dim = config['model']['hidden_dim']
    
    model = OntoAlignEncoder(
        text_dim=hidden_dim, 
        hidden_dim=hidden_dim,
        num_relations=config['model']['num_relations'],
        num_domains=config['model']['num_domains'],
        num_layers=2,
        use_lora=True, 
        lora_rank=16,
        num_properties=num_properties
    ).to(device)
    
    text_proj = nn.Linear(config['model']['input_text_dim'], hidden_dim).to(device)
    
    ckpt = torch.load(config['model']['pretrained_path'], map_location=device)
    state_dict = ckpt['model_state_dict']
    
    if 'rel_emb.weight' in state_dict and state_dict['rel_emb.weight'].shape != model.rel_emb.weight.shape:
        print(f"Resizing rel_emb: {state_dict['rel_emb.weight'].shape} -> {model.rel_emb.weight.shape}")
        state_dict['rel_emb.weight'] = state_dict['rel_emb.weight'][:model.num_relations]

    # Handle domain_emb mismatch (100 -> 8)
    if 'domain_emb.weight' in state_dict and state_dict['domain_emb.weight'].shape != model.domain_emb.weight.shape:
        print(f"Resizing domain_emb: {state_dict['domain_emb.weight'].shape} -> {model.domain_emb.weight.shape}")
        # Only take the first N domains (assuming they are contiguous 0..7)
        state_dict['domain_emb.weight'] = state_dict['domain_emb.weight'][:model.domain_emb.num_embeddings]

    model.load_state_dict(state_dict, strict=False)
    if 'text_proj_state_dict' in ckpt:
        text_proj.load_state_dict(ckpt['text_proj_state_dict'])
        
    # 3. Setup Adapters & Optimizer
    print("Setting up Domain Prompt Adapters (Mixed Prompt Mode) and Unfreezing GNN output layer...")
    
    # Helper to print weights
    def print_adapter_weights(adapter, name, domain_map, graph_embedding):
        weights, indices = adapter.get_top_weights(graph_embedding, top_k=5)
        w_str = ", ".join([f"{domain_map.get(idx, f'ID_{idx}')}: {w:.4f}" for w, idx in zip(weights, indices)])
        print(f"[DEBUG] {name} Mixed Weights: {w_str}")

    domain_map = get_domain_id_mapping(config['data'].get('output_dir', 'graphs'))
    
    # Force load all domain names if not fully populated
    # This is a fallback in case build_graphs didn't save names or IDs are mismatched
    if len(domain_map) < config['model']['num_domains']:
        print(f"Warning: Found only {len(domain_map)} domain names, but model expects {config['model']['num_domains']}.")
    
    pretrained_d_embs = model.domain_emb.weight.data
    src_adapter = MixedDomainPromptAdapter(pretrained_d_embs).to(device)
    tgt_adapter = MixedDomainPromptAdapter(pretrained_d_embs).to(device)
    
    with torch.no_grad():
        src_graph_embedding, _ = compute_graph_embedding(src_graph, text_proj, device)
        tgt_graph_embedding, _ = compute_graph_embedding(tgt_graph, text_proj, device)
        src_graph_embedding = src_graph_embedding.detach()
        tgt_graph_embedding = tgt_graph_embedding.detach()
    
    # Initial weights
    print_adapter_weights(src_adapter, "Source", domain_map, src_graph_embedding)
    print_adapter_weights(tgt_adapter, "Target", domain_map, tgt_graph_embedding)

    # --- Unfreeze Logic (Addressing Point 1) ---
    for p in model.parameters(): p.requires_grad = False
    for p in text_proj.parameters(): p.requires_grad = False
    
    # Unfreeze GNN Layer 2 (Last Layer)
    for p in model.gnn_layers[-1].parameters():
        p.requires_grad = True
        
    # Unfreeze LoRA
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            
    trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                       [p for p in src_adapter.parameters() if p.requires_grad] + \
                       [p for p in tgt_adapter.parameters() if p.requires_grad]
    
    print(f"Trainable params count: {sum(p.numel() for p in trainable_params)}")
    
    # Use AdamW for stable training
    # Increase learning rate for adapters to encourage weight updates
    # Separate parameter groups
    adapter_params = [p for p in src_adapter.parameters()] + [p for p in tgt_adapter.parameters()]
    model_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': model_params, 'lr': 5e-4},
        {'params': adapter_params, 'lr': 1e-2} # Higher LR for adapters
    ], weight_decay=1e-4)
    
    # 4. Evaluator
    evaluator = None
    if os.path.exists(config['data']['pair_file']):
        evaluator = AlignmentEvaluator(config['data']['pair_file'])
        
    # 5. Lexical Seeds
    lexical_seeds = generate_lexical_seeds(src_graph, tgt_graph)
    lexical_seeds = enforce_bijection(lexical_seeds)
    lexical_seed_set = set(lexical_seeds)

    current_anchors = []
    anchor_origin = {}
    anchor_score = {}
    source_anchor_map = {}
    target_anchor_map = {}

    def remove_anchor_pair(pair):
        if pair not in anchor_origin:
            return
        try:
            current_anchors.remove(pair)
        except ValueError:
            pass
        anchor_origin.pop(pair, None)
        anchor_score.pop(pair, None)
        s, t = pair
        source_anchor_map.pop(s, None)
        target_anchor_map.pop(t, None)

    lexical_add_batch = []
    lexical_override_enabled = False

    def add_anchor_pair(pair, score, origin):
        s, t = pair
        existing = source_anchor_map.get(s)
        if existing:
            existing_origin = anchor_origin.get(existing)
            if existing_origin == "lexical":
                if origin == "lexical":
                    return False
                if (not lexical_override_enabled) or score < lexical_override_threshold:
                    return False
            else:
                if score <= anchor_score.get(existing, float('-inf')) + anchor_replace_margin:
                    return False
            remove_anchor_pair(existing)
        existing_t = target_anchor_map.get(t)
        if existing_t:
            existing_origin_t = anchor_origin.get(existing_t)
            if existing_origin_t == "lexical":
                if origin == "lexical":
                    return False
                if (not lexical_override_enabled) or score < lexical_override_threshold:
                    return False
            else:
                if score <= anchor_score.get(existing_t, float('-inf')) + anchor_replace_margin:
                    return False
            remove_anchor_pair(existing_t)
        current_anchors.append(pair)
        anchor_origin[pair] = origin
        anchor_score[pair] = score
        source_anchor_map[s] = pair
        target_anchor_map[t] = pair
        return True

    def flush_lexical_batch(final=False):
        if not lexical_add_batch:
            return
        for pair in lexical_add_batch:
            add_anchor_pair(pair, score=1.0, origin="lexical")
        lexical_add_batch.clear()
        if final:
            return

    for pair in lexical_seeds:
        lexical_add_batch.append(pair)
    flush_lexical_batch(final=True)
    
    # 6. Zero-shot Eval
    print("\n=== Zero-shot Evaluation (Hybrid) ===")
    model.eval()
    h_src = run_model(model, text_proj, src_graph, device, adapter=src_adapter)
    h_tgt = run_model(model, text_proj, tgt_graph, device, adapter=tgt_adapter)
    hybrid_metrics = evaluate_hybrid(src_graph, tgt_graph, h_src, h_tgt, lexical_seeds, evaluator)
    evaluate_embeddings_only(src_graph, tgt_graph, h_src, h_tgt, evaluator, label="Embedding-only (Zero-shot)")
    
    # 7. Training Loop (InfoNCE + CSLS Mining)
    rounds = 20
    epochs = 10
    
    print("\n=== Iterative Bootstrapping (LoRA + High-Confidence Promotion) ===")

    for r in range(rounds):
        allow_heatup = r < warmup_rounds
        allow_expansion = True
        gate_precision = None
        if not allow_heatup and hybrid_metrics:
            gate_precision = hybrid_metrics.get("Precision", 1.0)
            if gate_precision < min_precision_to_expand:
                allow_expansion = False
                print(
                    f"  Round {r+1}: Precision {gate_precision:.4f} < "
                    f"{min_precision_to_expand:.2f}, pause candidate expansion."
                )

        lexical_non_seed_count = sum(1 for p in current_anchors if anchor_origin.get(p) != "lexical")
        lexical_override_enabled = (
            (r + 1) >= lexical_guard_rounds
            and lexical_non_seed_count >= lexical_min_hits
            and hybrid_metrics is not None
            and hybrid_metrics.get("Precision", 0.0) >= lexical_min_precision
        )

        # E-Step: Generate Candidates using CSLS + Structural Verification
        model.eval()
        with torch.no_grad():
            h_src = run_model(model, text_proj, src_graph, device, adapter=src_adapter)
            h_tgt = run_model(model, text_proj, tgt_graph, device, adapter=tgt_adapter)

            anchor_map = {s: pair[1] for s, pair in source_anchor_map.items()}

            csls_raw = []
            new_csls = []
            new_high_conf = []
            new_embedding = []
            added_this_round = 0
            budget = max_new_pairs if max_new_pairs is not None else None

            def budget_reached():
                return budget is not None and added_this_round >= budget

            if allow_expansion:
                csls_raw = get_csls_mnn(
                    h_src,
                    h_tgt,
                    threshold=csls_threshold,
                    k=10,
                    adaptive=csls_adaptive,
                    std_multiplier=csls_std_multiplier,
                    min_threshold=csls_min_threshold,
                    min_pairs=csls_min_pairs
                )
                csls_pairs = [(i, j) for i, j, _ in csls_raw]
                struct_filtered = []
                if csls_pairs:
                    struct_scores = compute_jaccard_similarity(csls_pairs, adj_src, adj_tgt, anchor_map)
                    struct_filtered = [
                        ((i, j, score), struct_score)
                        for (i, j, score), struct_score in zip(csls_raw, struct_scores.tolist())
                        if struct_score >= structure_threshold
                    ]

                for (i, j, score), _ in struct_filtered:
                    if budget_reached():
                        break
                    if add_anchor_pair((i, j), score, origin="csls"):
                        new_csls.append((i, j))
                        added_this_round += 1
                        limit_hit = csls_max_pairs and len(new_csls) >= csls_max_pairs
                        if limit_hit or budget_reached():
                            break

                anchor_map_for_struct = {s: pair[1] for s, pair in source_anchor_map.items()}

                if not budget_reached():
                    high_conf_pairs = get_high_conf_candidates(
                        h_src,
                        h_tgt,
                        topk=high_conf_topk,
                        min_score=high_conf_min_score,
                        margin=high_conf_margin
                    )
                    if high_conf_pairs:
                        hc_pairs = [(i, j) for i, j, _ in high_conf_pairs]
                        struct_scores = compute_jaccard_similarity(hc_pairs, adj_src, adj_tgt, anchor_map_for_struct)
                        for (i, j, score), struct_score in zip(high_conf_pairs, struct_scores.tolist()):
                            if struct_score < structure_threshold:
                                continue
                            if add_anchor_pair((i, j), score, origin="high_conf"):
                                new_high_conf.append((i, j))
                                added_this_round += 1
                                limit_hit = high_conf_max_pairs and len(new_high_conf) >= high_conf_max_pairs
                                if limit_hit or budget_reached():
                                    break
                    if new_high_conf:
                        anchor_map_for_struct = {s: pair[1] for s, pair in source_anchor_map.items()}

                if not budget_reached() and embedding_topk > 0 and embedding_score_threshold > 0:
                    embedding_candidates = get_embedding_topk_candidates(
                        h_src,
                        h_tgt,
                        topk=embedding_topk,
                        min_score=embedding_score_threshold,
                        margin=embedding_margin
                    )
                    if embedding_candidates:
                        emb_pairs = [(i, j) for i, j, _ in embedding_candidates]
                        struct_scores = compute_jaccard_similarity(
                            emb_pairs, adj_src, adj_tgt, anchor_map_for_struct
                        )
                        for (i, j, score), struct_score in zip(embedding_candidates, struct_scores.tolist()):
                            if struct_score < structure_threshold:
                                continue
                            if add_anchor_pair((i, j), score, origin="embedding"):
                                new_embedding.append((i, j))
                                added_this_round += 1
                                limit_hit = embedding_max_pairs and len(new_embedding) >= embedding_max_pairs
                                if limit_hit or budget_reached():
                                    break
                    if new_embedding:
                        anchor_map_for_struct = {s: pair[1] for s, pair in source_anchor_map.items()}

                print(f"  Round {r+1}: CSLS candidates={len(csls_raw)}, after structure={len(struct_filtered)}")
                print(
                    f"  Round {r+1}: New CSLS pairs={len(new_csls)}, "
                    f"new High-Conf pairs={len(new_high_conf)}, new Emb pairs={len(new_embedding)}"
                )
            else:
                print(f"  Round {r+1}: Expansion skipped (precision gate).")

            valid_new_pairs = new_csls + new_high_conf + new_embedding

            if anchor_prune_threshold > 0:
                pruned, removed_pairs = prune_noisy_anchors(list(current_anchors), h_src, h_tgt, lexical_seed_set, anchor_prune_threshold)
                if removed_pairs:
                    for pair in removed_pairs:
                        remove_anchor_pair(pair)
                    current_anchors = pruned

            new_count = len([p for p in current_anchors if p not in lexical_seed_set])
            print(f"  Round {r+1}: Total new trusted pairs: {len(valid_new_pairs)}")
            print(f"  Round {r+1}: Training on {len(current_anchors)} pairs (Starts: {len(lexical_seeds)})")

        # M-Step: Train with Margin Ranking Loss
        model.train()
        src_adapter.train()
        tgt_adapter.train()

        # Use current anchors as training pairs
        train_pairs = current_anchors
        train_src_idx = torch.tensor([p[0] for p in train_pairs], device=device)
        train_tgt_idx = torch.tensor([p[1] for p in train_pairs], device=device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            h_s = run_model(model, text_proj, src_graph, device, adapter=src_adapter)
            h_t = run_model(model, text_proj, tgt_graph, device, adapter=tgt_adapter)
            
            # Sample Negatives using Hard Negative Mining
            curr_src_emb = h_s[train_src_idx]
            curr_tgt_emb = h_t[train_tgt_idx]

            # Hard Negative Mining: Select negatives that are close but not the positive
            curr_neg_emb = get_hard_negatives(curr_src_emb, h_t, train_tgt_idx, device, beta=20)
            
            # InfoNCE-style Contrastive Loss
            # Normalize embeddings
            curr_src_emb = F.normalize(curr_src_emb, p=2, dim=1)
            curr_tgt_emb = F.normalize(curr_tgt_emb, p=2, dim=1)
            curr_neg_emb = F.normalize(curr_neg_emb, p=2, dim=1)

            # Positive similarity
            pos_sim = torch.sum(curr_src_emb * curr_tgt_emb, dim=1)  # [batch]

            # Negative similarity
            neg_sim = torch.sum(curr_src_emb * curr_neg_emb, dim=1)  # [batch]

            # InfoNCE loss: -log(exp(pos_sim/temp) / (exp(pos_sim/temp) + exp(neg_sim/temp)))
            temperature = 0.1
            logits = torch.stack([pos_sim, neg_sim], dim=1) / temperature  # [batch, 2]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)  # positives are index 0

            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            if epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
                print_adapter_weights(src_adapter, f"Source (Round {r+1})", domain_map, src_graph_embedding)
                print_adapter_weights(tgt_adapter, f"Target (Round {r+1})", domain_map, tgt_graph_embedding)
                
        # Evaluate
        if evaluator:
            model.eval()
            h_s_eval = run_model(model, text_proj, src_graph, device, adapter=src_adapter)
            h_t_eval = run_model(model, text_proj, tgt_graph, device, adapter=tgt_adapter)
            hybrid_metrics = evaluate_hybrid(src_graph, tgt_graph, h_s_eval, h_t_eval, lexical_seeds, evaluator)
            evaluate_embeddings_only(
                src_graph,
                tgt_graph,
                h_s_eval,
                h_t_eval,
                evaluator,
                label=f"Embedding-only (Round {r+1})"
            )

    torch.save(model.state_dict(), "checkpoints/finetuned_infonce.pt")
    print("Saved to checkpoints/finetuned_infonce.pt")

if __name__ == "__main__":
    finetune()
