import os
import yaml
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from dataset import OntologyGraphDataset
from model import OntoAlignEncoder, edge_prediction_loss, masked_recon_loss, contrastive_loss
from collections import defaultdict, deque

# --- Memory Bank Class ---
class CrossGraphMemoryBank:
    def __init__(self, max_history=16, device='cpu'):
        self.max_history = max_history
        # Stores list of dicts: {"h": tensor, "lexical": list, "graph_id": int}
        self.history = deque()
        self.device = device

    def update(self, h, node_lexical, graph_id):
        # Store features on CPU to save GPU memory
        # h: [N, d]
        self.history.append({
            "h": h.detach().cpu(),
            "node_lexical": node_lexical, # list of lists
            "graph_id": int(graph_id)
        })
        if len(self.history) > self.max_history:
            self.history.popleft()

    def get_contents(self):
        return list(self.history)

def normalize_term(term) -> str:
    if term is None:
        return ""
    return str(term).strip().lower()

def _iter_terms(terms):
    if terms is None:
        return []
    # Base case: string
    if isinstance(terms, str):
        # Simple tokenization strategy: split by _ and space
        # This helps matching composite terms like "Conference_Paper" with "Paper"
        parts = terms.replace("_", " ").split()
        if len(parts) > 1:
            return [terms] + parts
        return [terms]
    
    # Handle numpy/tensor conversion first
    if hasattr(terms, "tolist"):
        terms = terms.tolist()
    
    # If it's a collection (list, tuple, set, or converted array)
    if isinstance(terms, (list, tuple, set)):
        # Recursively flatten
        result = []
        for item in terms:
            result.extend(_iter_terms(item))
        return result
        
    # Fallback for scalars
    return [terms]

def _prepare_node_lexical(node_lexical, graph_ptr, num_nodes):
    if node_lexical is None:
        return None
    if hasattr(node_lexical, "tolist") and not isinstance(node_lexical, (list, tuple)):
        node_lexical = node_lexical.tolist()
    if isinstance(node_lexical, (list, tuple)):
        # Flat list handling if needed, but simplified for batch_size=1 usually
        pass
    if isinstance(node_lexical, tuple):
        node_lexical = list(node_lexical)
    return node_lexical

def _bin_tensor(values, bins=10):
    if values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.long)
    vmax = values.max()
    if vmax <= 0:
        return torch.zeros_like(values, dtype=torch.long)
    scaled = values / vmax
    return torch.clamp((scaled * (bins - 1)).long(), 0, bins - 1)

def _select_structural_nodes(x_text, edge_index, batch, max_nodes=2048, start_cluster_id=0):
    num_nodes = x_text.size(0)
    if num_nodes == 0:
        return []
    device = x_text.device
    if edge_index.numel() > 0:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float().to(device)
    else:
        deg = torch.zeros(num_nodes, device=device)
    text_norm = x_text.norm(p=2, dim=1)
    deg_bin = torch.clamp(deg.long(), max=10)
    norm_bin = _bin_tensor(text_norm, bins=8)
    buckets = defaultdict(lambda: defaultdict(list))
    for idx in range(num_nodes):
        key = (int(deg_bin[idx].item()), int(norm_bin[idx].item()))
        buckets[key][int(batch[idx].item())].append(idx)
    selected = []
    cluster_id = start_cluster_id
    for nodes_by_graph in buckets.values():
        graph_ids = list(nodes_by_graph.keys())
        if len(graph_ids) < 2:
            continue
        random.shuffle(graph_ids)
        cluster_nodes = []
        for g in graph_ids:
            node_list = nodes_by_graph[g]
            if not node_list:
                continue
            random_idx = random.choice(node_list)
            cluster_nodes.append((random_idx, g))
        if len(cluster_nodes) < 2:
            continue
        available = max_nodes - len(selected)
        if available < 2:
            break
        if len(cluster_nodes) > available:
            cluster_nodes = cluster_nodes[:available]
        for node_idx, g in cluster_nodes:
            selected.append((node_idx, g, cluster_id))
        cluster_id += 1
        if len(selected) >= max_nodes:
            break
    return selected

# --- Updated Selection Logic for Memory Bank ---
def _select_cross_graph_nodes(node_lexical, current_graph_id, max_nodes=2048, memory_bank_contents=None):
    term_to_nodes = defaultdict(list)
    
    # 1. Map current batch nodes
    # We use (node_index_in_tensor, graph_id)
    for idx, terms in enumerate(node_lexical):
        for term in _iter_terms(terms):
            norm = normalize_term(term)
            if norm:
                term_to_nodes[norm].append((idx, current_graph_id))
                
    # DEBUG: Check if we populated anything
    if len(term_to_nodes) == 0 and len(node_lexical) > 0:
         # Try to see why
         if hasattr(node_lexical, "tolist") and hasattr(node_lexical, "shape"):
             sample_raw = node_lexical[0]
         elif isinstance(node_lexical, list) and len(node_lexical) > 0:
             sample_raw = node_lexical[0]
         else:
             sample_raw = "Unknown"
             
         sample_iter = _iter_terms(sample_raw)
         print(f"DEBUG: term_to_nodes EMPTY! Graph {current_graph_id} Raw[0]: {sample_raw}, Iterated: {sample_iter}")

    # 2. Map memory bank nodes
    bank_offset = len(node_lexical) # The bank tensor will be concatenated AFTER current nodes
    bank_h_list = []
    
    if memory_bank_contents:
        current_offset = 0
        for entry in memory_bank_contents:
            b_h = entry["h"]
            b_lex = entry["node_lexical"]
            b_gid = entry["graph_id"]
            
            # Skip if it's the exact same graph (avoid self-contrast if needed, though unlikely with unique IDs)
            if b_gid == current_graph_id:
                continue
                
            bank_h_list.append(b_h)
            
            # Iterate bank nodes
            for idx, terms in enumerate(b_lex):
                abs_idx = bank_offset + current_offset + idx
                for term in _iter_terms(terms):
                    norm = normalize_term(term)
                    # Optimization: Only add if this term already exists in current batch
                    # (Because we only care about pairs where at least one node is from current batch)
                    if norm and norm in term_to_nodes: 
                        term_to_nodes[norm].append((abs_idx, b_gid))
            
            current_offset += b_h.size(0)

    # DEBUG: Verify matching
    if memory_bank_contents and len(bank_h_list) > 0:
        # Check for true CROSS-GRAPH matches (different graph IDs)
        cross_matches = 0
        for v in term_to_nodes.values():
            gids = set(item[1] for item in v)
            if len(gids) > 1:
                cross_matches += 1
        
        if cross_matches == 0:
             # Only print occasionally to avoid spam, e.g. for the first graph in a batch sequence logic or random
             if random.random() < 0.2: 
                 print(f"DEBUG: No CROSS-GRAPH matches for Graph {current_graph_id}. Bank size: {len(bank_h_list)}. Internal matches: {sum(1 for v in term_to_nodes.values() if len(v)>1)}")
                 if len(term_to_nodes) > 0:
                     sample_term = list(term_to_nodes.keys())[0]
                     print(f"DEBUG: Sample current term: '{sample_term}'")
                     
                     # Scan bank manually
                     found_in_bank = False
                     for entry in memory_bank_contents:
                         if entry['graph_id'] == current_graph_id: continue
                         for b_terms in entry['node_lexical']:
                             for t in _iter_terms(b_terms):
                                 if normalize_term(t) == sample_term:
                                     found_in_bank = True
                                     print(f"DEBUG: Term '{sample_term}' FOUND in Bank Graph {entry['graph_id']} but not matched!")
                                     break
                         if found_in_bank: break
                     if not found_in_bank:
                         print(f"DEBUG: Term '{sample_term}' NOT found in any Bank Graph.")

    # If no matches found or no bank
    if not bank_h_list and len(term_to_nodes) == 0:
        return [], None

    # Build Clusters
    selected = []
    assigned_nodes = set() # indices in the combined tensor
    cluster_id = 0
    max_nodes = max(2, int(max_nodes))

    items = list(term_to_nodes.items())
    if len(items) > 1:
        ordering = torch.randperm(len(items))
    else:
        ordering = torch.arange(len(items), dtype=torch.long)

    for order_idx in ordering.tolist():
        term, entries = items[order_idx]
        if len(entries) < 2:
            continue
        
        # Check if this cluster involves the current graph
        # entries is list of (abs_idx, gid)
        has_current = any(gid == current_graph_id for _, gid in entries)
        if not has_current:
            continue

        if len(entries) > 1:
            entry_order = torch.randperm(len(entries))
        else:
            entry_order = torch.arange(len(entries), dtype=torch.long)
            
        seen_graphs = set()
        cluster_nodes = []
        
        for pos in entry_order.tolist():
            node_idx, g_id = entries[pos]
            if node_idx in assigned_nodes or g_id in seen_graphs:
                continue
            seen_graphs.add(g_id)
            cluster_nodes.append((node_idx, g_id))
            
        if len(cluster_nodes) < 2:
            continue

        available = max_nodes - len(selected)
        if available < 2:
            break
        if len(cluster_nodes) > available:
            cluster_nodes = cluster_nodes[:available]
            if len(cluster_nodes) < 2:
                break
                
        for node_idx, g_id in cluster_nodes:
            assigned_nodes.add(node_idx)
            selected.append((node_idx, g_id, cluster_id))
        cluster_id += 1
        
        if len(selected) >= max_nodes:
            break
            
    # If we used bank, return the concatenated bank tensor
    bank_tensor = torch.cat(bank_h_list, dim=0) if bank_h_list else None
    return selected, bank_tensor

def _supervised_cross_graph_loss(
    h,
    selected,
    temperature,
    hard_negative_k=0,
    hard_negative_margin=0.0,
    hard_negative_weight=0.0
):
    device = h.device
    node_indices = torch.tensor([item[0] for item in selected], device=device, dtype=torch.long)
    graph_ids = torch.tensor([item[1] for item in selected], device=device, dtype=torch.long)
    cluster_ids = torch.tensor([item[2] for item in selected], device=device, dtype=torch.long)

    h_sel = F.normalize(h[node_indices], p=2, dim=1)
    logits = torch.mm(h_sel, h_sel.t()) / temperature

    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    pos_mask = torch.zeros_like(logits, dtype=torch.float32)
    for cid in cluster_ids.unique().tolist():
        cluster_positions = torch.nonzero(cluster_ids == cid, as_tuple=False).view(-1)
        if cluster_positions.numel() < 2:
            continue
        g_ids = graph_ids[cluster_positions]
        diff_graph = g_ids.unsqueeze(0) != g_ids.unsqueeze(1)
        cluster_mask = diff_graph & (~torch.eye(cluster_positions.numel(), dtype=torch.bool, device=device))
        if cluster_mask.any():
            pos_mask[cluster_positions[:, None], cluster_positions[None, :]] = cluster_mask.float()

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not torch.any(valid):
        return h.new_tensor(0.0)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1e-8)
    loss = -(mean_log_prob_pos[valid]).mean()
    
    hard_loss = h.new_tensor(0.0)
    if hard_negative_k > 0 and hard_negative_margin > 0 and hard_negative_weight > 0:
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask.fill_diagonal_(False)
        pos_bool = pos_mask > 0
        neg_mask &= ~pos_bool
        diff_graph = graph_ids.unsqueeze(0) != graph_ids.unsqueeze(1)
        neg_mask &= diff_graph
        if neg_mask.any():
            neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
            top_k = min(hard_negative_k, neg_logits.size(1))
            top_neg, _ = torch.topk(neg_logits, k=top_k, dim=1)
            pos_strength = (logits * pos_mask).sum(dim=1) / pos_counts.clamp(min=1e-8)
            pos_strength = pos_strength.unsqueeze(1)
            penalties = F.relu(hard_negative_margin + top_neg - pos_strength)
            finite_mask = torch.isfinite(top_neg)
            valid_mask = valid.unsqueeze(1) & finite_mask
            if valid_mask.any():
                hard_loss = penalties[valid_mask].mean()
    return loss + hard_negative_weight * hard_loss

def cross_graph_contrastive_loss(
    h,
    batch_assign,
    node_lexical,
    edge_index,
    temperature=0.1,
    max_pairs=2048,
    graph_ptr=None,
    hard_negative_k=0,
    hard_negative_margin=0.0,
    hard_negative_weight=0.0,
    memory_bank=None,
    current_graph_id=0
):
    node_lexical = _prepare_node_lexical(node_lexical, graph_ptr, h.size(0))
    
    selected = []
    
    # 1. Lexical Selection (with Memory Bank)
    bank_tensor = None
    if node_lexical is not None and len(node_lexical) == h.size(0):
        bank_contents = memory_bank.get_contents() if memory_bank else None
        lex_selected, bank_tensor = _select_cross_graph_nodes(
            node_lexical, 
            current_graph_id, 
            max_nodes=max_pairs,
            memory_bank_contents=bank_contents
        )
        selected.extend(lex_selected)
        cluster_offset = len({cid for _, _, cid in lex_selected}) if lex_selected else 0
    else:
        cluster_offset = 0

    # 2. Structural Selection (Intra-batch only for now, usually structural is for matching structure)
    # Structural selection relies on bucket hashes which might not align across graphs easily without seeds.
    # For batch_size=1, structural selection is mostly internal or skipped if <2 graphs.
    # We skip structural cross-graph if batch_size=1 because we don't have struct-seeds in bank.
    if edge_index is not None and bank_tensor is None: # Only if we have >1 graph in batch
        # Existing logic for multi-graph batch
        # But for batch_size=1, this returns empty usually
        struct_selected = _select_structural_nodes(
            h.detach(),
            edge_index,
            batch_assign,
            max_nodes=max_pairs,
            start_cluster_id=cluster_offset,
        )
        selected.extend(struct_selected)

    if len(selected) < 2:
        return h.new_tensor(0.0)
    
    # Prepare combined embedding tensor
    if bank_tensor is not None:
        # Move bank tensor to GPU
        bank_tensor = bank_tensor.to(h.device)
        full_h = torch.cat([h, bank_tensor], dim=0)
    else:
        full_h = h

    return _supervised_cross_graph_loss(
        full_h,
        selected,
        temperature,
        hard_negative_k=hard_negative_k,
        hard_negative_margin=hard_negative_margin,
        hard_negative_weight=hard_negative_weight,
    )


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path="config/pretrain.yaml"):
    print("Loading config...", flush=True)
    config = load_config(config_path)
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get('train', {}).get('batch_size', 4)
    epochs = config.get('train', {}).get('epochs', 10)
    lr = float(config.get('train', {}).get('lr', 1e-3))
    contrastive_max_nodes = int(config.get('train', {}).get('contrastive_max_nodes', 2048))
    cross_graph_max_pairs = int(config.get('train', {}).get('cross_graph_max_pairs', 2048))
    cross_hard_neg_k = config.get('train', {}).get('cross_graph_hard_neg_k', 0)
    cross_hard_neg_margin = config.get('train', {}).get('cross_graph_hard_neg_margin', 0.0)
    cross_hard_neg_weight = config.get('train', {}).get('cross_graph_hard_neg_weight', 0.0)
    use_amp = bool(config.get('train', {}).get('use_amp', True)) and device.type == "cuda"
    
    # Model dims
    input_text_dim = config.get('model', {}).get('input_text_dim', 768)
    hidden_dim = config.get('model', {}).get('hidden_dim', 256)
    text_dim = hidden_dim # Project to hidden_dim before model
    gnn_type = config.get('model', {}).get('gnn_type', 'rgcn')
    num_relations = config.get('model', {}).get('num_relations', 8)
    num_domains = config.get('model', {}).get('num_domains', 8)
    
    # Weights
    w_edge = config.get('model', {}).get('weights', {}).get('edge', 1.0)
    w_recon = config.get('model', {}).get('weights', {}).get('recon', 1.0)
    w_contrast = config.get('model', {}).get('weights', {}).get('contrast', 0.1)
    w_cross = config.get('model', {}).get('weights', {}).get('cross_graph', 0.5)
    
    # Dataset
    print("Initializing Dataset...", flush=True)
    graphs_dir = config['data']['output_dir']
    dataset = OntologyGraphDataset(graphs_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_properties = getattr(dataset, "num_properties", 1)
    
    # Text Projector
    text_projector = nn.Linear(input_text_dim, hidden_dim).to(device)
    
    # Model
    print("Initializing Model...", flush=True)
    model = OntoAlignEncoder(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        num_domains=num_domains,
        gnn_type=gnn_type,
        num_properties=num_properties
    ).to(device)
    
    # Initialize Memory Bank
    memory_bank = CrossGraphMemoryBank(max_history=16)
    
    optimizer = optim.Adam(list(model.parameters()) + list(text_projector.parameters()), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    
    print("Starting pre-training...", flush=True)
    model.train()
    
    for epoch in range(epochs):
        total_loss_val = 0
        total_l_edge = 0
        total_l_recon = 0
        total_l_contrast = 0
        total_l_cross = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            if num_batches % 10 == 0:
                print(f"Processing batch {num_batches}...", flush=True)
            
            # DEBUG: Check lexical content
            if epoch == 0 and num_batches == 0:
                node_lex = getattr(batch, "node_lexical", None)
                if node_lex is not None and len(node_lex) > 0:
                    print(f"DEBUG: First batch node_lexical[0]: {node_lex[0]} (Type: {type(node_lex[0])})")

            # For batch_size=1, batch.domain_id should be [1], containing the dataset idx
            if hasattr(batch, "domain_id"):
                # In PyG batching, simple attributes might be stacked.
                # We take the first one as the Graph ID if batch_size=1
                current_gid = int(batch.domain_id[0].item())
            else:
                current_gid = 0

            print(f"Batch info: {batch.x_text.size(0)} nodes, {batch.edge_index.size(1)} edges, GraphID: {current_gid}")

            edge_property = getattr(batch, "edge_property_id", None)
            optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                # Project text features: 768 -> 256
                batch.x_text = text_projector(batch.x_text)
                path_features = getattr(batch, "x_path", None)
                if path_features is not None:
                    batch.x_path = text_projector(path_features)
                    path_features = batch.x_path
                
                # --- Loss 1: Edge Prediction ---
                h_orig_nodes = model(
                    batch.x_text, 
                    batch.edge_index, 
                    batch.edge_type, 
                    batch.batch, 
                    batch.domain_id,
                    edge_property_id=edge_property,
                    x_path=path_features
                )
                
                l_edge = edge_prediction_loss(
                    h_orig_nodes, 
                    batch.edge_index, 
                    batch.edge_type, 
                    model.rel_emb,
                    property_emb=model.property_emb,
                    edge_property_id=edge_property,
                    num_neg=1
                )
                
                # --- Loss 2: Masked Neighborhood Reconstruction ---
                num_nodes = batch.x_text.size(0)
                mask_rate = 0.15
                perm = torch.randperm(num_nodes, device=device)
                mask_idx = perm[:int(num_nodes * mask_rate)]
                
                h_masked_nodes = model.encode_with_mask(
                    batch.x_text, 
                    batch.edge_index, 
                    batch.edge_type, 
                    batch.batch, 
                    batch.domain_id,
                    mask_idx,
                    edge_property_id=edge_property,
                    x_path=path_features
                )
                
                l_recon = masked_recon_loss(h_masked_nodes, h_orig_nodes, mask_idx)
                
                # --- Loss 3: Graph Contrastive Learning ---
                view1_mask = torch.rand(batch.edge_index.size(1), device=device) > 0.2
                edge_index_1 = batch.edge_index[:, view1_mask]
                edge_type_1 = batch.edge_type[view1_mask]
                edge_prop_1 = edge_property[view1_mask] if edge_property is not None else None
                x_1 = F.dropout(batch.x_text, p=0.2)
                
                h1_nodes = model(
                    x_1,
                    edge_index_1,
                    edge_type_1,
                    batch.batch,
                    batch.domain_id,
                    edge_property_id=edge_prop_1,
                    x_path=path_features
                )
                
                view2_mask = torch.rand(batch.edge_index.size(1), device=device) > 0.2
                edge_index_2 = batch.edge_index[:, view2_mask]
                edge_type_2 = batch.edge_type[view2_mask]
                edge_prop_2 = edge_property[view2_mask] if edge_property is not None else None
                x_2 = F.dropout(batch.x_text, p=0.2)
                
                h2_nodes = model(
                    x_2,
                    edge_index_2,
                    edge_type_2,
                    batch.batch,
                    batch.domain_id,
                    edge_property_id=edge_prop_2,
                    x_path=path_features
                )
                
                l_contrast = contrastive_loss(h1_nodes, h2_nodes, batch.batch, max_nodes=contrastive_max_nodes)
                
                # --- Loss 4: Cross-graph lexical alignment (WITH MEMORY BANK) ---
                node_lex = getattr(batch, "node_lexical", None)
                l_cross = cross_graph_contrastive_loss(
                    h_orig_nodes,
                    batch.batch,
                    node_lex,
                    batch.edge_index,
                    temperature=0.2,
                    max_pairs=cross_graph_max_pairs,
                    graph_ptr=getattr(batch, "ptr", None),
                    hard_negative_k=cross_hard_neg_k,
                    hard_negative_margin=cross_hard_neg_margin,
                    hard_negative_weight=cross_hard_neg_weight,
                    memory_bank=memory_bank,
                    current_graph_id=current_gid
                )
                
                # Update Memory Bank
                if node_lex is not None:
                    memory_bank.update(h_orig_nodes, node_lex, current_gid)
                
                # Total Loss
                loss = (
                    w_edge * l_edge
                    + w_recon * l_recon
                    + w_contrast * l_contrast
                    + w_cross * l_cross
                )
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss_val += loss.item()
            total_l_edge += l_edge.item()
            total_l_recon += l_recon.item()
            total_l_contrast += l_contrast.item()
            total_l_cross += l_cross.item()
            num_batches += 1
            
        avg_loss = total_loss_val / num_batches if num_batches > 0 else 0
        avg_l_edge = total_l_edge / num_batches if num_batches > 0 else 0
        avg_l_recon = total_l_recon / num_batches if num_batches > 0 else 0
        avg_l_contrast = total_l_contrast / num_batches if num_batches > 0 else 0
        
        avg_l_cross = total_l_cross / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_loss:.4f}, "
              f"Edge Loss: {avg_l_edge:.4f}, Recon Loss: {avg_l_recon:.4f}, "
              f"Contrast Loss: {avg_l_contrast:.4f}, Cross Loss: {avg_l_cross:.4f}")
        
        # Save model every 200 epochs
        if (epoch + 1) % 200 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_proj_state_dict': text_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, f"checkpoints/pretrained_model_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}.")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_proj_state_dict': text_projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "checkpoints/pretrained_model.pt")
    print("Model saved.")

if __name__ == "__main__":
    train()
