import os
import yaml
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from dataset import OntologyGraphDataset
from model import OntoAlignEncoder, edge_prediction_loss, masked_recon_loss, contrastive_loss
from collections import defaultdict

def normalize_term(term: str) -> str:
    if not isinstance(term, str):
        return ""
    return term.strip().lower()

def _iter_terms(terms):
    if terms is None:
        return []
    if isinstance(terms, (list, tuple, set)):
        return terms
    if hasattr(terms, "tolist"):
        return terms.tolist()
    return []

def _prepare_node_lexical(node_lexical, graph_ptr, num_nodes):
    if node_lexical is None:
        return None
    if hasattr(node_lexical, "tolist") and not isinstance(node_lexical, (list, tuple)):
        node_lexical = node_lexical.tolist()
    if isinstance(node_lexical, (list, tuple)):
        expected_nodes = None
        graph_count = None
        if graph_ptr is not None:
            if torch.is_tensor(graph_ptr):
                graph_count = int(graph_ptr.numel() - 1)
                expected_nodes = int(graph_ptr[-1].item())
            else:
                graph_count = len(graph_ptr) - 1
                expected_nodes = graph_ptr[-1]
        if graph_count is not None and len(node_lexical) == graph_count:
            flat = []
            for terms in node_lexical:
                if hasattr(terms, "tolist") and not isinstance(terms, (list, tuple)):
                    terms = terms.tolist()
                if isinstance(terms, (list, tuple)):
                    flat.extend(terms)
                else:
                    flat.append(terms)
            node_lexical = flat
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

def _select_cross_graph_nodes(node_lexical, batch_assign, max_nodes=2048):
    term_to_nodes = defaultdict(list)
    for idx, terms in enumerate(node_lexical):
        g_id = int(batch_assign[idx].item())
        for term in _iter_terms(terms):
            norm = normalize_term(term)
            if norm:
                term_to_nodes[norm].append((idx, g_id))

    if not term_to_nodes:
        return []

    items = list(term_to_nodes.items())
    if len(items) > 1:
        ordering = torch.randperm(len(items))
    else:
        ordering = torch.arange(len(items), dtype=torch.long)

    selected = []
    assigned_nodes = set()
    cluster_id = 0
    max_nodes = max(2, int(max_nodes))

    for order_idx in ordering.tolist():
        _, entries = items[order_idx]
        if len(entries) < 2:
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
    return selected

def _supervised_cross_graph_loss(h, selected, temperature):
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
    return loss

def cross_graph_contrastive_loss(
    h,
    batch_assign,
    node_lexical,
    edge_index,
    temperature=0.1,
    max_pairs=2048,
    graph_ptr=None,
):
    node_lexical = _prepare_node_lexical(node_lexical, graph_ptr, h.size(0))
    selected = []
    if node_lexical is not None and len(node_lexical) == h.size(0):
        lexical_selected = _select_cross_graph_nodes(node_lexical, batch_assign, max_nodes=max_pairs)
        selected.extend(lexical_selected)
        cluster_offset = len({cid for _, _, cid in lexical_selected}) if lexical_selected else 0
    else:
        cluster_offset = 0
    if edge_index is not None:
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
    return _supervised_cross_graph_loss(h, selected, temperature)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path="config/pretrain.yaml"):
    config = load_config(config_path)
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get('train', {}).get('batch_size', 4)
    epochs = config.get('train', {}).get('epochs', 10)
    lr = float(config.get('train', {}).get('lr', 1e-3))
    
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
    graphs_dir = config['data']['output_dir']
    dataset = OntologyGraphDataset(graphs_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_properties = getattr(dataset, "num_properties", 1)
    
    # Text Projector
    text_projector = nn.Linear(input_text_dim, hidden_dim).to(device)
    
    # Model
    model = OntoAlignEncoder(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        num_domains=num_domains,
        gnn_type=gnn_type,
        num_properties=num_properties
    ).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(text_projector.parameters()), lr=lr)
    
    print("Starting pre-training...")
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
            edge_property = getattr(batch, "edge_property_id", None)
            optimizer.zero_grad()
            
            # Project text features: 768 -> 256
            batch.x_text = text_projector(batch.x_text)
            
            # --- Loss 1: Edge Prediction ---
            # Forward on original graph
            h_orig_nodes = model(
                batch.x_text, 
                batch.edge_index, 
                batch.edge_type, 
                batch.batch, 
                batch.domain_id,
                edge_property_id=edge_property
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
                edge_property_id=edge_property
            )
            
            l_recon = masked_recon_loss(h_masked_nodes, h_orig_nodes, mask_idx)
            
            # --- Loss 3: Graph Contrastive Learning ---
            # View 1: Edge Dropout + Feature Dropout
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
                edge_property_id=edge_prop_1
            )
            
            # View 2: Different Dropout
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
                edge_property_id=edge_prop_2
            )
            
            l_contrast = contrastive_loss(h1_nodes, h2_nodes, batch.batch)
            
            # --- Loss 4: Cross-graph lexical alignment ---
            node_lex = getattr(batch, "node_lexical", None)
            l_cross = cross_graph_contrastive_loss(
                h_orig_nodes,
                batch.batch,
                node_lex,
                batch.edge_index,
                temperature=0.2,
                max_pairs=2048,
                graph_ptr=getattr(batch, "ptr", None)
            )
            
            # Total Loss
            loss = (
                w_edge * l_edge
                + w_recon * l_recon
                + w_contrast * l_contrast
                + w_cross * l_cross
            )
            
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
