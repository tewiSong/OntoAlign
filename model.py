import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import RGCNConv

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # A is initialized with Gaussian, B with zeros
        # So initially W + AB = W
        self.lora_a = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x):
        # x @ (A @ B) * scaling
        return (x @ self.lora_a @ self.lora_b) * self.scaling

class OntoAlignEncoder(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_relations, num_domains, num_layers=2, gnn_type='rgcn', use_lora=False, lora_rank=8, num_properties=1):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.use_lora = use_lora
        self.num_properties = max(1, num_properties)
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.path_proj = nn.Linear(text_dim, hidden_dim)
        
        # Domain embeddings (Tokens)
        self.domain_emb = nn.Embedding(num_domains, hidden_dim)
        
        # Relation embeddings
        self.num_relations = num_relations
        self.total_relations = num_relations 
        
        self.rel_emb = nn.Embedding(self.total_relations, hidden_dim)
        self.property_emb = nn.Embedding(self.num_properties, hidden_dim)
        nn.init.xavier_uniform_(self.property_emb.weight)
        with torch.no_grad():
            self.property_emb.weight[0].zero_()
        
        # GNN
        self.gnn_layers = nn.ModuleList()
        
        # LoRA Layers (parallel to GNN)
        self.lora_layers = nn.ModuleList() if use_lora else None

        # Gated fusion modules for path + semantic features and semantic + structural features
        self.path_fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.semantic_gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.structural_gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        for _ in range(num_layers):
            if gnn_type == 'rgat':
                self.gnn_layers.append(RGCNConv(hidden_dim, hidden_dim, self.total_relations, num_bases=8))
            else:
                # Fallback to RGCN (default)
                self.gnn_layers.append(RGCNConv(hidden_dim, hidden_dim, self.total_relations))
            
            if use_lora:
                self.lora_layers.append(LoRALayer(hidden_dim, hidden_dim, rank=lora_rank))
                
    def forward(self, x_text, edge_index, edge_type, batch, domain_ids, domain_emb_input=None, edge_property_id=None, x_path=None):
        """
        Args:
            x_text: [N, text_dim] - Node text features
            edge_index: [2, E] - Graph edges
            edge_type: [E] - Edge types
            batch: [N] - Graph assignment for each node (0..B-1)
            domain_ids: [B] - Domain ID for each graph in the batch
            domain_emb_input: [B, hidden_dim] - Optional custom domain embeddings
        
        Returns:
            h_nodes: [N, hidden_dim] - Embeddings of original nodes
        """
        
        # 1. Project text features
        h = self.text_proj(x_text) # [N, hidden_dim]
        path_ctx = None
        if x_path is not None:
            path_ctx = self.path_proj(x_path)
        
        # 2. Get Domain Tokens
        if domain_emb_input is not None:
            d_emb = domain_emb_input # [B, d]
        else:
            d_emb = self.domain_emb(domain_ids) # [B, d]
            
        # 3. Apply Domain Tokens (Element-wise Multiplication per MDGPT)
        # Broadcast d_emb to nodes
        # d_emb is [B, d], batch is [N] with values 0..B-1
        if d_emb.size(0) == 1 and batch.max() >= 1:
             # Handle case where we have 1 domain emb but multiple batches (or just broadcast generally if B=1)
             d_emb_nodes = d_emb.expand(h.size(0), -1)
        else:
             d_emb_nodes = d_emb[batch] # [N, d]
        
        # MDGPT: h_hat = t_s * h_tilde
        h = h * d_emb_nodes
        if path_ctx is not None:
            if path_ctx.size(0) != h.size(0):
                raise ValueError("x_path must align with x_text along node dimension.")
            path_ctx = path_ctx * d_emb_nodes
            path_ctx = self.path_fusion_proj(path_ctx)
            path_gate_input = torch.cat([h, path_ctx], dim=-1)
            path_gate = self.path_gate_network(path_gate_input)
            h = path_gate * h + (1.0 - path_gate) * path_ctx
        
        if edge_property_id is not None and edge_property_id.numel() == edge_index.size(1):
            h = self._apply_property_prompts(h, edge_index, edge_property_id)
        semantic_base = h
        
        # 4. GNN Pass
        curr_h = h
        edge_prop_emb = self._edge_property_message(
            edge_index,
            edge_property_id,
            curr_h.shape[0],
            target_dtype=curr_h.dtype
        )
        for i, conv in enumerate(self.gnn_layers):
            if edge_prop_emb is not None:
                curr_h = curr_h + edge_prop_emb
            if self.training and curr_h.requires_grad:
                h_conv = checkpoint(conv, curr_h, edge_index, edge_type, use_reentrant=False)
            else:
                h_conv = conv(curr_h, edge_index, edge_type)
            
            # Apply LoRA if enabled
            if self.use_lora:
                h_lora = self.lora_layers[i](curr_h)
                curr_h = F.relu(h_conv + h_lora)
            else:
                curr_h = F.relu(h_conv)

        # 5. Gated fusion between semantic and structural views
        semantic_view = self.semantic_gate_proj(semantic_base)
        structural_view = self.structural_gate_proj(curr_h)
        gate_input = torch.cat([semantic_view, structural_view], dim=-1)
        fusion_gate = self.gate_network(gate_input)
        curr_h = fusion_gate * semantic_view + (1.0 - fusion_gate) * structural_view
        
        return curr_h

    def get_graph_embedding(self, x_text):
        """
        Computes a graph-level embedding by averaging projected text features.
        Useful for content-based attention.
        """
        h = self.text_proj(x_text) # [N, hidden_dim]
        # Simple mean pooling for now
        # [1, hidden_dim]
        return torch.mean(h, dim=0, keepdim=True)


    def _edge_property_message(self, edge_index, edge_property_id, num_nodes, target_dtype=None):
        if edge_property_id is None or edge_property_id.numel() == 0:
            return None
        if edge_property_id.numel() != edge_index.size(1):
            return None
        prop_ids = edge_property_id.to(self.property_emb.weight.device)
        prop_ids = prop_ids.clamp(0, self.property_emb.num_embeddings - 1)
        edge_prop_embed = self.property_emb(prop_ids)
        if target_dtype is not None:
            edge_prop_embed = edge_prop_embed.to(dtype=target_dtype)
        src = edge_index[0]
        dtype = edge_prop_embed.dtype
        device = edge_prop_embed.device
        deg = torch.zeros(num_nodes, dtype=dtype, device=device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg_prop = torch.zeros(num_nodes, edge_prop_embed.size(1), dtype=dtype, device=device)
        agg_prop.index_add_(0, src, edge_prop_embed)
        return agg_prop / deg

    def _apply_property_prompts(self, h, edge_index, edge_property_id):
        if edge_property_id is None or edge_property_id.numel() == 0:
            return h
        prop_ids = edge_property_id.to(h.device)
        prop_ids = prop_ids.clamp(0, self.property_emb.num_embeddings - 1)
        prop_emb = self.property_emb(prop_ids)
        prop_emb = prop_emb.to(dtype=h.dtype)
        if prop_emb.numel() == 0:
            return h
        src = edge_index[0]
        prop_aggr = torch.zeros_like(h)
        prop_aggr.index_add_(0, src, prop_emb)
        deg = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=h.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        return h + prop_aggr / deg

    def encode_with_mask(self, x_text, edge_index, edge_type, batch, domain_ids, mask_nodes_idx, edge_property_id=None, x_path=None):
        """
        Performs encoding with some nodes masked (e.g. text features zeroed out).
        """
        # Mask text features
        x_masked = x_text.clone()
        x_masked[mask_nodes_idx] = 0 # Simple zero masking
        
        return self.forward(
            x_masked,
            edge_index,
            edge_type,
            batch,
            domain_ids,
            edge_property_id=edge_property_id,
            x_path=x_path
        )


# Loss Functions

def edge_prediction_loss(h, edge_index, edge_type, rel_emb, property_emb=None, edge_property_id=None, num_neg=1):
    """
    Loss 1: Multi-relational edge prediction.
    h: [N, d]
    rel_emb: Embedding matrix [num_rels, d]
    """
    device = h.device
    num_edges = edge_index.size(1)
    
    src, dst = edge_index
    
    # Positive scores
    # Score(h_i, h_j, r) = (h_i * r * h_j).sum(-1)
    r_emb = rel_emb(edge_type) # [E, d]
    if property_emb is not None and edge_property_id is not None and edge_property_id.numel() == edge_type.numel():
        prop_ids = edge_property_id.to(device)
        prop_ids = prop_ids.clamp(0, property_emb.num_embeddings - 1)
        r_emb = r_emb + property_emb(prop_ids)
    
    score_pos = (h[src] * r_emb * h[dst]).sum(dim=-1) # [E]
    
    # Negative samples
    # Randomly corrupt tail
    neg_dst = torch.randint(0, h.size(0), (num_edges * num_neg,), device=device)
    # Repeat src and rel for negatives
    src_rep = src.repeat_interleave(num_neg)
    rel_rep = edge_type.repeat_interleave(num_neg)
    r_emb_neg = rel_emb(rel_rep)
    if property_emb is not None and edge_property_id is not None and edge_property_id.numel() == edge_type.numel():
        prop_rep = edge_property_id.repeat_interleave(num_neg).to(device)
        prop_rep = prop_rep.clamp(0, property_emb.num_embeddings - 1)
        r_emb_neg = r_emb_neg + property_emb(prop_rep)
    
    score_neg = (h[src_rep] * r_emb_neg * h[neg_dst]).sum(dim=-1) # [E * num_neg]
    
    # BCE Loss
    scores = torch.cat([score_pos, score_neg])
    labels = torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)])
    
    return F.binary_cross_entropy_with_logits(scores, labels)

def masked_recon_loss(h_masked, h_orig, mask_nodes_idx):
    """
    Loss 2: Masked Neighborhood Reconstruction.
    MSE between masked and original (detached) embeddings for masked nodes.
    """
    return F.mse_loss(h_masked[mask_nodes_idx], h_orig[mask_nodes_idx].detach())

def contrastive_loss(h1, h2, batch, temperature=0.1, max_nodes=4096):
    """
    Loss 3: Graph Contrastive Loss (InfoNCE).
    h1, h2: [N, d] - two views of the same nodes.
    batch: [N] - graph assignment.
    """
    # --- SAMPLING TO AVOID OOM ---
    num_nodes = h1.size(0)
    if num_nodes > max_nodes:
        # Randomly sample max_nodes indices
        perm = torch.randperm(num_nodes, device=h1.device)[:max_nodes]
        h1 = h1[perm]
        h2 = h2[perm]
        
    # Normalize
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    
    # Sim matrix: [M, M] where M <= max_nodes
    # 4096^2 * 4 bytes = 64 MB (Very small compared to 40GB)
    logits = torch.mm(h1, h2.t()) / temperature
    
    # Labels are diagonal
    labels = torch.arange(h1.size(0), device=h1.device)
    
    return F.cross_entropy(logits, labels)
