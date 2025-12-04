import glob
import os
import json
from typing import Dict

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from torch_geometric.data import Data
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required to use OntologyGraphDataset. Install it via 'pip install torch-geometric'."
    ) from exc


class OntologyGraphDataset(Dataset):
    """Loads precomputed ontology graphs with text embeddings."""

    def __init__(self, graphs_dir: str) -> None:
        self.graph_paths = sorted(glob.glob(os.path.join(graphs_dir, "*.pt")))
        if not self.graph_paths:
            raise FileNotFoundError(f"No graph files found in {graphs_dir}.")
        self.graphs_dir = graphs_dir
        self.property_vocab = self._load_property_vocab(graphs_dir)
        if self.property_vocab:
            self.num_properties = max(self.property_vocab.values()) + 1
        else:
            self.num_properties = 1

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        graph_path = self.graph_paths[idx]
        graph = torch.load(graph_path, map_location="cpu")

        if "x_text" not in graph:
            raise KeyError(
                f"Graph {graph_path} is missing 'x_text'. Run build_graphs.py to precompute embeddings."
            )
        if "edge_index" not in graph or "edge_type" not in graph:
            raise KeyError(f"Graph {graph_path} is missing edge information.")

        x_text = self._to_tensor(graph["x_text"], dtype=torch.float32)
        edge_index = self._to_tensor(graph["edge_index"], dtype=torch.long)
        edge_type = self._to_tensor(graph["edge_type"], dtype=torch.long)

        data = Data()
        data.x_text = x_text
        if "x_path" in graph:
            data.x_path = self._to_tensor(graph["x_path"], dtype=torch.float32)
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.edge_property_id = self._load_edge_property_ids(graph)
        data.num_nodes = x_text.shape[0]
        data.graph_name = graph.get("name", os.path.splitext(os.path.basename(graph_path))[0])
        
        if "domain_id" in graph:
            data.domain_id = graph["domain_id"]
        else:
            data.domain_id = idx

        if "node_iris" in graph:
            data.node_iris = graph["node_iris"]
        if "node_text" in graph:
            data.node_text = graph["node_text"]
        if "node_labels" in graph:
            data.node_labels = graph["node_labels"]
        if "node_lexical" in graph:
            data.node_lexical = np.array(graph["node_lexical"], dtype=object)

        return data

    @staticmethod
    def _to_tensor(value, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _load_property_vocab(self, graphs_dir: str) -> Dict[str, int]:
        vocab_path = os.path.join(graphs_dir, "edge_property_vocab.json")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    vocab = json.load(f)
                vocab = {str(k): int(v) for k, v in vocab.items()}
                if "__NONE__" not in vocab:
                    vocab["__NONE__"] = 0
                return vocab
            except Exception:
                pass
        vocab = {"__NONE__": 0}
        for path in self.graph_paths:
            try:
                graph = torch.load(path, map_location="cpu")
            except Exception:
                continue
            props = graph.get("edge_property")
            if not props:
                continue
            for prop in props:
                if not prop:
                    continue
                if prop not in vocab:
                    vocab[prop] = len(vocab)
        os.makedirs(graphs_dir, exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        return vocab

    def _map_property_to_id(self, prop) -> int:
        if not prop:
            return self.property_vocab.get("__NONE__", 0)
        if prop not in self.property_vocab:
            self.property_vocab[prop] = len(self.property_vocab)
            self.num_properties = max(self.num_properties, self.property_vocab[prop] + 1)
        return self.property_vocab[prop]

    def _load_edge_property_ids(self, graph) -> torch.Tensor:
        raw_ids = graph.get("edge_property_id")
        if raw_ids is not None:
            if torch.is_tensor(raw_ids):
                return raw_ids.to(dtype=torch.long)
            return torch.tensor(raw_ids, dtype=torch.long)
        raw_props = graph.get("edge_property")
        num_edges = graph["edge_index"].shape[1]
        if raw_props is None:
            return torch.zeros(num_edges, dtype=torch.long)
        prop_ids = [self._map_property_to_id(prop) for prop in raw_props]
        if len(prop_ids) < num_edges:
            prop_ids.extend([self.property_vocab.get("__NONE__", 0)] * (num_edges - len(prop_ids)))
        elif len(prop_ids) > num_edges:
            prop_ids = prop_ids[:num_edges]
        return torch.tensor(prop_ids, dtype=torch.long)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    
    def main():
        print("Initializing OntologyGraphDataset...")
        # Use current directory 'graphs' which we saw earlier
        # We use a small batch size and workers as requested
        dataset = OntologyGraphDataset(
            graphs_dir="graphs"
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # batch_size=4, parallel workers (num_workers=2 for testing parallelism)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        
        print("Iterating through DataLoader...")
        for idx, batch in enumerate(loader):
            print(f"Batch {idx}: {batch}")

    main()
