import os
import glob
import time
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import torch
import yaml
from rdflib import BNode, Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection
from rdflib.namespace import SKOS
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Set, Tuple

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
SYNONYM_PROPS = [
    OBOINOWL.hasExactSynonym,
    OBOINOWL.hasRelatedSynonym,
    OBOINOWL.hasBroadSynonym,
    OBOINOWL.hasNarrowSynonym,
]
LABEL_PROPS = [RDFS.label, SKOS.prefLabel]
EXCLUDED_CLASS_IRIS = {OWL.Thing, OWL.Nothing}
NONE_PROPERTY_KEY = "__NONE__"


@dataclass
class TextEncoderConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 32
    max_length: int = 128
    device: Optional[str] = None


class TextEncoder:
    """Transformer-based encoder for node text descriptions."""

    def __init__(self, config: TextEncoderConfig):
        self.config = config
        if config.device:
            device_str = config.device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("Cannot encode empty text list.")

        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            model_out = self.model(**tokens)
            cls_embeddings = model_out.last_hidden_state[:, 0, :].detach().cpu()
            outputs.append(cls_embeddings)

        return torch.cat(outputs, dim=0).to(torch.float32)


def iri_fragment(iri: str) -> str:
    if not iri:
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rstrip("/").rsplit("/", 1)[-1]
    return iri


def is_named_class(node) -> bool:
    return isinstance(node, URIRef) and node not in EXCLUDED_CLASS_IRIS


def literal_to_str(node) -> Optional[str]:
    if isinstance(node, Literal):
        return str(node)
    return None


def resolve_annotation_value(graph: Graph, node) -> Optional[str]:
    if isinstance(node, Literal):
        return str(node)
    if isinstance(node, (URIRef, BNode)):
        for label in graph.objects(node, RDFS.label):
            return str(label)
        for label in graph.objects(node, SKOS.prefLabel):
            return str(label)
    return None


def collect_class_terms(graph: Graph) -> Set[URIRef]:
    """Gather every class IRI that participates in the ontology graph."""

    classes: Set[URIRef] = set()

    def add(node):
        if is_named_class(node):
            classes.add(node)

    for cls in graph.subjects(RDF.type, OWL.Class):
        add(cls)
    for cls in graph.subjects(RDF.type, RDFS.Class):
        add(cls)

    for subject, _, obj in graph.triples((None, RDFS.subClassOf, None)):
        add(subject)
        if isinstance(obj, URIRef):
            add(obj)
        elif isinstance(obj, BNode):
            for target in graph.objects(obj, OWL.someValuesFrom):
                add(target)
            for target in graph.objects(obj, OWL.allValuesFrom):
                add(target)

    for subject, _, obj in graph.triples((None, OWL.equivalentClass, None)):
        add(subject)
        add(obj)

    for subject, _, obj in graph.triples((None, OWL.disjointWith, None)):
        add(subject)
        add(obj)

    for disjoint in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        for member_list in graph.objects(disjoint, OWL.members):
            try:
                members = Collection(graph, member_list)
            except Exception:
                continue
            for member in members:
                add(member)

    for _, _, domain in graph.triples((None, RDFS.domain, None)):
        add(domain)
    for _, _, rng in graph.triples((None, RDFS.range, None)):
        add(rng)

    return classes


def build_node_metadata(graph: Graph, class_terms: List[URIRef]) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    node_iris: List[str] = []
    node_texts: List[str] = []
    node_labels: List[str] = []
    node_lexical: List[List[str]] = []

    for term in tqdm(class_terms, desc="Collecting node metadata"):
        iri = str(term)
        node_iris.append(iri)

        text_parts: List[str] = []
        seen: Set[str] = set()
        labels: List[str] = []
        synonyms: List[str] = []

        def add_text(value):
            if value and value not in seen:
                seen.add(value)
                text_parts.append(value)

        for label_prop in LABEL_PROPS:
            for label in graph.objects(term, label_prop):
                val = literal_to_str(label)
                if val:
                    labels.append(val)
                    add_text(val)

        for comment in graph.objects(term, RDFS.comment):
            add_text(literal_to_str(comment))

        for synonym_prop in SYNONYM_PROPS:
            for synonym in graph.objects(term, synonym_prop):
                val = resolve_annotation_value(graph, synonym)
                if val:
                    synonyms.append(val)
                    add_text(val)

        if not text_parts:
            add_text(iri_fragment(iri))

        lexical_terms: List[str] = []
        for candidate in labels + synonyms:
            normalized = candidate.strip()
            if normalized and normalized not in lexical_terms:
                lexical_terms.append(normalized)
        if not lexical_terms:
            lexical_terms.append(iri_fragment(iri))

        primary_label = labels[0] if labels else lexical_terms[0]
        node_labels.append(primary_label)
        node_lexical.append(lexical_terms)
        node_texts.append(" ".join(text_parts))

    return node_iris, node_texts, node_labels, node_lexical


def build_parent_map(graph: Graph, class_set: Set[URIRef]) -> Dict[URIRef, Set[URIRef]]:
    parents: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if child not in class_set or not isinstance(parent, URIRef):
            continue
        if parent in class_set:
            parents[child].add(parent)
    return parents


def _gather_ancestors(term, parent_map: Dict[URIRef, Set[URIRef]], max_depth: int, max_ancestors: int) -> List[URIRef]:
    ancestors: List[URIRef] = []
    visited: Set[URIRef] = set()
    frontier = list(parent_map.get(term, []))
    depth = 0
    while frontier and depth < max_depth and len(ancestors) < max_ancestors:
        next_frontier: List[URIRef] = []
        for parent in frontier:
            if parent in visited:
                continue
            visited.add(parent)
            ancestors.append(parent)
            if len(ancestors) >= max_ancestors:
                break
            next_frontier.extend(parent_map.get(parent, []))
        frontier = next_frontier
        depth += 1
    return ancestors


def compute_path_context_embeddings(
    ordered_terms: List[URIRef],
    term2id: Dict[URIRef, int],
    parent_map: Dict[URIRef, Set[URIRef]],
    node_embeddings: torch.Tensor,
    max_depth: int = 4,
    max_ancestors: int = 16
) -> torch.Tensor:
    context_vectors: List[torch.Tensor] = []
    for term in ordered_terms:
        node_idx = term2id[term]
        ancestor_terms = _gather_ancestors(term, parent_map, max_depth=max_depth, max_ancestors=max_ancestors)
        ancestor_ids = [term2id[a] for a in ancestor_terms if a in term2id]
        if ancestor_ids:
            ancestor_vecs = node_embeddings[ancestor_ids]
            combined = torch.cat([node_embeddings[node_idx].unsqueeze(0), ancestor_vecs], dim=0)
            context_vectors.append(combined.mean(dim=0))
        else:
            context_vectors.append(node_embeddings[node_idx])
    return torch.stack(context_vectors, dim=0)


def build_edges(
    graph: Graph,
    class_set: Set[URIRef],
    term2id: Dict[URIRef, int],
    rels: dict
) -> Tuple[List[int], List[int], List[int], List[Optional[str]]]:
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []
    edge_properties: List[Optional[str]] = []

    def add_edge(src_term, dst_term, rel, prop_iri: Optional[str] = None):
        if src_term not in class_set or dst_term not in class_set:
            return
        src_id = term2id.get(src_term)
        dst_id = term2id.get(dst_term)
        if src_id is None or dst_id is None:
            return
        edge_src.append(src_id)
        edge_dst.append(dst_id)
        edge_types.append(rel)
        edge_properties.append(prop_iri)

    rel_subclass = rels['REL_SUBCLASS']
    rel_subclass_inv = rels.get('REL_SUBCLASS_INV', rel_subclass)
    rel_equiv = rels['REL_EQUIV']
    rel_disjoint = rels['REL_DISJOINT']
    rel_exists = rels['REL_EXISTS']
    rel_exists_inv = rels.get('REL_EXISTS_INV')
    rel_forall = rels['REL_FORALL']
    rel_forall_inv = rels.get('REL_FORALL_INV')
    rel_domain = rels['REL_DOMAIN']
    rel_range = rels['REL_RANGE']
    rel_domain_inv = rels.get('REL_DOMAIN_INV')
    rel_range_inv = rels.get('REL_RANGE_INV')
    rel_subprop = rels['REL_SUBPROP']
    rel_subprop_inv = rels.get('REL_SUBPROP_INV')

    for cls, _, parent in tqdm(
        graph.triples((None, RDFS.subClassOf, None)),
        desc="Edges: rdfs:subClassOf",
        unit="triple"
    ):
        if cls not in class_set:
            continue
        if isinstance(parent, URIRef):
            add_edge(cls, parent, rel_subclass)
            add_edge(parent, cls, rel_subclass_inv)
        elif isinstance(parent, BNode):
            on_props = list(graph.objects(parent, OWL.onProperty))
            prop_iri = str(on_props[0]) if on_props else None
            some_targets = list(graph.objects(parent, OWL.someValuesFrom))
            all_targets = list(graph.objects(parent, OWL.allValuesFrom))
            if not some_targets and not all_targets:
                continue
            for restriction_target in some_targets:
                add_edge(cls, restriction_target, rel_exists, prop_iri)
                if rel_exists_inv is not None:
                    add_edge(restriction_target, cls, rel_exists_inv, prop_iri)
            for restriction_target in all_targets:
                add_edge(cls, restriction_target, rel_forall, prop_iri)
                if rel_forall_inv is not None:
                    add_edge(restriction_target, cls, rel_forall_inv, prop_iri)

    for cls_a, _, cls_b in tqdm(
        graph.triples((None, OWL.equivalentClass, None)),
        desc="Edges: owl:equivalentClass",
        unit="triple"
    ):
        add_edge(cls_a, cls_b, rel_equiv)
        add_edge(cls_b, cls_a, rel_equiv)

    for cls_a, _, cls_b in tqdm(
        graph.triples((None, OWL.disjointWith, None)),
        desc="Edges: owl:disjointWith",
        unit="triple"
    ):
        add_edge(cls_a, cls_b, rel_disjoint)
        add_edge(cls_b, cls_a, rel_disjoint)

    for disjoint in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        for member_list in graph.objects(disjoint, OWL.members):
            try:
                members = [m for m in Collection(graph, member_list) if m in class_set]
            except Exception:
                continue
            for cls_a, cls_b in combinations(members, 2):
                add_edge(cls_a, cls_b, rel_disjoint)
                add_edge(cls_b, cls_a, rel_disjoint)

    property_domains: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    property_ranges: Dict[URIRef, Set[URIRef]] = defaultdict(set)

    for prop, _, domain in tqdm(
        graph.triples((None, RDFS.domain, None)),
        desc="Scanning rdfs:domain",
        unit="triple"
    ):
        if isinstance(domain, URIRef) and domain in class_set:
            property_domains[prop].add(domain)

    for prop, _, rng in tqdm(
        graph.triples((None, RDFS.range, None)),
        desc="Scanning rdfs:range",
        unit="triple"
    ):
        if isinstance(rng, URIRef) and rng in class_set:
            property_ranges[prop].add(rng)

    for prop, domains in tqdm(property_domains.items(), desc="Edges: domain/range", unit="prop"):
        ranges = property_ranges.get(prop)
        if not ranges:
            continue
        for domain in domains:
            for rng in ranges:
                add_edge(domain, rng, rel_domain, str(prop))
                add_edge(rng, domain, rel_range, str(prop))
                if rel_domain_inv is not None:
                    add_edge(rng, domain, rel_domain_inv, str(prop))
                if rel_range_inv is not None:
                    add_edge(domain, rng, rel_range_inv, str(prop))

    for sub_prop, _, super_prop in tqdm(
        graph.triples((None, RDFS.subPropertyOf, None)),
        desc="Edges: rdfs:subPropertyOf",
        unit="triple"
    ):
        domains_sub = property_domains.get(sub_prop)
        domains_super = property_domains.get(super_prop)
        if not domains_sub or not domains_super:
            continue
        for domain_sub in domains_sub:
            for domain_super in domains_super:
                add_edge(domain_sub, domain_super, rel_subprop, str(sub_prop))
                if rel_subprop_inv is not None:
                    add_edge(domain_super, domain_sub, rel_subprop_inv, str(sub_prop))

    return edge_src, edge_dst, edge_types, edge_properties


def build_text_encoder(config: dict) -> TextEncoder:
    encoder_cfg = config.get('text_encoder', {}) or {}
    text_config = TextEncoderConfig(
        model_name=encoder_cfg.get('model_name', TextEncoderConfig.model_name),
        batch_size=encoder_cfg.get('batch_size', TextEncoderConfig.batch_size),
        max_length=encoder_cfg.get('max_length', TextEncoderConfig.max_length),
        device=encoder_cfg.get('device')
    )
    return TextEncoder(text_config)

def process_ontology(owl_path, config, text_encoder: TextEncoder, domain_id: int = 0, property_vocab: Optional[Dict[str, int]] = None):
    if property_vocab is None:
        property_vocab = {NONE_PROPERTY_KEY: 0}
    elif NONE_PROPERTY_KEY not in property_vocab:
        property_vocab[NONE_PROPERTY_KEY] = 0
    # Create output filename early to skip if exists
    filename = os.path.basename(owl_path).replace(".owl", ".pt")
    out_path = os.path.join(config['data']['output_dir'], filename)
    
    if os.path.exists(out_path):
        # If graph exists, load it to ensure vocab is complete
        # This handles cases where vocab.json is missing or partial
        print(f"Checking existing graph {filename} for properties...", flush=True)
        try:
            existing_graph = torch.load(out_path)
            raw_props = existing_graph.get("edge_property", [])
            if raw_props:
                for prop in raw_props:
                    if prop and prop not in property_vocab:
                        property_vocab[prop] = len(property_vocab)
            return None 
        except Exception as e:
            print(f"Failed to load existing {filename}, will regenerate: {e}")
            # Fall through to regeneration logic

    print(f"Loading {owl_path}...", flush=True)
    parse_start = time.time()
    graph = Graph()
    graph.parse(owl_path)
    parse_time = time.time() - parse_start
    print(f"Loaded {owl_path} in {parse_time:.1f}s. Collecting classes...", flush=True)

    rels = config['graph']['relations']
    collect_start = time.time()
    class_terms = collect_class_terms(graph)
    collect_time = time.time() - collect_start
    print(f"Found {len(class_terms)} class candidates in {collect_time:.1f}s.", flush=True)
    if not class_terms:
        return None

    ordered_terms = sorted(class_terms, key=lambda term: str(term))
    metadata_start = time.time()
    node_iris, node_text, node_labels, node_lexical = build_node_metadata(graph, ordered_terms)
    metadata_time = time.time() - metadata_start
    print(f"Node metadata collected in {metadata_time:.1f}s.", flush=True)

    encode_start = time.time()
    node_embeddings = text_encoder.encode(node_text)
    if node_embeddings.shape[0] != len(node_text):
        raise ValueError("Text encoder output size mismatch with node_text list.")
    encode_time = time.time() - encode_start
    print(f"Node text embeddings computed in {encode_time:.1f}s.", flush=True)

    class_set = set(ordered_terms)
    term2id = {term: idx for idx, term in enumerate(ordered_terms)}
    parent_map = build_parent_map(graph, class_set)
    path_embeddings = compute_path_context_embeddings(ordered_terms, term2id, parent_map, node_embeddings)
    edges_start = time.time()
    edge_src, edge_dst, edge_types, edge_properties = build_edges(graph, class_set, term2id, rels)
    edges_time = time.time() - edges_start
    print(f"Edges constructed in {edges_time:.1f}s ({len(edge_src)} edges).", flush=True)

    edge_property_ids: List[int] = []
    none_id = property_vocab.get(NONE_PROPERTY_KEY, 0)
    for prop in edge_properties:
        if prop is None:
            edge_property_ids.append(none_id)
        else:
            if prop not in property_vocab:
                property_vocab[prop] = len(property_vocab)
            edge_property_ids.append(property_vocab[prop])

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_property_tensor = torch.tensor(edge_property_ids, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_property_tensor = torch.empty((0,), dtype=torch.long)

    graph = {
        "name": os.path.splitext(os.path.basename(owl_path))[0],
        "domain_id": domain_id,
        "node_iris": node_iris,
        "node_text": node_text,
        "node_labels": node_labels,
        "node_lexical": node_lexical,
        "num_nodes": len(node_iris),
        "x_text": node_embeddings,
        "x_path": path_embeddings,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_property": edge_properties,
        "edge_property_id": edge_property_tensor
    }
    
    return graph

def main():
    config = load_config("config/pretrain.yaml")
    input_dir = config['data']['input_dir']
    output_dir = config['data']['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    owl_files = glob.glob(os.path.join(input_dir, "*.owl"))
    print(f"Found {len(owl_files)} OWL files.")
    text_encoder = build_text_encoder(config)
    
    vocab_path = os.path.join(output_dir, "edge_property_vocab.json")
    if os.path.exists(vocab_path):
        print(f"Loading existing property vocab from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            property_vocab = json.load(f)
    else:
        property_vocab = {NONE_PROPERTY_KEY: 0}
    
    for idx, owl_file in enumerate(tqdm(owl_files)):
        try:
            graph = process_ontology(owl_file, config, text_encoder, domain_id=idx, property_vocab=property_vocab)
            if graph is None:
                continue
            filename = os.path.basename(owl_file).replace(".owl", ".pt")
            out_path = os.path.join(output_dir, filename)
            torch.save(graph, out_path)
        except Exception as e:
            print(f"Error processing {owl_file}: {e}")
    
    vocab_path = os.path.join(output_dir, "edge_property_vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(property_vocab, f)
    print(f"Saved property vocab with {len(property_vocab)} entries to {vocab_path}")

if __name__ == "__main__":
    main()
