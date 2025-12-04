import os
import time
import itertools
from typing import List, Set, Tuple, Dict
import textdistance
from tqdm import tqdm
from alignment_utils import OntologyLoader, AlignmentEvaluator

# Configuration
DATASET_DIR = "datasets/oaei_anatomy"
SOURCE_ONTO = os.path.join(DATASET_DIR, "human.owl")
TARGET_ONTO = os.path.join(DATASET_DIR, "mouse.owl")
REF_ALIGN = os.path.join(DATASET_DIR, "reference.rdf")

class BaselineMethods:
    def __init__(self, source_loader: OntologyLoader, target_loader: OntologyLoader):
        self.source = source_loader
        self.target = target_loader
        self.source_classes = source_loader.get_classes()
        self.target_classes = target_loader.get_classes()
        
        print(f"Source classes: {len(self.source_classes)}")
        print(f"Target classes: {len(self.target_classes)}")
        
        # DEBUG: Print sample IRIs
        print("DEBUG: Sample Source IRIs:", [self.source.get_iri(c) for c in self.source_classes[:3]])
        print("DEBUG: Sample Target IRIs:", [self.target.get_iri(c) for c in self.target_classes[:3]])

    def _get_labels(self, entity, loader) -> List[str]:
        """Get normalized labels."""
        l = loader.get_label(entity)
        return [l.lower()] if l else []

    def _get_all_synonyms(self, entity, loader) -> Set[str]:
        syns = loader.get_synonyms(entity)
        return set([s.lower() for s in syns])

    def string_based_exact(self) -> Set[Tuple[str, str]]:
        """
        Exact Label Matching.
        """
        print("Running String-based: Exact Match...")
        mapping = set()
        
        # Create dictionary for target {label: iri}
        target_map = {}
        for t in self.target_classes:
            lbl = self.target.get_label(t).lower()
            if lbl:
                if lbl not in target_map:
                    target_map[lbl] = []
                target_map[lbl].append(self.target.get_iri(t))
                
        for s in self.source_classes:
            lbl = self.source.get_label(s).lower()
            if lbl and lbl in target_map:
                for t_iri in target_map[lbl]:
                    mapping.add((self.source.get_iri(s), t_iri))
                    
        return mapping

    def string_based_similarity(self, threshold=0.8) -> Set[Tuple[str, str]]:
        """
        String similarity using Levenshtein (normalized).
        """
        print(f"Running String-based: Levenshtein Similarity (threshold={threshold})...")
        mapping = set()
        
        # Precompute labels
        s_data = [(s, self.source.get_label(s).lower()) for s in self.source_classes]
        t_data = [(t, self.target.get_label(t).lower()) for t in self.target_classes]
        
        # Remove empty
        s_data = [x for x in s_data if x[1]]
        t_data = [x for x in t_data if x[1]]
        
        for s_ent, s_lbl in tqdm(s_data, desc="Comparing"):
            for t_ent, t_lbl in t_data:
                # Optimization: Length diff
                if abs(len(s_lbl) - len(t_lbl)) / max(len(s_lbl), len(t_lbl)) > (1 - threshold):
                    continue
                    
                sim = textdistance.levenshtein.normalized_similarity(s_lbl, t_lbl)
                if sim >= threshold:
                    mapping.add((self.source.get_iri(s_ent), self.target.get_iri(t_ent)))
                    
        return mapping

    def lexical_plus_rules(self) -> Set[Tuple[str, str]]:
        """
        Lexical + Rules:
        1. Exact Match on Labels
        2. Exact Match on Synonyms
        """
        print("Running Lexical + Rules...")
        mapping = set()
        
        # 1. Build index for target synonyms
        target_index = {}
        
        for t in self.target_classes:
            syns = self._get_all_synonyms(t, self.target)
            t_iri = self.target.get_iri(t)
            for syn in syns:
                if syn not in target_index:
                    target_index[syn] = []
                target_index[syn].append(t_iri)
                
        # 2. Iterate source
        for s in self.source_classes:
            s_syns = self._get_all_synonyms(s, self.source)
            s_iri = self.source.get_iri(s)
            
            for syn in s_syns:
                if syn in target_index:
                    for t_iri in target_index[syn]:
                        mapping.add((s_iri, t_iri))
                        
        return mapping

    def _check_neighbors(self, s_candidates, t_candidates, result_set, threshold):
        for s in s_candidates:
            for t in t_candidates:
                s_lbl = self.source.get_label(s).lower()
                t_lbl = self.target.get_label(t).lower()
                if not s_lbl or not t_lbl:
                    continue
                    
                sim = textdistance.levenshtein.normalized_similarity(s_lbl, t_lbl)
                if sim >= threshold:
                    result_set.add((self.source.get_iri(s), self.target.get_iri(t)))


def main():
    print("Loading Ontologies...")
    try:
        source = OntologyLoader(SOURCE_ONTO)
        target = OntologyLoader(TARGET_ONTO)
    except Exception as e:
        print(f"Error loading ontologies: {e}")
        return

    print("Loading Reference Alignment...")
    evaluator = None
    try:
        evaluator = AlignmentEvaluator(REF_ALIGN)
        print(f"Reference pairs loaded: {len(evaluator.reference_pairs)}")
        print(f"DEBUG: Sample Ref Pairs: {list(evaluator.reference_pairs)[:3]}")
    except Exception as e:
        print(f"Error loading reference: {e}")

    baselines = BaselineMethods(source, target)
    
    # Method 1: String-based (Exact)
    print("\n--- Method 1: String-based (Exact Match) ---")
    map_exact = baselines.string_based_exact()
    if evaluator:
        res = evaluator.evaluate(map_exact)
        print(res)

    # Method 1b: String-based (Similarity)
    print("\n--- Method 1b: String-based (Levenshtein > 0.9) ---")
    map_sim = baselines.string_based_similarity(threshold=0.9)
    if evaluator:
        res = evaluator.evaluate(map_sim)
        print(res)

    # Method 2: Lexical + Rules
    print("\n--- Method 2: Lexical + Rules (Synonyms) ---")
    map_lex = baselines.lexical_plus_rules()
    if evaluator:
        res = evaluator.evaluate(map_lex)
        print(res)


if __name__ == "__main__":
    main()
