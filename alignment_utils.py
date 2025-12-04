import os
import xml.etree.ElementTree as ET
from typing import Set, Tuple, Dict, List
import rdflib
from owlready2 import get_ontology, Thing, Ontology, World

class AlignmentEvaluator:
    def __init__(self, reference_path: str):
        """
        Load reference alignment from RDF/XML format.
        """
        self.reference_pairs = self._load_reference(reference_path)
        
    def _load_reference(self, path: str) -> Set[Tuple[str, str]]:
        """
        Parses the OAEI reference alignment file (RDF/XML).
        Returns a set of (iri_source, iri_target) tuples.
        Uses tag suffix checking (local-name style) for robustness against namespace variations.
        """
        pairs = set()
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Iterate over all elements to find Cells, ignoring specific namespace URIs
            # We look for tags ending in "Cell", "entity1", "entity2"
            
            # Recursive search for all elements
            for elem in root.iter():
                if elem.tag.endswith("Cell"):
                    e1 = None
                    e2 = None
                    
                    for child in elem:
                        if child.tag.endswith("entity1"):
                            e1 = child
                        elif child.tag.endswith("entity2"):
                            e2 = child
                            
                    if e1 is not None and e2 is not None:
                        # Get resource attribute, ignoring namespace in attribute name if possible
                        # Attributes in ET are {ns}attr or just attr. 
                        # We look for 'resource' at the end of the attribute key.
                        
                        res1 = None
                        res2 = None
                        
                        for k, v in e1.attrib.items():
                            if k.endswith("resource"):
                                res1 = v
                                break
                        
                        for k, v in e2.attrib.items():
                            if k.endswith("resource"):
                                res2 = v
                                break
                        
                        if res1 and res2:
                            pairs.add((res1, res2))
                        
        except Exception as e:
            print(f"XML Parsing failed: {e}. Pairs loaded so far: {len(pairs)}")
                
        return pairs

    def evaluate(self, predicted_pairs: Set[Tuple[str, str]]) -> Dict[str, float]:
        """
        Calculate Precision, Recall, F1.
        Handles directionality (Source-Target vs Target-Source).
        """
        # Check which direction matches better
        tp_1 = len(predicted_pairs.intersection(self.reference_pairs))
        
        # Create swapped set
        swapped_pairs = {(p[1], p[0]) for p in predicted_pairs}
        tp_2 = len(swapped_pairs.intersection(self.reference_pairs))
        
        tp = max(tp_1, tp_2)
        
        # Use the count of the predicted set (same size)
        total_pred = len(predicted_pairs)
        
        fp = total_pred - tp
        fn = len(self.reference_pairs) - tp
        
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / len(self.reference_pairs) if len(self.reference_pairs) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "Total Predicted": total_pred,
            "Correct (TP)": tp,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4)
        }

class OntologyLoader:
    def __init__(self, path: str):
        self.path = path
        self.onto = get_ontology(path).load()
        # Filter out built-in classes using owlready2.Thing
        # This is safer than string checking.
        self.classes = [c for c in self.onto.classes() 
                        if c != Thing]
        
        # Pre-compute synonym properties to avoid searching per entity
        self.synonym_props = []
        # Search all properties in the world (including imports)
        for prop in self.onto.world.properties():
            if hasattr(prop, 'iri') and 'synonym' in prop.iri.lower():
                self.synonym_props.append(prop)

    def get_classes(self):
        return self.classes
        
    def get_label(self, entity) -> str:
        """Get primary label."""
        if hasattr(entity, "label") and entity.label:
            return entity.label[0]
        return entity.name # Fallback to name fragment
    
    def get_synonyms(self, entity) -> List[str]:
        """
        Get synonyms using a generic property search.
        Iterates over pre-computed synonym properties.
        """
        synonyms = []
        
        for prop in self.synonym_props:
            # Get values safely
            if hasattr(prop, 'python_name'):
                vals = getattr(entity, prop.python_name, [])
                for v in vals:
                    if isinstance(v, str):
                        synonyms.append(v)
                    elif hasattr(v, "label") and v.label:
                        synonyms.extend(v.label)
                    else:
                        synonyms.append(str(v))
                        
        # Also standard label
        if hasattr(entity, "label") and entity.label:
            synonyms.extend(entity.label)
            
        return [str(s) for s in synonyms]

    def get_iri(self, entity) -> str:
        return entity.iri

