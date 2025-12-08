#!/usr/bin/env python3
"""
Approach 4: Custom Graph-Text Similarity for ERD Grade Prediction
Implements the custom_graph_text approach as described in the project specification.
"""

import json
import os
import csv
import re
import argparse
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Hyperparameters (can be tuned)
A, B, C, D = 0.1, 0.4, 0.1, 0.4  # Entity similarity weights (A+B+C+D=1)
T, U, V, X, Y, Z = 0.1, 0.2, 0.3, 0.2, 0.1, 0.1  # Relationship similarity weights (T+U+V+X+Y+Z=1)
alpha, beta, lambda_val = 0.4, 0.4, 0.2  # ERD similarity weights (alpha+beta+lambda=1)
c_type_similarity = 0.5  # Type similarity between entity and weak_entity
K = 5  # Number of neighbors for KNN
q_threshold = 0.3  # Similarity threshold for discarding low-similarity ERDs

# Load sentence transformer model for embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, replacing special characters, splitting camelCase, etc.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace underscores and hyphens with spaces
    text = text.replace('_', ' ').replace('-', ' ')
    
    # Split camelCase/PascalCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Remove extra spaces and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Standardize common abbreviations
    replacements = {
        'no': 'number', 'num': 'number', 'id': 'identifier',
        'pk': 'primary key', 'fk': 'foreign key'
    }
    for abbrev, full in replacements.items():
        text = re.sub(r'\b' + abbrev + r'\b', full, text)
    
    return text


def get_entity_embedding(entity: Dict, all_attributes: List[str]) -> np.ndarray:
    """
    Get embedding for an entity by averaging embeddings of its name and attributes.
    """
    texts = [normalize_text(entity['name'])]
    texts.extend([normalize_text(attr) for attr in all_attributes])
    
    if not texts:
        return np.zeros(384)  # Dimension of all-MiniLM-L6-v2
    
    embeddings = model.encode(texts)
    return np.mean(embeddings, axis=0)


# Cache for embeddings to avoid recomputing
_embedding_cache = {}

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text, using cache if available."""
    if not text:
        return np.zeros(384)
    if text in _embedding_cache:
        return _embedding_cache[text]
    emb = model.encode([text])[0]
    _embedding_cache[text] = emb
    return emb

def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Get embeddings for multiple texts, using cache where possible."""
    if not texts:
        return np.zeros((0, 384))
    
    # Check cache and batch encode only uncached texts
    uncached_texts = []
    uncached_indices = []
    cached_embeddings = []
    
    for i, text in enumerate(texts):
        if text in _embedding_cache:
            cached_embeddings.append((i, _embedding_cache[text]))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Batch encode uncached texts
    if uncached_texts:
        new_embeddings = model.encode(uncached_texts)
        for text, emb in zip(uncached_texts, new_embeddings):
            _embedding_cache[text] = emb
    
    # Combine cached and new embeddings
    all_embeddings = [None] * len(texts)
    for i, emb in cached_embeddings:
        all_embeddings[i] = emb
    for idx, text in zip(uncached_indices, uncached_texts):
        all_embeddings[idx] = _embedding_cache[text]
    
    return np.array(all_embeddings)


def entity_similarity(e1: Dict, e2: Dict) -> float:
    """
    Compute entity similarity between two entities/weak entities.
    ES(E1, E2) = A * type_similarity + B * name_similarity + C * num_attrs_similarity + D * attrs_similarity
    """
    # Type similarity
    type1 = e1.get('kind', 'entity')
    type2 = e2.get('kind', 'entity')
    
    if type1 == type2:
        type_sim = 1.0
    elif (type1 == 'entity' and type2 == 'weak_entity') or (type1 == 'weak_entity' and type2 == 'entity'):
        type_sim = c_type_similarity
    else:
        type_sim = 0.0
    
    # Name similarity (cosine of word embeddings)
    name1 = normalize_text(e1.get('name', ''))
    name2 = normalize_text(e2.get('name', ''))
    
    if name1 and name2:
        name_emb1 = get_embedding(name1)
        name_emb2 = get_embedding(name2)
        name_sim = float(cosine_similarity([name_emb1], [name_emb2])[0][0])
    else:
        name_sim = 0.0
    
    # Number of attributes similarity
    attrs1 = e1.get('attributes', []) + e1.get('primary_keys', [])
    attrs2 = e2.get('attributes', []) + e2.get('primary_keys', [])
    num_attrs1 = len(attrs1)
    num_attrs2 = len(attrs2)
    
    if num_attrs1 == 0 and num_attrs2 == 0:
        num_attrs_sim = 1.0
    elif num_attrs1 == 0 or num_attrs2 == 0:
        num_attrs_sim = 0.0
    else:
        num_attrs_sim = min(num_attrs1, num_attrs2) / max(num_attrs1, num_attrs2)
    
    # Attributes similarity (average embeddings)
    if attrs1 and attrs2:
        attrs_texts1 = [normalize_text(attr) for attr in attrs1]
        attrs_texts2 = [normalize_text(attr) for attr in attrs2]
        attrs_emb1 = get_embeddings_batch(attrs_texts1)
        attrs_emb2 = get_embeddings_batch(attrs_texts2)
        avg_emb1 = np.mean(attrs_emb1, axis=0)
        avg_emb2 = np.mean(attrs_emb2, axis=0)
        attrs_sim = float(cosine_similarity([avg_emb1], [avg_emb2])[0][0])
    else:
        attrs_sim = 0.0
    
    # Combined similarity
    similarity = A * type_sim + B * name_sim + C * num_attrs_sim + D * attrs_sim
    return similarity


def match_entities(entities1: List[Dict], entities2: List[Dict]) -> List[Tuple[Dict, Optional[Dict], float]]:
    """
    Greedy matching of entities between two ERDs.
    Returns list of (entity1, entity2 or None, similarity) tuples.
    """
    if not entities1 and not entities2:
        return []
    
    if not entities1:
        return [(None, e2, 0.0) for e2 in entities2]
    if not entities2:
        return [(e1, None, 0.0) for e1 in entities1]
    
    # Compute all pairwise similarities
    similarities = []
    for i, e1 in enumerate(entities1):
        for j, e2 in enumerate(entities2):
            sim = entity_similarity(e1, e2)
            similarities.append((i, j, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching
    matched = []
    used1 = set()
    used2 = set()
    
    for i, j, sim in similarities:
        if i not in used1 and j not in used2:
            matched.append((entities1[i], entities2[j], sim))
            used1.add(i)
            used2.add(j)
    
    # Add unmatched entities
    for i, e1 in enumerate(entities1):
        if i not in used1:
            matched.append((e1, None, 0.0))
    
    for j, e2 in enumerate(entities2):
        if j not in used2:
            matched.append((None, e2, 0.0))
    
    return matched


def all_entities_similarity(entities1: List[Dict], entities2: List[Dict], exclude_null: bool = True) -> float:
    """
    Compute similarity between all entities in two ERDs.
    Option 2 (excluding NULL) is used by default.
    """
    matches = match_entities(entities1, entities2)
    
    if exclude_null:
        # Option 2: Exclude NULL matches
        valid_matches = [sim for _, _, sim in matches if sim > 0]
        if not valid_matches:
            return 0.0
        return np.mean(valid_matches)
    else:
        # Option 1: Include all matches
        if not matches:
            return 0.0
        return np.mean([sim for _, _, sim in matches])


def relationship_similarity(rel1: Dict, rel2: Dict, entity_matches: List[Tuple[Dict, Optional[Dict], float]]) -> float:
    """
    Compute similarity between two relationships.
    """
    # Type similarity
    type1 = rel1.get('kind', 'relationship')
    type2 = rel2.get('kind', 'relationship')
    type_sim = 1.0 if type1 == type2 else 0.5
    
    # Arity similarity
    entities1 = rel1.get('involved_entities', [])
    entities2 = rel2.get('involved_entities', [])
    arity1 = len(entities1)
    arity2 = len(entities2)
    
    if arity1 == arity2:
        arity_sim = 1.0
    elif arity1 == 0 or arity2 == 0:
        arity_sim = 0.0
    else:
        arity_sim = min(arity1, arity2) / max(arity1, arity2)
    
    # Participating entities similarity (based on entity matches)
    # Create a mapping from entity names to their match similarities
    entity_match_map = {}
    for e1, e2, sim in entity_matches:
        if e1 is not None and e2 is not None:
            entity_match_map[(e1.get('name', ''), e2.get('name', ''))] = sim
    
    # Match entities in relationships
    entity_names1 = [e.get('name', '') for e in entities1]
    entity_names2 = [e.get('name', '') for e in entities2]
    
    # Try to match entities in relationships
    participating_sim = 0.0
    if entity_names1 and entity_names2:
        # Simple matching: check if entity pairs exist in matches
        matched_pairs = 0
        total_pairs = min(len(entity_names1), len(entity_names2))
        
        if total_pairs > 0:
            # For simplicity, compute average similarity of best matches
            similarities = []
            for name1 in entity_names1[:total_pairs]:
                for name2 in entity_names2[:total_pairs]:
                    if (name1, name2) in entity_match_map:
                        similarities.append(entity_match_map[(name1, name2)])
            if similarities:
                participating_sim = np.mean(similarities)
            else:
                participating_sim = 0.0
    elif not entity_names1 and not entity_names2:
        participating_sim = 1.0
    
    # Attributes similarity
    attrs1 = rel1.get('attributes', [])
    attrs2 = rel2.get('attributes', [])
    
    if attrs1 and attrs2:
        attrs_texts1 = [normalize_text(attr) for attr in attrs1]
        attrs_texts2 = [normalize_text(attr) for attr in attrs2]
        attrs_emb1 = get_embeddings_batch(attrs_texts1)
        attrs_emb2 = get_embeddings_batch(attrs_texts2)
        avg_emb1 = np.mean(attrs_emb1, axis=0)
        avg_emb2 = np.mean(attrs_emb2, axis=0)
        attrs_sim = float(cosine_similarity([avg_emb1], [avg_emb2])[0][0])
    elif not attrs1 and not attrs2:
        attrs_sim = 1.0
    else:
        attrs_sim = 0.0
    
    # Cardinality similarity (for binary relationships only)
    max_card_sim = 1.0
    min_card_sim = 1.0
    
    if arity1 == 2 and arity2 == 2:
        # Extract cardinalities
        cards1 = [e.get('cardinality', 'Unknown') for e in entities1]
        cards2 = [e.get('cardinality', 'Unknown') for e in entities2]
        
        if cards1 and cards2:
            # Max cardinality similarity
            max_cards1 = set([c[0] if len(c) > 0 else 'U' for c in cards1])
            max_cards2 = set([c[0] if len(c) > 0 else 'U' for c in cards2])
            max_card_sim = len(max_cards1 & max_cards2) / len(max_cards1 | max_cards2) if (max_cards1 | max_cards2) else 1.0
            
            # Min cardinality similarity
            min_cards1 = set([c[1] if len(c) > 1 else 'U' for c in cards1])
            min_cards2 = set([c[1] if len(c) > 1 else 'U' for c in cards2])
            min_card_sim = len(min_cards1 & min_cards2) / len(min_cards1 | min_cards2) if (min_cards1 | min_cards2) else 1.0
    
    # Combined relationship similarity
    similarity = (T * type_sim + U * arity_sim + V * participating_sim + 
                  X * attrs_sim + Y * max_card_sim + Z * min_card_sim)
    return similarity


def match_relationships(rels1: List[Dict], rels2: List[Dict], entity_matches: List[Tuple[Dict, Optional[Dict], float]]) -> List[Tuple[Dict, Optional[Dict], float]]:
    """
    Greedy matching of relationships between two ERDs.
    """
    if not rels1 and not rels2:
        return []
    
    if not rels1:
        return [(None, r2, 0.0) for r2 in rels2]
    if not rels2:
        return [(r1, None, 0.0) for r1 in rels1]
    
    # Compute all pairwise similarities
    similarities = []
    for i, r1 in enumerate(rels1):
        for j, r2 in enumerate(rels2):
            sim = relationship_similarity(r1, r2, entity_matches)
            similarities.append((i, j, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching
    matched = []
    used1 = set()
    used2 = set()
    
    for i, j, sim in similarities:
        if i not in used1 and j not in used2:
            matched.append((rels1[i], rels2[j], sim))
            used1.add(i)
            used2.add(j)
    
    # Add unmatched relationships
    for i, r1 in enumerate(rels1):
        if i not in used1:
            matched.append((r1, None, 0.0))
    
    for j, r2 in enumerate(rels2):
        if j not in used2:
            matched.append((None, r2, 0.0))
    
    return matched


def all_relationships_similarity(rels1: List[Dict], rels2: List[Dict], entity_matches: List[Tuple[Dict, Optional[Dict], float]], exclude_null: bool = True) -> float:
    """
    Compute similarity between all relationships in two ERDs.
    """
    matches = match_relationships(rels1, rels2, entity_matches)
    
    if exclude_null:
        valid_matches = [sim for _, _, sim in matches if sim > 0]
        if not valid_matches:
            return 0.0
        return np.mean(valid_matches)
    else:
        if not matches:
            return 0.0
        return np.mean([sim for _, _, sim in matches])


def get_erd_features(erd: Dict) -> np.ndarray:
    """
    Extract additional ERD features as a vector.
    Features: num_entities, num_weak_entities, num_relationship_attrs, 
              num_binary_rels, num_binary_identifying_rels, num_nway_rels, num_nway_identifying_rels
    """
    entities = erd.get('entities', [])
    relationships = erd.get('relationships', [])
    
    num_entities = sum(1 for e in entities if e.get('kind') == 'entity')
    num_weak_entities = sum(1 for e in entities if e.get('kind') == 'weak_entity')
    
    # Count relationship attributes (attributes in relationships)
    num_relationship_attrs = sum(len(r.get('attributes', [])) for r in relationships)
    
    # Count binary relationships
    num_binary_rels = 0
    num_binary_identifying_rels = 0
    num_nway_rels = 0
    num_nway_identifying_rels = 0
    
    for rel in relationships:
        arity = len(rel.get('involved_entities', []))
        is_identifying = rel.get('kind') == 'identifying relationship'
        
        if arity == 2:
            if is_identifying:
                num_binary_identifying_rels += 1
            else:
                num_binary_rels += 1
        elif arity > 2:
            if is_identifying:
                num_nway_identifying_rels += 1
            else:
                num_nway_rels += 1
    
    features = np.array([
        num_entities, num_weak_entities, num_relationship_attrs,
        num_binary_rels, num_binary_identifying_rels, num_nway_rels, num_nway_identifying_rels
    ])
    
    # Normalize features to [0, 1] range (using max values from training data)
    # For simplicity, we'll use a reasonable max value
    max_vals = np.array([20, 10, 50, 20, 10, 10, 5])
    features = features / (max_vals + 1)  # +1 to avoid division by zero
    
    return features


def additional_features_similarity(erd1: Dict, erd2: Dict) -> float:
    """
    Compute similarity of additional ERD features using cosine similarity.
    """
    features1 = get_erd_features(erd1)
    features2 = get_erd_features(erd2)
    
    # Use cosine similarity
    return float(cosine_similarity([features1], [features2])[0][0])


def erd_similarity(erd1: Dict, erd2: Dict) -> float:
    """
    Compute overall ERD similarity.
    ERD similarity = α * All Entities similarity + β * All relationships similarity + λ * Additional features similarity
    """
    entities1 = erd1.get('entities', [])
    entities2 = erd2.get('entities', [])
    relationships1 = erd1.get('relationships', [])
    relationships2 = erd2.get('relationships', [])
    
    # Compute entity matches first (needed for relationship similarity)
    entity_matches = match_entities(entities1, entities2)
    
    # All entities similarity
    entities_sim = all_entities_similarity(entities1, entities2, exclude_null=True)
    
    # All relationships similarity
    rels_sim = all_relationships_similarity(relationships1, relationships2, entity_matches, exclude_null=True)
    
    # Additional features similarity
    features_sim = additional_features_similarity(erd1, erd2)
    
    # Combined similarity
    similarity = alpha * entities_sim + beta * rels_sim + lambda_val * features_sim
    return similarity


def load_erd_data(dataset_dir: str) -> Dict[int, Dict]:
    """
    Load all ERD JSON files from a dataset directory.
    Returns a dictionary mapping ERD number to ERD data.
    """
    erds = {}
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            # Extract ERD number from filename (e.g., "104_1.json" -> 104)
            try:
                erd_num = int(filename.split('_')[0])
                filepath = os.path.join(dataset_dir, filename)
                with open(filepath, 'r') as f:
                    erd_data = json.load(f)
                    erds[erd_num] = erd_data
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
    
    return erds


def load_grades(csv_path: str) -> Dict[int, Tuple[float, float]]:
    """
    Load grades from ERD_grades.csv.
    Returns a dictionary mapping ERD number to (dataset1_grade, dataset2_grade) tuple.
    """
    grades = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            erd_no = int(row['ERD_No'])
            grade1 = float(row['dataset1_grade']) if row['dataset1_grade'] else None
            grade2 = float(row['dataset2_grade']) if row['dataset2_grade'] else None
            grades[erd_no] = (grade1, grade2)
    
    return grades


def knn_predict(test_erd: Dict, training_erds: List[Dict], training_grades: List[float], k: int = K, threshold: float = q_threshold) -> float:
    """
    Predict grade for a test ERD using KNN regression.
    """
    # Compute similarities to all training ERDs
    similarities = []
    for train_erd, train_grade in zip(training_erds, training_grades):
        if train_grade is not None:
            sim = erd_similarity(test_erd, train_erd)
            similarities.append((sim, train_grade))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Filter by threshold
    filtered_similarities = [(sim, grade) for sim, grade in similarities if sim >= threshold]
    
    if not filtered_similarities:
        # Fall back to global mean if all similarities are below threshold
        valid_grades = [g for g in training_grades if g is not None]
        if valid_grades:
            return np.mean(valid_grades)
        return 0.0
    
    # Use top K (or fewer if not enough)
    top_k = filtered_similarities[:k]
    
    if not top_k:
        valid_grades = [g for g in training_grades if g is not None]
        if valid_grades:
            return np.mean(valid_grades)
        return 0.0
    
    # Weighted average prediction
    weights = [sim for sim, _ in top_k]
    grades = [grade for _, grade in top_k]
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(grades)
    
    prediction = sum(w * g for w, g in zip(weights, grades)) / total_weight
    return prediction


def save_training_data(train_data1: List[Dict], train_grades1: List[float],
                       train_data2: List[Dict], train_grades2: List[float],
                       embedding_cache: Dict, save_path: str = 'training_data.pkl'):
    """
    Save training data and embedding cache to disk.
    """
    print(f"Saving training data and embeddings to {save_path}...")
    data_to_save = {
        'train_data1': train_data1,
        'train_grades1': train_grades1,
        'train_data2': train_data2,
        'train_grades2': train_grades2,
        'embedding_cache': embedding_cache
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Training data saved successfully!")


def load_training_data(load_path: str = 'training_data.pkl') -> Tuple[List[Dict], List[float], List[Dict], List[float], Dict]:
    """
    Load training data and embedding cache from disk.
    Returns (train_data1, train_grades1, train_data2, train_grades2, embedding_cache)
    """
    print(f"Loading training data and embeddings from {load_path}...")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Training data file {load_path} not found. Please run in train mode first.")
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Training data loaded successfully!")
    return (data['train_data1'], data['train_grades1'],
            data['train_data2'], data['train_grades2'],
            data['embedding_cache'])


def main():
    global _embedding_cache
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='ERD Grade Prediction using Custom Graph-Text Similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python a4_custom_graph.py --mode train    # Train and save training data
  python a4_custom_graph.py --mode inference  # Load saved data and run inference
  python a4_custom_graph.py                 # Default: auto-detect (train if no saved data exists)
        """
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'inference', 'auto'],
        default='auto',
        help='Mode: train (load and save training data), inference (load saved data), or auto (train if no saved data exists)'
    )
    parser.add_argument(
        '--save-path',
        default='training_data.pkl',
        help='Path to save/load training data (default: training_data.pkl)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    mode = args.mode
    if mode == 'auto':
        if os.path.exists(args.save_path):
            mode = 'inference'
            print(f"Found saved training data at {args.save_path}. Using inference mode.")
        else:
            mode = 'train'
            print(f"No saved training data found. Using train mode.")
    
    # Load or prepare training data
    if mode == 'inference':
        # Load saved training data
        train_data1, train_grades1, train_data2, train_grades2, _embedding_cache = load_training_data(args.save_path)
        print(f"Loaded {len(train_data1)} training samples for Dataset1")
        print(f"Loaded {len(train_data2)} training samples for Dataset2")
    else:
        # Train mode: load training data from files
        _embedding_cache = {}  # Clear cache at start
        
        print("Loading training data...")
        train_dataset1_dir = 'for_students/Dataset1'
        train_dataset2_dir = 'for_students/Dataset2'
        grades_csv = 'for_students/ERD_grades.csv'
        
        train_erds1 = load_erd_data(train_dataset1_dir)
        train_erds2 = load_erd_data(train_dataset2_dir)
        grades = load_grades(grades_csv)
        
        print(f"Loaded {len(train_erds1)} training ERDs from Dataset1")
        print(f"Loaded {len(train_erds2)} training ERDs from Dataset2")
        print(f"Loaded grades for {len(grades)} ERDs")
        
        # Prepare training data for each dataset
        train_data1 = []
        train_grades1 = []
        for erd_num, erd_data in train_erds1.items():
            if erd_num in grades and grades[erd_num][0] is not None:
                train_data1.append(erd_data)
                train_grades1.append(grades[erd_num][0])
        
        train_data2 = []
        train_grades2 = []
        for erd_num, erd_data in train_erds2.items():
            if erd_num in grades and grades[erd_num][1] is not None:
                train_data2.append(erd_data)
                train_grades2.append(grades[erd_num][1])
        
        print(f"Prepared {len(train_data1)} training samples for Dataset1")
        print(f"Prepared {len(train_data2)} training samples for Dataset2")
        
        # Warm up embedding cache by computing some embeddings from training data
        print("Warming up embedding cache...")
        texts_to_cache = []
        for erd in train_data1[:10] + train_data2[:10]:  # Sample ERDs
            entities = erd.get('entities', [])
            relationships = erd.get('relationships', [])
            for entity in entities:
                texts_to_cache.append(normalize_text(entity.get('name', '')))
                # Add attribute names
                for attr in (entity.get('attributes', []) + entity.get('primary_keys', [])):
                    texts_to_cache.append(normalize_text(attr))
            for rel in relationships:
                for attr in rel.get('attributes', []):
                    texts_to_cache.append(normalize_text(attr))
        
        # Compute embeddings in batch to populate cache
        if texts_to_cache:
            unique_texts = list(set([t for t in texts_to_cache if t]))  # Remove empty strings and duplicates
            if unique_texts:
                get_embeddings_batch(unique_texts[:100])  # Cache up to 100 unique texts
        
        print(f"Cache warmed up with {len(_embedding_cache)} embeddings")
        
        # Save training data
        save_training_data(train_data1, train_grades1, train_data2, train_grades2, _embedding_cache, args.save_path)
    
    # Load test data
    print("Loading test data...")
    test_dataset1_dir = 'for_testing/Dataset1'
    test_dataset2_dir = 'for_testing/Dataset2'
    
    test_erds1 = load_erd_data(test_dataset1_dir)
    test_erds2 = load_erd_data(test_dataset2_dir)
    
    print(f"Loaded {len(test_erds1)} test ERDs from Dataset1")
    print(f"Loaded {len(test_erds2)} test ERDs from Dataset2")
    
    # Get all test ERD numbers (union of both datasets)
    all_test_nums = sorted(set(list(test_erds1.keys()) + list(test_erds2.keys())))
    
    # Predict grades for test data
    print(f"\nPredicting grades for {len(all_test_nums)} test ERDs...")
    print("This may take a while as we compute similarities...")
    predictions = {}
    
    total = len(all_test_nums)
    for idx, erd_num in enumerate(all_test_nums, 1):
        pred1 = None
        pred2 = None
        
        if erd_num in test_erds1:
            print(f"  [{idx}/{total}] Predicting Dataset1 for ERD {erd_num}...", end=' ', flush=True)
            pred1 = knn_predict(test_erds1[erd_num], train_data1, train_grades1)
            print(f"Grade: {pred1:.2f}")
        
        if erd_num in test_erds2:
            print(f"  [{idx}/{total}] Predicting Dataset2 for ERD {erd_num}...", end=' ', flush=True)
            pred2 = knn_predict(test_erds2[erd_num], train_data2, train_grades2)
            print(f"Grade: {pred2:.2f}")
        
        predictions[erd_num] = (pred1, pred2)
    
    # Write predictions to CSV
    print("Writing predictions to a4_custom_graph.csv...")
    with open('a4_custom_graph.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ERD_No', 'dataset1_grade', 'dataset2_grade'])
        
        for erd_num in sorted(predictions.keys()):
            pred1, pred2 = predictions[erd_num]
            writer.writerow([
                erd_num,
                f"{pred1:.2f}" if pred1 is not None else "",
                f"{pred2:.2f}" if pred2 is not None else ""
            ])
    
    print("Done! Predictions written to a4_custom_graph.csv")


if __name__ == '__main__':
    main()

