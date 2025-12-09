#!/usr/bin/env python3
"""
Hyperparameter tuning script for ERD grade prediction.
Uses leave-one-out cross-validation to evaluate different hyperparameter combinations.
"""

import json
import os
import csv
import numpy as np
import itertools
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import functions from a4_custom_graph
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll need to import and modify the functions to accept hyperparameters
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load sentence transformer model (shared across all runs)
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for embeddings
_embedding_cache = {}

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, replacing special characters, splitting camelCase, etc."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace('_', ' ').replace('-', ' ')
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    replacements = {
        'no': 'number', 'num': 'number', 'id': 'identifier',
        'pk': 'primary key', 'fk': 'foreign key'
    }
    for abbrev, full in replacements.items():
        text = re.sub(r'\b' + abbrev + r'\b', full, text)
    return text

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
    uncached_texts = []
    uncached_indices = []
    cached_embeddings = []
    for i, text in enumerate(texts):
        if text in _embedding_cache:
            cached_embeddings.append((i, _embedding_cache[text]))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    if uncached_texts:
        new_embeddings = model.encode(uncached_texts)
        for text, emb in zip(uncached_texts, new_embeddings):
            _embedding_cache[text] = emb
    all_embeddings = [None] * len(texts)
    for i, emb in cached_embeddings:
        all_embeddings[i] = emb
    for idx, text in zip(uncached_indices, uncached_texts):
        all_embeddings[idx] = _embedding_cache[text]
    return np.array(all_embeddings)

def entity_similarity(e1: Dict, e2: Dict, A: float, B: float, C: float, D: float, c_type_similarity: float) -> float:
    """Compute entity similarity between two entities/weak entities."""
    type1 = e1.get('kind', 'entity')
    type2 = e2.get('kind', 'entity')
    if type1 == type2:
        type_sim = 1.0
    elif (type1 == 'entity' and type2 == 'weak_entity') or (type1 == 'weak_entity' and type2 == 'entity'):
        type_sim = c_type_similarity
    else:
        type_sim = 0.0
    
    name1 = normalize_text(e1.get('name', ''))
    name2 = normalize_text(e2.get('name', ''))
    if name1 and name2:
        name_emb1 = get_embedding(name1)
        name_emb2 = get_embedding(name2)
        name_sim = float(cosine_similarity([name_emb1], [name_emb2])[0][0])
    else:
        name_sim = 0.0
    
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
    
    similarity = A * type_sim + B * name_sim + C * num_attrs_sim + D * attrs_sim
    return similarity

def match_entities(entities1: List[Dict], entities2: List[Dict], A: float, B: float, C: float, D: float, c_type_similarity: float) -> List[Tuple[Dict, Dict, float]]:
    """Greedy matching of entities between two ERDs."""
    if not entities1 and not entities2:
        return []
    if not entities1:
        return [(None, e2, 0.0) for e2 in entities2]
    if not entities2:
        return [(e1, None, 0.0) for e1 in entities1]
    
    similarities = []
    for i, e1 in enumerate(entities1):
        for j, e2 in enumerate(entities2):
            sim = entity_similarity(e1, e2, A, B, C, D, c_type_similarity)
            similarities.append((i, j, sim))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    matched = []
    used1 = set()
    used2 = set()
    for i, j, sim in similarities:
        if i not in used1 and j not in used2:
            matched.append((entities1[i], entities2[j], sim))
            used1.add(i)
            used2.add(j)
    for i, e1 in enumerate(entities1):
        if i not in used1:
            matched.append((e1, None, 0.0))
    for j, e2 in enumerate(entities2):
        if j not in used2:
            matched.append((None, e2, 0.0))
    return matched

def all_entities_similarity(entities1: List[Dict], entities2: List[Dict], A: float, B: float, C: float, D: float, c_type_similarity: float, exclude_null: bool = True) -> float:
    """Compute similarity between all entities in two ERDs."""
    matches = match_entities(entities1, entities2, A, B, C, D, c_type_similarity)
    if exclude_null:
        valid_matches = [sim for _, _, sim in matches if sim > 0]
        if not valid_matches:
            return 0.0
        return np.mean(valid_matches)
    else:
        if not matches:
            return 0.0
        return np.mean([sim for _, _, sim in matches])

def relationship_similarity(rel1: Dict, rel2: Dict, entity_matches: List[Tuple], T: float, U: float, V: float, X: float, Y: float, Z: float) -> float:
    """Compute similarity between two relationships."""
    type1 = rel1.get('kind', 'relationship')
    type2 = rel2.get('kind', 'relationship')
    type_sim = 1.0 if type1 == type2 else 0.5
    
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
    
    entity_match_map = {}
    for e1, e2, sim in entity_matches:
        if e1 is not None and e2 is not None:
            entity_match_map[(e1.get('name', ''), e2.get('name', ''))] = sim
    
    entity_names1 = [e.get('name', '') for e in entities1]
    entity_names2 = [e.get('name', '') for e in entities2]
    participating_sim = 0.0
    if entity_names1 and entity_names2:
        matched_pairs = 0
        total_pairs = min(len(entity_names1), len(entity_names2))
        if total_pairs > 0:
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
    
    max_card_sim = 1.0
    min_card_sim = 1.0
    if arity1 == 2 and arity2 == 2:
        cards1 = [e.get('cardinality', 'Unknown') for e in entities1]
        cards2 = [e.get('cardinality', 'Unknown') for e in entities2]
        if cards1 and cards2:
            max_cards1 = set([c[0] if len(c) > 0 else 'U' for c in cards1])
            max_cards2 = set([c[0] if len(c) > 0 else 'U' for c in cards2])
            max_card_sim = len(max_cards1 & max_cards2) / len(max_cards1 | max_cards2) if (max_cards1 | max_cards2) else 1.0
            min_cards1 = set([c[1] if len(c) > 1 else 'U' for c in cards1])
            min_cards2 = set([c[1] if len(c) > 1 else 'U' for c in cards2])
            min_card_sim = len(min_cards1 & min_cards2) / len(min_cards1 | min_cards2) if (min_cards1 | min_cards2) else 1.0
    
    similarity = (T * type_sim + U * arity_sim + V * participating_sim + 
                  X * attrs_sim + Y * max_card_sim + Z * min_card_sim)
    return similarity

def match_relationships(rels1: List[Dict], rels2: List[Dict], entity_matches: List[Tuple], T: float, U: float, V: float, X: float, Y: float, Z: float) -> List[Tuple]:
    """Greedy matching of relationships between two ERDs."""
    if not rels1 and not rels2:
        return []
    if not rels1:
        return [(None, r2, 0.0) for r2 in rels2]
    if not rels2:
        return [(r1, None, 0.0) for r1 in rels1]
    
    similarities = []
    for i, r1 in enumerate(rels1):
        for j, r2 in enumerate(rels2):
            sim = relationship_similarity(r1, r2, entity_matches, T, U, V, X, Y, Z)
            similarities.append((i, j, sim))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    matched = []
    used1 = set()
    used2 = set()
    for i, j, sim in similarities:
        if i not in used1 and j not in used2:
            matched.append((rels1[i], rels2[j], sim))
            used1.add(i)
            used2.add(j)
    for i, r1 in enumerate(rels1):
        if i not in used1:
            matched.append((r1, None, 0.0))
    for j, r2 in enumerate(rels2):
        if j not in used2:
            matched.append((None, r2, 0.0))
    return matched

def all_relationships_similarity(rels1: List[Dict], rels2: List[Dict], entity_matches: List[Tuple], T: float, U: float, V: float, X: float, Y: float, Z: float, exclude_null: bool = True) -> float:
    """Compute similarity between all relationships in two ERDs."""
    matches = match_relationships(rels1, rels2, entity_matches, T, U, V, X, Y, Z)
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
    """Extract additional ERD features as a vector."""
    entities = erd.get('entities', [])
    relationships = erd.get('relationships', [])
    num_entities = sum(1 for e in entities if e.get('kind') == 'entity')
    num_weak_entities = sum(1 for e in entities if e.get('kind') == 'weak_entity')
    num_relationship_attrs = sum(len(r.get('attributes', [])) for r in relationships)
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
    max_vals = np.array([20, 10, 50, 20, 10, 10, 5])
    features = features / (max_vals + 1)
    return features

def additional_features_similarity(erd1: Dict, erd2: Dict) -> float:
    """Compute similarity of additional ERD features using cosine similarity."""
    features1 = get_erd_features(erd1)
    features2 = get_erd_features(erd2)
    return float(cosine_similarity([features1], [features2])[0][0])

def erd_similarity(erd1: Dict, erd2: Dict, A: float, B: float, C: float, D: float, 
                   T: float, U: float, V: float, X: float, Y: float, Z: float,
                   alpha: float, beta: float, lambda_val: float, c_type_similarity: float) -> float:
    """Compute overall ERD similarity."""
    entities1 = erd1.get('entities', [])
    entities2 = erd2.get('entities', [])
    relationships1 = erd1.get('relationships', [])
    relationships2 = erd2.get('relationships', [])
    entity_matches = match_entities(entities1, entities2, A, B, C, D, c_type_similarity)
    entities_sim = all_entities_similarity(entities1, entities2, A, B, C, D, c_type_similarity, exclude_null=True)
    rels_sim = all_relationships_similarity(relationships1, relationships2, entity_matches, T, U, V, X, Y, Z, exclude_null=True)
    features_sim = additional_features_similarity(erd1, erd2)
    similarity = alpha * entities_sim + beta * rels_sim + lambda_val * features_sim
    return similarity

def knn_predict(test_erd: Dict, training_erds: List[Dict], training_grades: List[float], 
                A: float, B: float, C: float, D: float,
                T: float, U: float, V: float, X: float, Y: float, Z: float,
                alpha: float, beta: float, lambda_val: float, c_type_similarity: float,
                k: int, threshold: float) -> float:
    """Predict grade for a test ERD using KNN regression."""
    similarities = []
    for train_erd, train_grade in zip(training_erds, training_grades):
        if train_grade is not None:
            sim = erd_similarity(test_erd, train_erd, A, B, C, D, T, U, V, X, Y, Z, 
                                 alpha, beta, lambda_val, c_type_similarity)
            similarities.append((sim, train_grade))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    filtered_similarities = [(sim, grade) for sim, grade in similarities if sim >= threshold]
    
    if not filtered_similarities:
        valid_grades = [g for g in training_grades if g is not None]
        if valid_grades:
            return np.mean(valid_grades)
        return 0.0
    
    top_k = filtered_similarities[:k]
    if not top_k:
        valid_grades = [g for g in training_grades if g is not None]
        if valid_grades:
            return np.mean(valid_grades)
        return 0.0
    
    weights = [sim for sim, _ in top_k]
    grades = [grade for _, grade in top_k]
    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(grades)
    prediction = sum(w * g for w, g in zip(weights, grades)) / total_weight
    return prediction

def load_erd_data(dataset_dir: str) -> Dict[int, Dict]:
    """Load all ERD JSON files from a dataset directory."""
    erds = {}
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            try:
                erd_num = int(filename.split('_')[0])
                filepath = os.path.join(dataset_dir, filename)
                with open(filepath, 'r') as f:
                    erd_data = json.load(f)
                    erds[erd_num] = erd_data
            except (ValueError, json.JSONDecodeError) as e:
                continue
    return erds

def load_grades(csv_path: str) -> Dict[int, Tuple[float, float]]:
    """Load grades from ERD_grades.csv."""
    grades = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            erd_no = int(row['ERD_No'])
            grade1 = float(row['dataset1_grade']) if row['dataset1_grade'] else None
            grade2 = float(row['dataset2_grade']) if row['dataset2_grade'] else None
            grades[erd_no] = (grade1, grade2)
    return grades

def evaluate_hyperparameters(params: Dict) -> Tuple[float, float, float]:
    """Evaluate hyperparameters using leave-one-out cross-validation."""
    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']
    T = params['T']
    U = params['U']
    V = params['V']
    X = params['X']
    Y = params['Y']
    Z = params['Z']
    alpha = params['alpha']
    beta = params['beta']
    lambda_val = params['lambda']
    c_type_similarity = params['c_type_similarity']
    K = params['K']
    q_threshold = params['q_threshold']
    
    # Load training data
    train_dataset1_dir = 'for_students/Dataset1'
    train_dataset2_dir = 'for_students/Dataset2'
    grades_csv = 'for_students/ERD_grades.csv'
    
    train_erds1 = load_erd_data(train_dataset1_dir)
    train_erds2 = load_erd_data(train_dataset2_dir)
    grades = load_grades(grades_csv)
    
    # Prepare training data
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
    
    # Dataset 1 predictions
    predictions1 = []
    actuals1 = []
    for i in range(len(train_data1)):
        test_erd = train_data1[i]
        actual_grade = train_grades1[i]
        train_erds_loo = train_data1[:i] + train_data1[i+1:]
        train_grades_loo = train_grades1[:i] + train_grades1[i+1:]
        predicted_grade = knn_predict(test_erd, train_erds_loo, train_grades_loo,
                                     A, B, C, D, T, U, V, X, Y, Z,
                                     alpha, beta, lambda_val, c_type_similarity,
                                     K, q_threshold)
        predictions1.append(predicted_grade)
        actuals1.append(actual_grade)
    
    # Dataset 2 predictions
    predictions2 = []
    actuals2 = []
    for i in range(len(train_data2)):
        test_erd = train_data2[i]
        actual_grade = train_grades2[i]
        train_erds_loo = train_data2[:i] + train_data2[i+1:]
        train_grades_loo = train_grades2[:i] + train_grades2[i+1:]
        predicted_grade = knn_predict(test_erd, train_erds_loo, train_grades_loo,
                                     A, B, C, D, T, U, V, X, Y, Z,
                                     alpha, beta, lambda_val, c_type_similarity,
                                     K, q_threshold)
        predictions2.append(predicted_grade)
        actuals2.append(actual_grade)
    
    # Calculate RMSE
    predictions1 = np.array(predictions1)
    actuals1 = np.array(actuals1)
    predictions2 = np.array(predictions2)
    actuals2 = np.array(actuals2)
    
    rmse1 = np.sqrt(np.mean((predictions1 - actuals1) ** 2))
    rmse2 = np.sqrt(np.mean((predictions2 - actuals2) ** 2))
    all_predictions = np.concatenate([predictions1, predictions2])
    all_actuals = np.concatenate([actuals1, actuals2])
    rmse_combined = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
    
    return rmse1, rmse2, rmse_combined

def generate_param_combinations_random(n_samples=200):
    """Generate random hyperparameter combinations to test."""
    import random
    random.seed(42)
    np.random.seed(42)
    
    combinations = []
    
    for _ in range(n_samples):
        # Entity weights: A, B, C, D (must sum to 1)
        # Generate random weights that sum to 1
        weights = np.random.dirichlet([1, 1, 1, 1])
        A, B, C, D = weights[0], weights[1], weights[2], weights[3]
        
        # Relationship weights: T, U, V, X, Y, Z (must sum to 1)
        weights = np.random.dirichlet([1, 1, 1, 1, 1, 1])
        T, U, V, X, Y, Z = weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]
        
        # ERD weights: alpha, beta, lambda (must sum to 1)
        weights = np.random.dirichlet([1, 1, 1])
        alpha, beta, lambda_val = weights[0], weights[1], weights[2]
        
        # Other parameters
        c_type_sim = np.random.uniform(0.2, 0.8)
        K = random.choice([3, 5, 7, 10, 15])
        q_thresh = np.random.uniform(0.1, 0.5)
        
        combinations.append({
            'A': float(A), 'B': float(B), 'C': float(C), 'D': float(D),
            'T': float(T), 'U': float(U), 'V': float(V), 'X': float(X), 
            'Y': float(Y), 'Z': float(Z),
            'alpha': float(alpha), 'beta': float(beta), 'lambda': float(lambda_val),
            'c_type_similarity': float(c_type_sim),
            'K': int(K),
            'q_threshold': float(q_thresh)
        })
    
    return combinations

def generate_param_combinations_focused():
    """Generate focused hyperparameter combinations around current values."""
    # Entity weights: A, B, C, D (must sum to 1)
    entity_weights = [
        (0.1, 0.4, 0.1, 0.4),  # Current
        (0.05, 0.5, 0.05, 0.4),  # More weight on name
        (0.1, 0.3, 0.1, 0.5),  # More weight on attributes
        (0.15, 0.35, 0.15, 0.35),  # More balanced
        (0.2, 0.3, 0.2, 0.3),  # Even more balanced
        (0.05, 0.45, 0.05, 0.45),  # Name and attributes equal
    ]
    
    # Relationship weights: T, U, V, X, Y, Z (must sum to 1)
    rel_weights = [
        (0.1, 0.2, 0.3, 0.2, 0.1, 0.1),  # Current
        (0.05, 0.15, 0.4, 0.2, 0.1, 0.1),  # More weight on participating entities
        (0.1, 0.25, 0.25, 0.25, 0.075, 0.075),  # More balanced
        (0.15, 0.2, 0.25, 0.2, 0.1, 0.1),  # More weight on type
        (0.1, 0.2, 0.35, 0.2, 0.075, 0.075),  # Even more on participating
    ]
    
    # ERD weights: alpha, beta, lambda (must sum to 1)
    erd_weights = [
        (0.4, 0.4, 0.2),  # Current
        (0.45, 0.35, 0.2),  # More weight on entities
        (0.35, 0.45, 0.2),  # More weight on relationships
        (0.5, 0.3, 0.2),  # Even more on entities
        (0.3, 0.5, 0.2),  # Even more on relationships
        (0.4, 0.3, 0.3),  # More weight on features
        (0.5, 0.4, 0.1),  # Less on features
    ]
    
    # Other parameters
    c_type_sims = [0.3, 0.4, 0.5, 0.6, 0.7]
    K_values = [3, 5, 7, 10]
    q_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    
    combinations = []
    for A, B, C, D in entity_weights:
        for T, U, V, X, Y, Z in rel_weights:
            for alpha, beta, lambda_val in erd_weights:
                for c_type_sim in c_type_sims:
                    for K in K_values:
                        for q_thresh in q_thresholds:
                            combinations.append({
                                'A': A, 'B': B, 'C': C, 'D': D,
                                'T': T, 'U': U, 'V': V, 'X': X, 'Y': Y, 'Z': Z,
                                'alpha': alpha, 'beta': beta, 'lambda': lambda_val,
                                'c_type_similarity': c_type_sim,
                                'K': K,
                                'q_threshold': q_thresh
                            })
    
    return combinations

def main():
    """Main hyperparameter tuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for ERD grade prediction')
    parser.add_argument('--method', choices=['random', 'focused'], default='random',
                       help='Search method: random (random search) or focused (grid search around current values)')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of random samples to test (only for random method)')
    args = parser.parse_args()
    
    print("="*70)
    print("HYPERPARAMETER TUNING FOR ERD GRADE PREDICTION")
    print("="*70)
    print("\nThis will test multiple hyperparameter combinations using")
    print("leave-one-out cross-validation. This may take a while...\n")
    
    # Generate parameter combinations
    if args.method == 'random':
        param_combinations = generate_param_combinations_random(args.n_samples)
        print(f"Using random search with {len(param_combinations)} samples\n")
    else:
        param_combinations = generate_param_combinations_focused()
        print(f"Using focused grid search with {len(param_combinations)} combinations\n")
    
    # Add current hyperparameters as baseline
    current_params = {
        'A': 0.1, 'B': 0.4, 'C': 0.1, 'D': 0.4,
        'T': 0.1, 'U': 0.2, 'V': 0.3, 'X': 0.2, 'Y': 0.1, 'Z': 0.1,
        'alpha': 0.4, 'beta': 0.4, 'lambda': 0.2,
        'c_type_similarity': 0.5,
        'K': 5,
        'q_threshold': 0.3
    }
    param_combinations.insert(0, current_params)
    print("Including current hyperparameters as baseline (first test)\n")
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    for idx, params in enumerate(param_combinations, 1):
        print(f"[{idx}/{len(param_combinations)}] Testing hyperparameters...", end=' ', flush=True)
        try:
            rmse1, rmse2, rmse_combined = evaluate_hyperparameters(params)
            print(f"RMSE: {rmse_combined:.4f} (D1: {rmse1:.4f}, D2: {rmse2:.4f})")
            
            results.append({
                'params': params,
                'rmse1': rmse1,
                'rmse2': rmse2,
                'rmse_combined': rmse_combined
            })
            
            if rmse_combined < best_rmse:
                best_rmse = rmse_combined
                best_params = params
                print(f"  *** NEW BEST! Combined RMSE: {rmse_combined:.4f} ***")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    print(f"\nBest Combined RMSE: {best_rmse:.4f}")
    if best_params:
        print("\nBest Hyperparameters:")
        print(f"  Entity weights: A={best_params['A']:.2f}, B={best_params['B']:.2f}, "
              f"C={best_params['C']:.2f}, D={best_params['D']:.2f}")
        print(f"  Relationship weights: T={best_params['T']:.2f}, U={best_params['U']:.2f}, "
              f"V={best_params['V']:.2f}, X={best_params['X']:.2f}, "
              f"Y={best_params['Y']:.2f}, Z={best_params['Z']:.2f}")
        print(f"  ERD weights: alpha={best_params['alpha']:.2f}, "
              f"beta={best_params['beta']:.2f}, lambda={best_params['lambda']:.2f}")
        print(f"  c_type_similarity={best_params['c_type_similarity']:.2f}")
        print(f"  K={best_params['K']}")
        print(f"  q_threshold={best_params['q_threshold']:.2f}")
    
    # Sort results by combined RMSE
    results.sort(key=lambda x: x['rmse_combined'])
    
    print("\n" + "="*70)
    print("TOP 10 HYPERPARAMETER COMBINATIONS")
    print("="*70)
    for i, result in enumerate(results[:10], 1):
        p = result['params']
        print(f"\n{i}. Combined RMSE: {result['rmse_combined']:.4f} "
              f"(D1: {result['rmse1']:.4f}, D2: {result['rmse2']:.4f})")
        print(f"   A={p['A']:.2f}, B={p['B']:.2f}, C={p['C']:.2f}, D={p['D']:.2f} | "
              f"alpha={p['alpha']:.2f}, beta={p['beta']:.2f}, lambda={p['lambda']:.2f} | "
              f"K={p['K']}, q_thresh={p['q_threshold']:.2f}, c_type={p['c_type_similarity']:.2f}")
    
    # Save results to file
    with open('hyperparameter_tuning_results.txt', 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Combined RMSE: {best_rmse:.4f}\n\n")
        if best_params:
            f.write("Best Hyperparameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key} = {value}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("TOP 10 COMBINATIONS\n")
            f.write("="*70 + "\n")
            for i, result in enumerate(results[:10], 1):
                p = result['params']
                f.write(f"\n{i}. RMSE: {result['rmse_combined']:.4f}\n")
                for key, value in p.items():
                    f.write(f"   {key} = {value}\n")
    
    print("\nResults saved to hyperparameter_tuning_results.txt")

if __name__ == '__main__':
    main()

