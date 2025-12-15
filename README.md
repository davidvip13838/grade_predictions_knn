# ERD Grade Prediction

Implements a machine learning approach to predict grades for Entity Relationship Diagrams (ERDs) using custom graph-text similarity and K-Nearest Neighbors (KNN) regression.

## Overview

The system uses a custom similarity metric that combines:
- **Entity similarity**: Based on type, name, number of attributes, and attribute similarity
- **Relationship similarity**: Based on type, arity, participating entities, attributes, and cardinality
- **Additional ERD features**: Structural features like counts of entities, relationships, etc.

The approach uses sentence transformers to generate embeddings for text-based features and computes cosine similarity between ERDs to find the most similar training examples for grade prediction.

## Requirements

### Python Dependencies

Install the required packages:

```bash
pip install numpy scikit-learn sentence-transformers
```

### Required Packages:
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning utilities (cosine similarity)
- `sentence-transformers` - Text embeddings (uses `all-MiniLM-L6-v2` model)

## Project Structure

```
project_stage_2/
├── a4_custom_graph.py          # Main prediction script
├── compute_training_rmse.py    # Script to compute training RMSE
├── a4_custom_graph.csv          # Output predictions file
├── training_data.pkl           # Saved training data (generated after first run)
├── for_students/               # Training data
│   ├── Dataset1/              # Training ERDs for dataset 1
│   ├── Dataset2/              # Training ERDs for dataset 2
│   └── ERD_grades.csv         # Ground truth grades for training
└── for_testing/                # Test data
    ├── Dataset1/              # Test ERDs for dataset 1
    └── Dataset2/              # Test ERDs for dataset 2
```

## Usage

### Basic Usage

The script supports three modes of operation:

#### 1. Train Mode (Default if no saved data exists)
Loads training data, computes embeddings, saves them to disk, and runs inference:

```bash
python a4_custom_graph.py --mode train
```

#### 2. Inference Mode
Loads previously saved training data and runs inference (much faster):

```bash
python a4_custom_graph.py --mode inference
```

#### 3. Auto Mode (Default)
Automatically detects if saved training data exists:
- If `training_data.pkl` exists → uses inference mode
- Otherwise → uses train mode

```bash
python a4_custom_graph.py
```

### Command-Line Options

```bash
python a4_custom_graph.py [OPTIONS]

Options:
  --mode {train,inference,auto}
                        Mode: train (load and save training data), 
                        inference (load saved data), or auto 
                        (train if no saved data exists)
                        [default: auto]
  
  --save-path PATH      Path to save/load training data 
                        [default: training_data.pkl]
```

### Examples

```bash
# First run - will train and save data
python a4_custom_graph.py

# Subsequent runs - will use saved data (faster)
python a4_custom_graph.py

# Force train mode (recompute and save)
python a4_custom_graph.py --mode train

# Force inference mode (will error if no saved data)
python a4_custom_graph.py --mode inference

# Use custom save path
python a4_custom_graph.py --save-path my_training_data.pkl
```

## Output

The script generates `a4_custom_graph.csv` with predictions in the following format:

```csv
ERD_No,dataset1_grade,dataset2_grade
104,80.55,93.57
105,75.77,88.00
...
```

Each row contains:
- `ERD_No`: The ERD number
- `dataset1_grade`: Predicted grade for Dataset1 (if available)
- `dataset2_grade`: Predicted grade for Dataset2 (if available)

## Additional Scripts

### Compute Training RMSE

Evaluate the model's performance on the training dataset using leave-one-out cross-validation:

```bash
python compute_training_rmse.py
```

This script:
- Uses leave-one-out cross-validation on the training data
- Computes RMSE and MAE for both datasets
- Outputs performance metrics to the console

## Algorithm Details

### Hyperparameters

The model uses the following hyperparameters (defined in `a4_custom_graph.py`):

**Entity Similarity Weights:**
- `A = 0.1` - Type similarity weight
- `B = 0.4` - Name similarity weight
- `C = 0.1` - Number of attributes similarity weight
- `D = 0.4` - Attributes similarity weight

**Relationship Similarity Weights:**
- `T = 0.1` - Type similarity weight
- `U = 0.2` - Arity similarity weight
- `V = 0.3` - Participating entities similarity weight
- `X = 0.2` - Attributes similarity weight
- `Y = 0.1` - Max cardinality similarity weight
- `Z = 0.1` - Min cardinality similarity weight

**ERD Similarity Weights:**
- `alpha = 0.4` - All entities similarity weight
- `beta = 0.4` - All relationships similarity weight
- `lambda = 0.2` - Additional features similarity weight

**KNN Parameters:**
- `K = 5` - Number of nearest neighbors
- `q_threshold = 0.3` - Minimum similarity threshold

### Similarity Computation

1. **Entity Similarity**: Computes similarity between entities/weak entities based on:
   - Type (entity vs weak_entity)
   - Name (using sentence transformer embeddings)
   - Number of attributes
   - Attribute similarity (average embedding similarity)

2. **Relationship Similarity**: Computes similarity between relationships based on:
   - Type (relationship vs identifying relationship)
   - Arity (number of participating entities)
   - Participating entities (based on entity matches)
   - Attributes
   - Cardinality (for binary relationships)

3. **ERD Similarity**: Combines entity, relationship, and structural feature similarities

4. **Prediction**: Uses KNN regression with weighted average of top-K most similar training ERDs

## Performance

The model's performance on the training dataset (leave-one-out cross-validation):
- **Dataset1 RMSE**: ~6.71
- **Dataset2 RMSE**: ~8.27
- **Combined RMSE**: ~7.53

## Notes

- The first run in train mode will take longer as it needs to:
  - Load all training ERD files
  - Compute embeddings for text features
  - Save training data to disk
  
- Subsequent runs in inference mode are much faster as they:
  - Load pre-computed training data
  - Reuse cached embeddings
  - Only compute similarities for test ERDs

- The embedding cache is saved with training data, so common text features don't need to be recomputed

- Some ERD numbers may be missing from the datasets (as noted in `for_students/readme.txt`)

## File Format

ERD files are JSON format with the following structure:
```json
{
  "entities": [...],
  "relationships": [...]
}
```

Each entity contains:
- `name`: Entity name
- `kind`: "entity" or "weak_entity"
- `attributes`: List of attribute names
- `primary_keys`: List of primary key names

Each relationship contains:
- `kind`: "relationship" or "identifying relationship"
- `involved_entities`: List of participating entities with cardinality
- `attributes`: List of relationship attribute names

## License

MIT

