# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""
Product Embeddings - Machine Learning Features for Investment Analysis

Functions to create, store, and manage product embeddings for clustering
and similarity analysis of political investment portfolios.
Includes TF-IDF, custom financial features, Word2Vec, and GloVe methods.
"""

import re
from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from capitolwatch.services.product_embeddings import (
    store_embeddings, get_embeddings
)
from capitolwatch.services.products import get_all_products_for_embeddings
from config import CONFIG

# ML libraries for embeddings
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download required NLTK data (with error handling)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    GENSIM_AVAILABLE = True
except ImportError:
    print(
        "Warning: gensim or nltk not available. "
        "Word2Vec and GloVe embeddings will be disabled."
    )
    GENSIM_AVAILABLE = False


SUPPORTED_METHODS = [
    "tfidf_basic",       # TF-IDF on concatenated text features
    "custom_financial",  # Label-encoded categorical + scaled numerical
    "word2vec",          # Word2Vec embeddings on text features
    "glove"             # GloVe-style averaged embeddings
]

# Configuration for Word2Vec
WORD2VEC_CONFIG = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "workers": 4,
    "epochs": 10,
    "sg": 1
}

# Configuration for text preprocessing
TEXT_PREPROCESSING = {
    "min_length": 2,
    "remove_numbers": True,
    "remove_special_chars": True
}


# -------------------------------------------------------------------
# Text Preprocessing Utilities
# -------------------------------------------------------------------

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for embedding generation.

    Args:
        text: Raw text string

    Returns:
        List of preprocessed tokens
    """
    if not text or not isinstance(text, str):
        return []

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers if configured
    if TEXT_PREPROCESSING["remove_special_chars"]:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    if TEXT_PREPROCESSING["remove_numbers"]:
        text = re.sub(r'\d+', '', text)

    # Tokenize
    if GENSIM_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            tokens = [
                token for token in tokens
                if len(token) >= TEXT_PREPROCESSING["min_length"]
                and token not in stop_words
            ]
        except Exception:
            # Fallback to simple preprocessing
            min_len = TEXT_PREPROCESSING["min_length"]
            tokens = simple_preprocess(text, min_len=min_len)
    else:
        # Simple tokenization without NLTK
        tokens = text.split()
        tokens = [
            token for token in tokens
            if len(token) >= TEXT_PREPROCESSING["min_length"]
        ]

    return tokens


def prepare_documents_for_embedding(products: List[Dict]) -> List[List[str]]:
    """
    Prepare product documents for Word2Vec training.

    Args:
        products: List of product dictionaries

    Returns:
        List of tokenized documents
    """
    documents = []

    for product in products:
        features = get_vectorizable_features(product)

        # Combine all text features
        text_parts = []
        for feature in features["text_features"]:
            if feature and str(feature).strip():
                text_parts.append(str(feature))

        # Add categorical features as text
        for key, value in features["categorical_features"].items():
            if value and str(value).strip():
                text_parts.append(f"{key}_{str(value)}")

        # Create document
        full_text = " ".join(text_parts)
        tokens = preprocess_text(full_text)

        if tokens:  # Only add non-empty documents
            documents.append(tokens)

    return documents


# -------------------------------------------------------------------
# Feature Extraction
# -------------------------------------------------------------------

def get_vectorizable_features(product: Dict) -> Dict[str, Any]:
    """
    Extract and prepare features for vectorization.

    Args:
        product: Product dictionary from database

    Returns:
        Dict with categorized features ready for vectorization
    """
    return {
        "text_features": [
            product.get("name", ""),
            product.get("sector", ""),
            product.get("industry", ""),
            product.get("asset_class", ""),
            product.get("country", "")
        ],
        "categorical_features": {
            "type": product.get("type", ""),
            "sector": product.get("sector", ""),
            "industry": product.get("industry", ""),
            "asset_class": product.get("asset_class", ""),
            "market_cap_tier": product.get("market_cap_tier", ""),
            "risk_rating": product.get("risk_rating", ""),
            "country": product.get("country", ""),
            "currency": product.get("currency", "USD")
        },
        "numerical_features": {
            "market_cap": product.get("market_cap", 0) or 0,
            "beta": product.get("beta", 0) or 0,
            "dividend_yield": product.get("dividend_yield", 0) or 0,
            "expense_ratio": product.get("expense_ratio", 0) or 0,
            "is_etf": int(product.get("is_etf", 0) or 0),
            "is_mutual_fund": int(product.get("is_mutual_fund", 0) or 0),
            "is_index_fund": int(product.get("is_index_fund", 0) or 0)
        }
    }


# -------------------------------------------------------------------
# Embedding Generation Methods
# -------------------------------------------------------------------

def create_tfidf_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create TF-IDF embeddings from text features.

    Args:
        products: List of product dictionaries

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    # Build documents
    documents = []
    for product in products:
        features = get_vectorizable_features(product)
        doc = " ".join([str(f) for f in features["text_features"] if f])
        documents.append(doc.lower().strip())

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    metadata = {
        "vectorizer_params": {
            "max_features": vectorizer.max_features,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df
        },
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "vocabulary_size": len(vectorizer.vocabulary_)
    }

    return tfidf_matrix.toarray(), metadata


def create_custom_financial_embedding(
    products: List[Dict]
) -> Tuple[np.ndarray, Dict]:
    """
    Create embeddings based on categorical and numerical features.

    Args:
        products: List of product dictionaries

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    features_list = []
    categorical_values = {}
    numerical_values = {}

    # Extract features from all products
    for product in products:
        features = get_vectorizable_features(product)
        features_list.append(features)

        # Collect unique categorical values
        for key, value in features["categorical_features"].items():
            if key not in categorical_values:
                categorical_values[key] = set()
            categorical_values[key].add(str(value))

        # Collect numerical values
        for key, value in features["numerical_features"].items():
            if key not in numerical_values:
                numerical_values[key] = []
            numerical_values[key].append(float(value))

    # Create encoders for categorical features
    encoders = {}
    for key, values in categorical_values.items():
        encoder = LabelEncoder()
        encoder.fit(list(values))
        encoders[key] = encoder

    # Create scaler for numerical features
    scaler = StandardScaler()
    numerical_matrix = np.array([
        [features["numerical_features"][key]
         for key in sorted(numerical_values.keys())]
        for features in features_list
    ])
    scaled_numerical = scaler.fit_transform(numerical_matrix)

    # Encode categorical features
    categorical_encoded = []
    for features in features_list:
        encoded_row = []
        for key in sorted(categorical_values.keys()):
            value = str(features["categorical_features"][key])
            encoded_value = encoders[key].transform([value])[0]
            encoded_row.append(encoded_value)
        categorical_encoded.append(encoded_row)

    categorical_encoded = np.array(categorical_encoded)

    # Combine categorical and numerical features
    combined_embedding = np.hstack([categorical_encoded, scaled_numerical])

    metadata = {
        "categorical_features": list(sorted(categorical_values.keys())),
        "numerical_features": list(sorted(numerical_values.keys())),
        "categorical_dimensions": len(categorical_values),
        "numerical_dimensions": len(numerical_values),
        "total_dimensions": combined_embedding.shape[1]
    }

    return combined_embedding, metadata


def create_word2vec_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create Word2Vec embeddings from text features.

    Args:
        products: List of product dictionaries

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required for Word2Vec embeddings")

    # Prepare documents
    documents = prepare_documents_for_embedding(products)

    if not documents:
        raise ValueError("No valid documents found for Word2Vec training")

    # Train Word2Vec model
    model = Word2Vec(
        documents,
        vector_size=WORD2VEC_CONFIG["vector_size"],
        window=WORD2VEC_CONFIG["window"],
        min_count=WORD2VEC_CONFIG["min_count"],
        workers=WORD2VEC_CONFIG["workers"],
        epochs=WORD2VEC_CONFIG["epochs"],
        sg=WORD2VEC_CONFIG["sg"]
    )

    # Create product embeddings by averaging word vectors
    embeddings = []
    for document in documents:
        # Get vectors for words in document that are in vocabulary
        word_vectors = []
        for word in document:
            if word in model.wv:
                word_vectors.append(model.wv[word])

        if word_vectors:
            # Average word vectors for document
            doc_embedding = np.mean(word_vectors, axis=0)
        else:
            # Use zero vector if no words found
            doc_embedding = np.zeros(WORD2VEC_CONFIG["vector_size"])

        embeddings.append(doc_embedding)

    embeddings_matrix = np.array(embeddings)

    metadata = {
        "model_config": WORD2VEC_CONFIG.copy(),
        "vocabulary_size": len(model.wv.key_to_index),
        "total_documents": len(documents),
        "vector_size": WORD2VEC_CONFIG["vector_size"]
    }

    return embeddings_matrix, metadata


def create_glove_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create GloVe-style embeddings using global co-occurrence statistics.

    This is a simplified implementation that creates embeddings based on
    co-occurrence patterns in the product features.

    Args:
        products: List of product dictionaries

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    # Prepare documents
    documents = prepare_documents_for_embedding(products)

    if not documents:
        raise ValueError("No valid documents found for GloVe embedding")

    # Build vocabulary
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    vocab = list(vocab)
    vocab_size = len(vocab)

    if vocab_size == 0:
        raise ValueError("Empty vocabulary for GloVe embedding")

    # Create word-to-index mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    # Build co-occurrence matrix
    window_size = 5
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))

    for doc in documents:
        for i, word in enumerate(doc):
            word_idx = word_to_idx[word]
            # Look at context window
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word = doc[j]
                    context_idx = word_to_idx[context_word]
                    # Weight by distance (closer words get higher weights)
                    distance = abs(i - j)
                    weight = 1.0 / distance
                    cooccurrence_matrix[word_idx, context_idx] += weight

    # Simple dimensionality reduction using SVD
    embedding_dim = min(50, vocab_size - 1)  # Adaptive dimension
    if embedding_dim <= 0:
        embedding_dim = 1

    # Add small regularization to avoid numerical issues
    cooccurrence_matrix += 1e-6

    # Apply SVD
    U, s, Vt = np.linalg.svd(cooccurrence_matrix, full_matrices=False)

    # Take top components
    word_embeddings = U[:, :embedding_dim] * np.sqrt(s[:embedding_dim])

    # Create document embeddings by averaging word embeddings
    embeddings = []
    for doc in documents:
        doc_vectors = []
        for word in doc:
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                doc_vectors.append(word_embeddings[word_idx])

        if doc_vectors:
            doc_embedding = np.mean(doc_vectors, axis=0)
        else:
            doc_embedding = np.zeros(embedding_dim)

        embeddings.append(doc_embedding)

    embeddings_matrix = np.array(embeddings)

    metadata = {
        "vocabulary_size": vocab_size,
        "embedding_dimension": embedding_dim,
        "window_size": window_size,
        "total_documents": len(documents),
        "method": "simplified_glove"
    }

    return embeddings_matrix, metadata


def generate_embeddings_for_all_products(
    config,
    method: str = "custom_financial"
) -> Dict:
    """
    Generate embeddings for all products in the database.

    Args:
        config: Database configuration
        method: Embedding method to use

    Returns:
        Dict with generation results
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method '{method}' not supported. "
            f"Use one of: {SUPPORTED_METHODS}"
        )

    # Load products using service
    products = get_all_products_for_embeddings(config=config)

    if not products:
        print("No products found in database")
        return {}

    product_ids = [p["id"] for p in products]

    print(f"Generating {method} embeddings for {len(products)} products...")

    # Generate embeddings based on method
    if method == "tfidf_basic":
        embeddings, metadata = create_tfidf_embedding(products)
        features_used = [
            "name", "sector", "industry", "asset_class", "country"
        ]

    elif method == "custom_financial":
        embeddings, metadata = create_custom_financial_embedding(products)
        features_used = [
            "type", "sector", "industry", "asset_class",
            "market_cap_tier", "risk_rating", "country",
            "market_cap", "beta", "dividend_yield",
            "expense_ratio", "is_etf", "is_mutual_fund",
            "is_index_fund"
        ]

    elif method == "word2vec":
        embeddings, metadata = create_word2vec_embedding(products)
        features_used = [
            "name", "sector", "industry", "asset_class", "country", "type"
        ]

    elif method == "glove":
        embeddings, metadata = create_glove_embedding(products)
        features_used = [
            "name", "sector", "industry", "asset_class", "country", "type"
        ]

    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented")

    # Store in database using service
    store_embeddings(
        product_ids, embeddings, method,
        features_used, metadata, config=config
    )

    return {
        "method": method,
        "product_count": len(products),
        "embedding_dimension": embeddings.shape[1],
        "features_used": features_used,
        "metadata": metadata
    }


def compare_methods(config, methods: List[str]) -> Dict:
    """
    Compare different embedding methods.

    Args:
        config: Database configuration
        methods: List of methods to compare

    Returns:
        Dict with comparison statistics
    """
    comparison = {}

    for method in methods:
        embeddings_data = get_embeddings(method, config=config)
        if embeddings_data["embeddings"] is not None:
            embeddings = embeddings_data["embeddings"]
            comparison[method] = {
                "product_count": len(embeddings_data["product_ids"]),
                "dimension": embeddings.shape[1],
                "features_used": embeddings_data["features_used"],
                "mean_magnitude": float(
                    np.mean(np.linalg.norm(embeddings, axis=1))
                ),
                "variance_explained": float(np.var(embeddings)),
            }

    return comparison


def generate_embeddings_for_method(config, method: str) -> Dict:
    """
    Generate embeddings for a single method.

    Args:
        config: Database configuration
        method: Embedding method name

    Returns:
        Dict containing status and metadata
    """
    # Skip if gensim/nltk is required but not available
    if method in ["word2vec", "glove"] and not GENSIM_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "gensim/nltk not available"
        }

    try:
        result = generate_embeddings_for_all_products(config, method=method)
        if not result:
            return {
                "status": "failed",
                "reason": "No result returned"
            }

        return {
            "status": "success",
            "product_count": result["product_count"],
            "embedding_dimension": result["embedding_dimension"],
            "features_used": result["features_used"],
            "metadata": result["metadata"],
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def generate_all_embeddings(config) -> Dict:
    """
    Generate embeddings for all supported methods.

    Args:
        config: Database configuration

    Returns:
        Dict with results for each method
    """
    results: Dict[str, Dict] = {}

    print("Generating embeddings for all supported methods")

    for method in SUPPORTED_METHODS:
        print(f"\nMethod: {method}")
        result = generate_embeddings_for_method(config, method)

        results[method] = result
        status = result["status"]

        if status == "success":
            print(f"Generated {method} embeddings successfully")
            print(f"  - Products: {result['product_count']}")
            print(f"  - Dimensions: {result['embedding_dimension']}")
        elif status == "skipped":
            print(f"Skipped {method} ({result['reason']})")
        elif status == "failed":
            print(
                f"Failed to generate {method} embeddings "
                f"({result['reason']})"
            )
        elif status == "error":
            print(f"Error generating {method} embeddings: {result['error']}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    success_count = sum(
        1 for r in results.values() if r["status"] == "success"
    )
    print(f"Successful methods: {success_count}/{len(SUPPORTED_METHODS)}\n")

    for method, result in results.items():
        status = result["status"].upper()
        print(f"{method}: {status}")
        if status == "SUCCESS":
            print(f"  -> {result['product_count']} products, "
                  f"{result['embedding_dimension']} dimensions")

    return results


if __name__ == "__main__":
    results = generate_all_embeddings(CONFIG)
