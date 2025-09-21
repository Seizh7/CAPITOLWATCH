# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""Product embeddings generator: custom, TF-IDF, Word2Vec, GloVe methods."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from capitolwatch.services.product_embeddings import store_embeddings
from capitolwatch.services.products import (
    get_all_products_for_embeddings,
    get_product_features
)
from config import CONFIG

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

SUPPORTED_METHODS = ["custom_financial", "tfidf_basic", "word2vec", "glove"]


def create_custom_financial_embedding(
    products: List[Dict]
) -> Tuple[np.ndarray, Dict]:
    """
    Generate product embeddings from categorical and numerical financial
    features.

    The embedding is composed of:
        - Encoded categorical variables (LabelEncoder)
        - Scaled numerical variables (StandardScaler)

    Args:
        products: List of product dictionaries with financial information

    Returns:
        A tuple (embedding_matrix, metadata) where:
            embedding_matrix: 2-D numpy array [n_products, n_features]
            metadata: dictionary describing feature sets and dimensions
    """
    if not products:
        raise ValueError("No products provided for embedding generation.")

    embeddings = []

    for product in products:
        # Extract product features using the service layer
        features = get_product_features(product)

        # Convert categorical values to stable numeric representations
        categorical_nums = []
        for cat_value in features["categorical"].values():
            # Use hash modulo to get consistent numbers between 0-999
            categorical_nums.append(hash(str(cat_value)) % 1000)

        # Use numerical features directly without scaling
        numerical_nums = list(features["numerical"].values())

        # Combine categorical and numerical features into single vector
        product_embedding = categorical_nums + numerical_nums
        embeddings.append(product_embedding)

    # Convert to numpy array for efficient computation
    embeddings_array = np.array(embeddings, dtype=float)

    # Handle any NaN or infinite values by replacing with zeros
    embeddings_array = np.nan_to_num(embeddings_array, nan=0.0)

    # Prepare metadata for tracking and debugging
    metadata = {
        "method": "custom_financial",
        "total_dimensions": embeddings_array.shape[1],
        "description": "Hash-based categorical + direct numerical features"
    }

    return embeddings_array, metadata


def create_tfidf_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create TF-IDF embeddings from product text features.

    This method converts product names, sectors, and other text features
    into numerical vectors using Term Frequency-Inverse Document Frequency,
    which highlights important words that are specific to each product.

    Args:
        products: List of product dictionaries with text information

    Returns:
        A tuple (embedding_matrix, metadata) where:
            embedding_matrix: 2-D numpy array [n_products, n_features]
            metadata: dictionary with vocabulary size and dimensions
    """
    # Combine all text features into documents
    documents = []
    for product in products:
        features = get_product_features(product)
        # Join all text features into one document per product
        doc = " ".join([str(f) for f in features["text_features"] if f])
        documents.append(doc.lower().strip())

    # Create TF-IDF vectors with limited vocabulary
    vectorizer = TfidfVectorizer(
        max_features=50,        # Keep only top 50 words
        stop_words="english",   # Remove common English words
        min_df=1,              # Word must appear at least once
        max_df=0.95            # Ignore words in >95% of documents
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Prepare metadata for tracking
    metadata = {
        "method": "tfidf_basic",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "total_dimensions": tfidf_matrix.shape[1]
    }

    return tfidf_matrix.toarray(), metadata


def create_word2vec_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create Word2Vec embeddings from product text features.

    This method learns word relationships from product descriptions and
    creates dense vector representations by averaging word embeddings
    for each product.

    Args:
        products: List of product dictionaries with text information

    Returns:
        A tuple (embedding_matrix, metadata) where:
            embedding_matrix: 2-D numpy array [n_products, 50]
            metadata: dictionary with vocabulary size and dimensions
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim library is required for Word2Vec embeddings")

    # Prepare documents as tokenized lists
    documents = []
    for product in products:
        # Extract and tokenize text features
        text = " ".join(get_product_features(product)["text_features"])
        tokens = simple_preprocess(text, min_len=2)
        if tokens:
            documents.append(tokens)

    if not documents:
        raise ValueError("No valid text documents found for Word2Vec training")

    # Train Word2Vec model on product descriptions
    model = Word2Vec(
        documents,
        vector_size=50,    # 50-dimensional vectors
        window=3,          # Consider 3 words on each side
        min_count=1,       # Include all words
        epochs=5           # Training iterations
    )

    # Create embeddings by averaging word vectors for each product
    embeddings = []
    for doc in documents:
        # Get vectors for all words in document
        word_vectors = [model.wv[word] for word in doc if word in model.wv]
        # Average all word vectors or use zero vector if no words found
        if word_vectors:
            product_vector = np.mean(word_vectors, axis=0)
        else:
            product_vector = np.zeros(50)
        embeddings.append(product_vector)

    metadata = {
        "method": "word2vec",
        "vocabulary_size": len(model.wv.key_to_index),
        "total_dimensions": 50
    }

    return np.array(embeddings), metadata


def create_glove_embedding(products: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create simplified GloVe-style embeddings using word co-occurrence.

    This method counts how often words appear together, then compresses
    this information into dense vectors that represent each product.

    Args:
        products: List of product dictionaries with text information

    Returns:
        A tuple (embedding_matrix, metadata) where:
            embedding_matrix: 2-D numpy array [n_products, embedding_dim]
            metadata: dictionary with vocabulary size and dimensions
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim library is required for text preprocessing")

    # Get all words from all products
    documents = []
    for product in products:
        text = " ".join(get_product_features(product)["text_features"])
        tokens = simple_preprocess(text, min_len=2)
        if tokens:
            documents.append(tokens)

    if not documents:
        raise ValueError("No valid text documents found for GloVe processing")

    # Create simple word list and count matrix
    all_words = sorted({word for doc in documents for word in doc})
    word_index = {word: i for i, word in enumerate(all_words)}

    # Count how often words appear together (simplified)
    word_counts = np.zeros((len(all_words), len(all_words)))
    for doc in documents:
        for i, word1 in enumerate(doc):
            for word2 in doc[i:i+3]:  # Look at next 2 words only
                if word1 != word2:
                    word_counts[word_index[word1], word_index[word2]] += 1

    # Use math to compress the word relationships
    embedding_size = min(30, len(all_words) - 1) if len(all_words) > 1 else 1
    U, s, _ = np.linalg.svd(word_counts + 0.000001)  # to avoid errors
    word_vectors = U[:, :embedding_size] * np.sqrt(s[:embedding_size])

    # Create one vector per product by averaging its word vectors
    embeddings = []
    for doc in documents:
        doc_vectors = [word_vectors[word_index[word]] for word in doc]
        if doc_vectors:
            product_vector = np.mean(doc_vectors, axis=0)
        else:
            product_vector = np.zeros(embedding_size)
        embeddings.append(product_vector)

    metadata = {
        "method": "glove",
        "vocabulary_size": len(all_words),
        "total_dimensions": embedding_size
    }

    return np.array(embeddings), metadata


def generate_embeddings(
    products: List[Dict],
    method: str = "custom_financial"
) -> Tuple[List[int], np.ndarray, List[str], Dict]:
    """
    Generate embeddings for given products

    Args:
        products: List of product dictionaries
        method: Embedding method to use (see SUPPORTED_METHODS)

    Returns:
        Tuple of (product_ids, embeddings, features_used, metadata)
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. Available: {SUPPORTED_METHODS}"
        )

    if not products:
        raise ValueError("No products provided for embedding generation")

    product_ids = [p["id"] for p in products]
    print(f"Generating '{method}' embeddings for {len(products)} products...")

    if method == "custom_financial":
        embeddings, metadata = create_custom_financial_embedding(products)
        features_used = ["categorical_financial", "numerical_financial"]
    elif method == "tfidf_basic":
        embeddings, metadata = create_tfidf_embedding(products)
        features_used = ["text_features"]
    elif method == "word2vec":
        embeddings, metadata = create_word2vec_embedding(products)
        features_used = ["text_features_word2vec"]
    elif method == "glove":
        embeddings, metadata = create_glove_embedding(products)
        features_used = ["text_features_glove"]
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented")

    return product_ids, embeddings, features_used, metadata


def generate_and_store_products_embeddings(
    config: Optional[object] = None,
    method: str = "custom_financial"
) -> Dict:
    """
    Generate and store embeddings for all products in the database.

    Args:
        config: Database configuration
        method: Embedding method to use (see SUPPORTED_METHODS)

    Returns:
        Dict with method, product count, embedding dimension, and metadata
    """
    config = config or CONFIG

    products = get_all_products_for_embeddings(config=config)
    if not products:
        print("No products found in database.")
        return {}

    try:
        product_ids, embeddings, features_used, metadata = generate_embeddings(
            products, method
        )

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
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


if __name__ == "__main__":
    print("Testing all embedding methods")

    products = get_all_products_for_embeddings(config=CONFIG)
    if not products:
        print("No products found in database for testing.")
    else:
        test_products = products[:5]
        print(f"Testing with {len(test_products)} products")

        for method in SUPPORTED_METHODS:
            print(f"\nTesting method: {method.upper()}")
            try:
                (
                    product_ids, embeddings,
                    features_used, metadata
                ) = generate_embeddings(
                    products=test_products,
                    method=method
                )

                print(f"   Generated embeddings shape: {embeddings.shape}")
                print(f"   Product IDs: {product_ids}")
                print(f"   Features used: {features_used}")
                print(f"   Metadata: {metadata}")
                print(f"   Method '{method}' completed successfully")

            except ImportError as ie:
                print(f"   Method '{method}' skipped: {ie}")
            except Exception as me:
                print(f"   Method '{method}' failed: {me}")
