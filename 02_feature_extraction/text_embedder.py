# Text Embedder
"""
TextEmbedder: Traditional text embedding methods for cyberbullying detection.
Provides TF-IDF, Word2Vec, and FastText embeddings.
Optimized for Kannada-English code-mixed text.
"""

import os
import re
import pickle
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. TF-IDF features disabled.")

try:
    from gensim.models import Word2Vec, FastText  # type: ignore
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("gensim not available. Word2Vec/FastText features disabled.")


class TextEmbedder:
    """
    Text embedding extractor for cyberbullying detection.
    
    Provides multiple embedding methods:
    - TF-IDF: Term Frequency-Inverse Document Frequency
    - Word2Vec: Word embeddings trained on corpus
    - FastText: Subword-aware embeddings (better for code-mixed text)
    
    Attributes:
        config: Configuration dictionary
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        word2vec_model: Trained Word2Vec model
        fasttext_model: Trained FastText model
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TextEmbedder.
        
        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config or self._default_config()
        
        # Initialize vectorizers/models
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.fasttext_model = None
        
        # Paths for saved models
        self.embeddings_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
        
        # Load pre-trained models if available
        self._load_pretrained_models()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'tfidf': {
                'enabled': True,
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True,
                'analyzer': 'word'
            },
            'word2vec': {
                'enabled': True,
                'vector_size': 100,
                'window': 5,
                'min_count': 2,
                'workers': 4,
                'sg': 1,  # Skip-gram
                'epochs': 10
            },
            'fasttext': {
                'enabled': True,
                'vector_size': 100,
                'window': 5,
                'min_count': 2,
                'workers': 4,
                'min_n': 3,  # Min char n-gram
                'max_n': 6   # Max char n-gram
            },
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': False,
                'remove_numbers': False,
                'remove_emojis': False
            }
        }
    
    def _load_pretrained_models(self) -> None:
        """Load pre-trained models from embeddings directory."""
        if not os.path.exists(self.embeddings_dir):
            return
        
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(self.embeddings_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            try:
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded pre-trained TF-IDF vectorizer")
            except Exception as e:
                logger.warning(f"Could not load TF-IDF vectorizer: {e}")
        
        # Load Word2Vec model
        if GENSIM_AVAILABLE:
            w2v_path = os.path.join(self.embeddings_dir, 'word2vec_model.bin')
            if os.path.exists(w2v_path):
                try:
                    self.word2vec_model = Word2Vec.load(w2v_path)
                    logger.info("Loaded pre-trained Word2Vec model")
                except Exception as e:
                    logger.warning(f"Could not load Word2Vec model: {e}")
            
            # Load FastText model
            ft_path = os.path.join(self.embeddings_dir, 'fasttext_model.bin')
            if os.path.exists(ft_path):
                try:
                    self.fasttext_model = FastText.load(ft_path)
                    logger.info("Loaded pre-trained FastText model")
                except Exception as e:
                    logger.warning(f"Could not load FastText model: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding."""
        if self.config.get('preprocessing', {}).get('lowercase', True):
            text = text.lower()
        
        if self.config.get('preprocessing', {}).get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if self.config.get('preprocessing', {}).get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        if self.config.get('preprocessing', {}).get('remove_emojis', False):
            # Remove emojis
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F900-\U0001F9FF"
                "]+",
                flags=re.UNICODE
            )
            text = emoji_pattern.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = self._preprocess_text(text)
        # Simple whitespace tokenization
        tokens = text.split()
        return [t for t in tokens if len(t) > 0]
    
    # =========================================================================
    # TF-IDF Methods
    # =========================================================================
    def fit_tfidf(self, texts: List[str]) -> 'TextEmbedder':
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            self
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for TF-IDF")
        
        processed_texts = [self._preprocess_text(t) for t in texts]
        
        tfidf_config = self.config.get('tfidf', {})
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 5000),
            ngram_range=tuple(tfidf_config.get('ngram_range', (1, 2))),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.95),
            sublinear_tf=tfidf_config.get('sublinear_tf', True),
            analyzer=tfidf_config.get('analyzer', 'word')
        )
        
        self.tfidf_vectorizer.fit(processed_texts)
        logger.info(f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features")
        
        return self
    
    def extract_tfidf(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            TF-IDF feature matrix (n_samples, n_features)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for TF-IDF")
        
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self._preprocess_text(t) for t in texts]
        
        if self.tfidf_vectorizer is None:
            # Fit and transform
            self.fit_tfidf(texts)
        
        return self.tfidf_vectorizer.transform(processed_texts).toarray()
    
    def save_tfidf(self, path: Optional[str] = None) -> None:
        """Save TF-IDF vectorizer to file."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted")
        
        if path is None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            path = os.path.join(self.embeddings_dir, 'tfidf_vectorizer.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer to {path}")
    
    # =========================================================================
    # Word2Vec Methods
    # =========================================================================
    def fit_word2vec(self, texts: List[str]) -> 'TextEmbedder':
        """
        Train Word2Vec model on texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            self
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required for Word2Vec")
        
        # Tokenize all texts
        sentences = [self._tokenize(t) for t in texts]
        
        w2v_config = self.config.get('word2vec', {})
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=w2v_config.get('vector_size', 100),
            window=w2v_config.get('window', 5),
            min_count=w2v_config.get('min_count', 2),
            workers=w2v_config.get('workers', 4),
            sg=w2v_config.get('sg', 1),
            epochs=w2v_config.get('epochs', 10)
        )
        
        logger.info(f"Trained Word2Vec model with {len(self.word2vec_model.wv)} words")
        return self
    
    def extract_word2vec(
        self,
        texts: Union[str, List[str]],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Extract Word2Vec embeddings from texts.
        
        Args:
            texts: Text or list of texts
            pooling: Pooling method ('mean', 'max', 'concat')
            
        Returns:
            Embedding matrix (n_samples, vector_size)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required for Word2Vec")
        
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained. Call fit_word2vec first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        vector_size = self.word2vec_model.vector_size
        
        for text in texts:
            tokens = self._tokenize(text)
            word_vectors = []
            
            for token in tokens:
                if token in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[token])
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                if pooling == 'mean':
                    embedding = np.mean(word_vectors, axis=0)
                elif pooling == 'max':
                    embedding = np.max(word_vectors, axis=0)
                else:
                    embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(vector_size)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save_word2vec(self, path: Optional[str] = None) -> None:
        """Save Word2Vec model to file."""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained")
        
        if path is None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            path = os.path.join(self.embeddings_dir, 'word2vec_model.bin')
        
        self.word2vec_model.save(path)
        logger.info(f"Saved Word2Vec model to {path}")
    
    # =========================================================================
    # FastText Methods
    # =========================================================================
    def fit_fasttext(self, texts: List[str]) -> 'TextEmbedder':
        """
        Train FastText model on texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            self
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required for FastText")
        
        # Tokenize all texts
        sentences = [self._tokenize(t) for t in texts]
        
        ft_config = self.config.get('fasttext', {})
        self.fasttext_model = FastText(
            sentences=sentences,
            vector_size=ft_config.get('vector_size', 100),
            window=ft_config.get('window', 5),
            min_count=ft_config.get('min_count', 2),
            workers=ft_config.get('workers', 4),
            min_n=ft_config.get('min_n', 3),
            max_n=ft_config.get('max_n', 6)
        )
        
        logger.info(f"Trained FastText model with {len(self.fasttext_model.wv)} words")
        return self
    
    def extract_fasttext(
        self,
        texts: Union[str, List[str]],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Extract FastText embeddings from texts.
        
        Args:
            texts: Text or list of texts
            pooling: Pooling method ('mean', 'max')
            
        Returns:
            Embedding matrix (n_samples, vector_size)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required for FastText")
        
        if self.fasttext_model is None:
            raise ValueError("FastText model not trained. Call fit_fasttext first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        vector_size = self.fasttext_model.vector_size
        
        for text in texts:
            tokens = self._tokenize(text)
            word_vectors = []
            
            for token in tokens:
                # FastText can generate vectors for OOV words
                try:
                    word_vectors.append(self.fasttext_model.wv[token])
                except KeyError:
                    pass
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                if pooling == 'mean':
                    embedding = np.mean(word_vectors, axis=0)
                elif pooling == 'max':
                    embedding = np.max(word_vectors, axis=0)
                else:
                    embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(vector_size)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save_fasttext(self, path: Optional[str] = None) -> None:
        """Save FastText model to file."""
        if self.fasttext_model is None:
            raise ValueError("FastText model not trained")
        
        if path is None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            path = os.path.join(self.embeddings_dir, 'fasttext_model.bin')
        
        self.fasttext_model.save(path)
        logger.info(f"Saved FastText model to {path}")
    
    # =========================================================================
    # Combined Methods
    # =========================================================================
    def extract(
        self,
        texts: Union[str, List[str]],
        methods: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings using multiple methods.
        
        Args:
            texts: Text or list of texts
            methods: List of methods to use ('tfidf', 'word2vec', 'fasttext')
            
        Returns:
            Dictionary of embedding matrices
        """
        if methods is None:
            methods = ['tfidf']
        
        if isinstance(texts, str):
            texts = [texts]
        
        results = {}
        
        if 'tfidf' in methods and SKLEARN_AVAILABLE:
            try:
                results['tfidf'] = self.extract_tfidf(texts)
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}")
        
        if 'word2vec' in methods and GENSIM_AVAILABLE and self.word2vec_model:
            try:
                results['word2vec'] = self.extract_word2vec(texts)
            except Exception as e:
                logger.warning(f"Word2Vec extraction failed: {e}")
        
        if 'fasttext' in methods and GENSIM_AVAILABLE and self.fasttext_model:
            try:
                results['fasttext'] = self.extract_fasttext(texts)
            except Exception as e:
                logger.warning(f"FastText extraction failed: {e}")
        
        return results
    
    def fit(self, texts: List[str], methods: Optional[List[str]] = None) -> 'TextEmbedder':
        """
        Fit all embedding models.
        
        Args:
            texts: Training texts
            methods: Methods to train
            
        Returns:
            self
        """
        if methods is None:
            methods = ['tfidf', 'word2vec', 'fasttext']
        
        if 'tfidf' in methods and SKLEARN_AVAILABLE:
            self.fit_tfidf(texts)
        
        if 'word2vec' in methods and GENSIM_AVAILABLE:
            self.fit_word2vec(texts)
        
        if 'fasttext' in methods and GENSIM_AVAILABLE:
            self.fit_fasttext(texts)
        
        return self
    
    def get_vocabulary_size(self) -> Dict[str, int]:
        """Get vocabulary sizes for all models."""
        sizes = {}
        
        if self.tfidf_vectorizer:
            sizes['tfidf'] = len(self.tfidf_vectorizer.vocabulary_)
        
        if self.word2vec_model:
            sizes['word2vec'] = len(self.word2vec_model.wv)
        
        if self.fasttext_model:
            sizes['fasttext'] = len(self.fasttext_model.wv)
        
        return sizes
    
    def get_similar_words(
        self,
        word: str,
        topn: int = 10,
        model: str = 'word2vec'
    ) -> List[tuple]:
        """
        Find similar words using trained model.
        
        Args:
            word: Query word
            topn: Number of similar words
            model: Model to use ('word2vec' or 'fasttext')
            
        Returns:
            List of (word, similarity) tuples
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim required")
        
        word = word.lower()
        
        if model == 'word2vec' and self.word2vec_model:
            if word in self.word2vec_model.wv:
                return self.word2vec_model.wv.most_similar(word, topn=topn)
        elif model == 'fasttext' and self.fasttext_model:
            try:
                return self.fasttext_model.wv.most_similar(word, topn=topn)
            except KeyError:
                pass
        
        return []
    
    def __repr__(self) -> str:
        """String representation."""
        sizes = self.get_vocabulary_size()
        return (
            f"TextEmbedder("
            f"tfidf={sizes.get('tfidf', 'not fitted')}, "
            f"word2vec={sizes.get('word2vec', 'not fitted')}, "
            f"fasttext={sizes.get('fasttext', 'not fitted')})"
        )


if __name__ == "__main__":
    # Quick test
    embedder = TextEmbedder()
    
    test_texts = [
        "nee tumba irritating agthiya yaar",
        "This is very annoying behavior",
        "exam tumba tough aaytu",
        "Stop messaging me!",
    ]
    
    print("TextEmbedder Test")
    print("=" * 50)
    
    # Fit and extract TF-IDF
    if SKLEARN_AVAILABLE:
        embedder.fit_tfidf(test_texts)
        tfidf_features = embedder.extract_tfidf(test_texts)
        print(f"TF-IDF shape: {tfidf_features.shape}")
    
    # Fit and extract Word2Vec
    if GENSIM_AVAILABLE:
        embedder.fit_word2vec(test_texts)
        w2v_features = embedder.extract_word2vec(test_texts)
        print(f"Word2Vec shape: {w2v_features.shape}")
    
    print(f"\nVocabulary sizes: {embedder.get_vocabulary_size()}")
