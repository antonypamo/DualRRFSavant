from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================
# 1. Embedding layer
# =====================

@dataclass
class RRFEmbedder:
    model_id: str

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_id)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# =====================
# 2. RRF metrics (Φ, Ω, SRRF, etc.)
# =====================

@dataclass
class RRFMetricExtractor:
    """
    Aquí conectas tus funciones reales de Φ, Ω, SRRF, CRRF, E_H, etc.
    Input: embeddings (respuesta completa) + opcionalmente embeddings del KB / query.
    Output: vector de features (15 dimensiones, por ejemplo).
    """

    def compute_features(
        self,
        cand_embs: np.ndarray,
        kb_embs: np.ndarray | None = None,
        query_emb: np.ndarray | None = None
    ) -> np.ndarray:
        # --- EJEMPLO DE PLACEHOLDER TÉCNICO ---
        # Podrías implementar:
        # - coherencia interna: media de la matriz de similitud entre frases de la respuesta
        # - alineación con KB: max/mean coseno contra EMB_KB
        # - "phi", "omega": funciones sobre espectro de la matriz de similitud, etc.
        sim_mat = cosine_similarity(cand_embs, cand_embs)
        if sim_mat.shape[0] > 1:
            mask = np.triu_indices_from(sim_mat, k=1)
            coherencia_interna = float(sim_mat[mask].mean())
        else:
            coherencia_interna = 1.0  # si solo hay una frase, asumimos coherencia máxima

        # Dummy features (ejemplo, reemplazar por los tuyos reales)
        phi = coherencia_interna
        omega = 1.0 - coherencia_interna
        s_rrf = coherencia_interna ** 2
        c_rrf = coherencia_interna
        e_h = -coherencia_interna * np.log(max(coherencia_interna, 1e-8))

        # Placeholder de one-hot nodos Φ (ej. 10 nodos)
        phi_nodes = np.zeros(10)
        phi_nodes[0] = 1.0  # modo "Φ0" dummy

        features = np.concatenate([
            np.array([phi, omega, s_rrf, c_rrf, e_h]),
            phi_nodes
        ])
        return features  # shape (15,)


# =====================
# 3. Meta-modelo Savant (RRFSavantMetaLogit)
# =====================

@dataclass
class RRFQualityMetaModel:
    """
    Wrapper para el modelo de calidad (ej. LogisticRegression entrenado y guardado con joblib).
    """
    model_path: str

    def __post_init__(self):
        import joblib
        self.model = joblib.load(self.model_path)

    def predict_score(self, features: np.ndarray) -> float:
        """
        Devuelve probabilidad de 'resonant' (clase 1).
        """
        proba = self.model.predict_proba([features])[0]
        # Asumimos que la clase 1 es "resonant"
        return float(proba[1])


# =====================
# 4. Motor de búsqueda semántica RRF
# =====================

@dataclass
class RRFSemanticSearch:
    kb_sentences: List[str]
    kb_embeddings: np.ndarray

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        scores = cosine_similarity(query_emb[None, :], self.kb_embeddings)[0]
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs]


# =====================
# 5. Re-ranker conceptual
# =====================

@dataclass
class RRFReRanker:
    w_rrf_quality: float = 0.6
    w_kb_relevance: float = 0.3
    w_llm_conf: float = 0.1

    def rerank(
        self,
        candidates: List[str],
        quality_scores: List[float],
        kb_scores: List[float],
        llm_confidences: List[float]
    ) -> List[int]:
        scores = []
        for i in range(len(candidates)):
            s = (
                self.w_rrf_quality * quality_scores[i] +
                self.w_kb_relevance * kb_scores[i] +
                self.w_llm_conf * llm_confidences[i]
            )
            scores.append((i, s))
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores_sorted]


# =====================
# 6. Orquestación de todo el pipeline
# =====================

@dataclass
class RRFSavantPipeline:
    embedder: RRFEmbedder
    metric_extractor: RRFMetricExtractor
    quality_model: RRFQualityMetaModel
    semantic_search: RRFSemanticSearch
    reranker: RRFReRanker

    def run(
        self,
        query: str,
        candidate_answers: List[str],
        llm_confidences: List[float]
    ) -> Dict[str, Any]:
        # 1) Embedding de query
        query_emb = self.embedder.encode_one(query)

        # 2) Semantic search en KB
        kb_results = self.semantic_search.search(query_emb, top_k=5)
        kb_best_score = kb_results[0][1] if kb_results else 0.0

        # 3) Embeddings de candidatos (por ahora 1 embedding por respuesta)
        cand_embs = self.embedder.encode(candidate_answers)

        # 4) Features + meta-modelo de calidad RRF
        quality_scores = []
        kb_scores = []
        for i in range(len(candidate_answers)):
            cand_emb = cand_embs[i : i + 1]  # shape (1, dim)
            features = self.metric_extractor.compute_features(cand_embs=cand_emb)
            q_score = self.quality_model.predict_score(features)
            quality_scores.append(q_score)
            kb_scores.append(kb_best_score)  # simplificación: mismo score KB para todos

        # 5) Re-ranking
        ranking = self.reranker.rerank(
            candidates=candidate_answers,
            quality_scores=quality_scores,
            kb_scores=kb_scores,
            llm_confidences=llm_confidences
        )

        best_idx = ranking[0]
        return {
            "best_index": best_idx,
            "best_answer": candidate_answers[best_idx],
            "ranking": ranking,
            "quality_scores": quality_scores,
            "kb_scores": kb_scores,
            "llm_confidences": llm_confidences
        }
