"""Cross-cutting — measurement infrastructure.

Responsibility: the retriever-agnostic eval harness, its dataset loader, and
the metrics.

**Hard rule:** the harness never learns domain knowledge. It receives a search
function and a dataset; it must not know that BM25, Qdrant, reranking or
companies exist. Filters travel from dataset to retriever as an opaque mapping.
That constraint is what makes A/B numbers across strategies honest.

Nothing else in the package imports `evaluation` — measurement must never be
able to change behaviour.
"""

from secfiler_rag.evaluation.dataset import EvalDataset, EvalItem, load_dataset
from secfiler_rag.evaluation.harness import EvalReport, ItemResult, SearchFn, evaluate
from secfiler_rag.evaluation.metrics import hit_rate, mean_reciprocal_rank

__all__ = [
    "EvalDataset",
    "EvalItem",
    "EvalReport",
    "ItemResult",
    "SearchFn",
    "evaluate",
    "hit_rate",
    "load_dataset",
    "mean_reciprocal_rank",
]
