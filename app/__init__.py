"""Source selection experiment pipeline package."""

from importlib.metadata import PackageNotFoundError, version

from app.api import app as analytics_api
from app.api import calculate_all_metrics, load_db, sync_outputs_to_db
from app.experiment import build_condition_bundles, build_selection_prompt, selection_response_json_schema
from app.models import OllamaClient, OllamaGeneration
from app.parsing import ParseResult, parse_model_response
from app.schemas import (
	Article,
	ConditionName,
	ExperimentRequest,
	ModelDecision,
	ModelManifest,
	ModelSpec,
	ParseStatus,
	PreparedIncident,
	PresentedArticle,
)

try:
	__version__ = version("sourcerers")
except PackageNotFoundError:  # pragma: no cover
	__version__ = "0.0.0"

__all__ = [
	"__version__",
	"analytics_api",
	"calculate_all_metrics",
	"load_db",
	"sync_outputs_to_db",
	"build_condition_bundles",
	"build_selection_prompt",
	"selection_response_json_schema",
	"OllamaClient",
	"OllamaGeneration",
	"ParseResult",
	"parse_model_response",
	"Article",
	"ConditionName",
	"ExperimentRequest",
	"ModelDecision",
	"ModelManifest",
	"ModelSpec",
	"ParseStatus",
	"PreparedIncident",
	"PresentedArticle",
]
