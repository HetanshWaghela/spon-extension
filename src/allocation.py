"""
SPON Allocation Configuration System

Manages which layers and modules receive SPON biases,
and handles model-agnostic layer selection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
import yaml
import numpy as np


@dataclass
class SPONConfig:
    """Configuration for SPON bias allocation."""
    
    name: str
    layer_mask: List[int]  # Which layer indices get SPON
    modules: List[str]  # Which modules: ["down_proj", "o_proj", "q_proj", etc.]
    capacity_multiplier: Dict[int, float] = field(default_factory=dict)
    description: str = ""
    
    @property
    def num_spon_layers(self) -> int:
        return len(self.layer_mask)
    
    def get_capacity(self, layer_idx: int) -> float:
        """Get capacity multiplier for a specific layer."""
        return self.capacity_multiplier.get(layer_idx, 1.0)
    
    def has_spon(self, layer_idx: int) -> bool:
        """Check if a layer should have SPON biases."""
        return layer_idx in self.layer_mask


class AllocationBuilder:
    """Builds SPONConfig from YAML configuration."""
    
    def __init__(
        self,
        config_path: str = "configs/allocation_configs.yaml",
        data_driven_overrides_path: Optional[str] = None
    ):
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)
        self.defaults = self.raw_config.get("defaults", {})
        self.models = self.raw_config.get("models", {})
        self.configurations = self.raw_config.get("configurations", {})
        self.data_driven_overrides = {}

        if data_driven_overrides_path is not None:
            with open(data_driven_overrides_path, "r") as f:
                overrides = yaml.safe_load(f) or {}
            if not isinstance(overrides, dict):
                raise ValueError(
                    f"Data-driven overrides must be a mapping, got {type(overrides).__name__}"
                )
            self.data_driven_overrides = overrides
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get model-specific information."""
        # Normalize model name
        model_key = model_name.lower().replace("meta-llama/", "").replace("_", "-")
        for key in self.models:
            if key in model_key:
                return self.models[key]
        raise ValueError(f"Unknown model: {model_name}. Known models: {list(self.models.keys())}")
    
    def build_config(
        self,
        config_name: str,
        model_name: str,
        custom_layers: Optional[List[int]] = None
    ) -> SPONConfig:
        """
        Build a SPONConfig for a specific configuration and model.
        
        Args:
            config_name: Name of the configuration (e.g., "TOP-50")
            model_name: HuggingFace model name or short name
            custom_layers: Override layer selection with specific indices
        
        Returns:
            SPONConfig ready for use
        """
        if config_name not in self.configurations:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(self.configurations.keys())}")
        
        cfg = self.configurations[config_name]
        model_info = self.get_model_info(model_name)
        num_layers = model_info["num_layers"]
        
        # Determine layer mask
        if custom_layers is not None:
            layer_mask = custom_layers
        elif cfg.get("layer_fraction", 0) == 0:
            layer_mask = []
        else:
            layer_mask = self._compute_layer_mask(config_name, cfg, num_layers)
        
        # Compute capacity multipliers
        capacity_multiplier = self._compute_capacity(cfg, layer_mask, num_layers)
        
        return SPONConfig(
            name=config_name,
            layer_mask=layer_mask,
            modules=cfg.get("modules", ["down_proj"]),
            capacity_multiplier=capacity_multiplier,
            description=cfg.get("description", "")
        )
    
    def _compute_layer_mask(self, config_name: str, cfg: Dict, num_layers: int) -> List[int]:
        """Compute which layers get SPON based on configuration."""
        layer_fraction = cfg.get("layer_fraction", 1.0)
        layer_selection = cfg.get("layer_selection", "all")
        
        if layer_fraction is None or layer_selection == "data_driven":
            return self._get_data_driven_layer_mask(config_name, num_layers)
        
        num_selected = max(1, int(num_layers * layer_fraction))
        
        if layer_selection == "all":
            return list(range(num_layers))
        elif layer_selection == "top":
            # "Top" = early layers (closer to embeddings, index 0)
            return list(range(num_selected))
        elif layer_selection == "bottom":
            # "Bottom" = later layers (closer to output)
            return list(range(num_layers - num_selected, num_layers))
        elif layer_selection == "middle":
            start = (num_layers - num_selected) // 2
            return list(range(start, start + num_selected))
        elif layer_selection == "alternating":
            step = max(1, num_layers // num_selected)
            return list(range(0, num_layers, step))[:num_selected]
        else:
            raise ValueError(f"Unknown layer_selection: {layer_selection}")

    def _get_data_driven_layer_mask(self, config_name: str, num_layers: int) -> List[int]:
        """Resolve layer indices for data-driven configs from overrides."""
        override = self.data_driven_overrides.get(config_name)
        if override is None:
            raise ValueError(
                f"Config '{config_name}' requires data-driven layer indices, but no override "
                "was found. Provide --data_driven_overrides pointing to generated YAML."
            )

        if not isinstance(override, dict):
            raise ValueError(
                f"Override for '{config_name}' must be a mapping, got {type(override).__name__}"
            )

        layer_indices = override.get("layer_indices")
        if layer_indices is None:
            layer_indices = override.get("layers")

        if layer_indices is None:
            raise ValueError(
                f"Override for '{config_name}' is missing 'layer_indices' (or 'layers')."
            )
        if not isinstance(layer_indices, list) or not all(
            isinstance(i, int) for i in layer_indices
        ):
            raise ValueError(
                f"Override for '{config_name}' must contain a list[int] for layer indices."
            )

        unique_sorted = sorted(set(layer_indices))
        if any(i < 0 or i >= num_layers for i in unique_sorted):
            raise ValueError(
                f"Override for '{config_name}' has out-of-range layer indices for model "
                f"with {num_layers} layers: {unique_sorted}"
            )

        if not unique_sorted:
            raise ValueError(f"Override for '{config_name}' produced an empty layer set.")

        return unique_sorted
    
    def _compute_capacity(
        self, 
        cfg: Dict, 
        layer_mask: List[int], 
        num_layers: int
    ) -> Dict[int, float]:
        """Compute capacity multipliers per layer."""
        capacity_schedule = cfg.get("capacity_schedule")
        if capacity_schedule is None:
            return {}
        
        capacity_multiplier = {}
        top_25_end = num_layers // 4
        middle_50_end = top_25_end + num_layers // 2
        
        for layer_idx in layer_mask:
            if layer_idx < top_25_end:
                capacity_multiplier[layer_idx] = capacity_schedule.get("top_25", 1.0)
            elif layer_idx < middle_50_end:
                capacity_multiplier[layer_idx] = capacity_schedule.get("middle_50", 1.0)
            else:
                capacity_multiplier[layer_idx] = capacity_schedule.get("bottom_25", 1.0)
        
        return capacity_multiplier
    
    def list_configurations(self) -> List[str]:
        """List all available configuration names."""
        return list(self.configurations.keys())
    
    def get_relative_params(self, config_name: str) -> Optional[float]:
        """Get relative parameter count for a configuration."""
        if config_name in self.configurations:
            return self.configurations[config_name].get("relative_params")
        return None


def create_layer_mask_from_importance(
    importance_scores: np.ndarray,
    fraction: float = 0.5,
    method: Literal["top_k", "threshold", "cornerstone"] = "top_k",
    threshold: float = 0.5
) -> List[int]:
    """
    Create layer mask from importance scores (e.g., Shapley values).
    
    Args:
        importance_scores: Array of importance scores per layer
        fraction: Fraction of layers to select (for top_k method)
        method: Selection method
        threshold: Threshold for cornerstone method (PPL degradation ratio)
    
    Returns:
        List of layer indices to include
    """
    n_layers = len(importance_scores)
    
    if method == "top_k":
        k = max(1, int(n_layers * fraction))
        top_indices = np.argsort(importance_scores)[::-1][:k]
        return sorted(top_indices.tolist())
    
    elif method == "threshold":
        return [i for i, score in enumerate(importance_scores) if score >= threshold]
    
    elif method == "cornerstone":
        # Cornerstone layers: those where removal causes >threshold relative PPL increase
        # importance_scores here should be normalized PPL degradation ratios
        return [i for i, score in enumerate(importance_scores) if score > threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}")


# Convenience functions
def get_config(config_name: str, model_name: str, config_path: str = None) -> SPONConfig:
    """Quick helper to get a SPON configuration."""
    path = config_path or "configs/allocation_configs.yaml"
    builder = AllocationBuilder(path)
    return builder.build_config(config_name, model_name)


def get_all_configs(model_name: str, config_path: str = None) -> Dict[str, SPONConfig]:
    """Get all configurations for a model."""
    path = config_path or "configs/allocation_configs.yaml"
    builder = AllocationBuilder(path)
    configs = {}
    for name in builder.list_configurations():
        try:
            configs[name] = builder.build_config(name, model_name)
        except ValueError:
            # Data-driven configs may require explicit overrides.
            continue
    return configs
