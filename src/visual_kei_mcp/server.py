"""
Visual Kei Aesthetic Vocabulary MCP Server
==========================================

A Lushy brick that translates natural language descriptions into
authentic Visual Kei aesthetic parameters for image generation.

Three-Layer Architecture:
- Layer 1: Deterministic taxonomy lookup from YAML ologs (zero LLM cost)
- Layer 2: Structured parameter assembly (zero LLM cost)
- Layer 3: LLM synthesis interface (returns data for Claude)

Subgenres: Kote Kei, Oshare Kei, Nagoya Kei, Angura Kei, Eroguro Kei, Lolita Kei, Iryou Kei
Eras: Pioneer (1980s), Golden Age (1990s), Diversification (2000s), Neo-Revival (2010s+)
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import json
import yaml
import numpy as np

# Initialize MCP server
mcp = FastMCP("visual_kei_mcp")

# =============================================================================
# OLOG LOADING - Load taxonomy from YAML files
# =============================================================================

OLOG_DIR = Path(__file__).parent / "ologs"


def load_olog(filename: str) -> Dict[str, Any]:
    """Load a YAML olog file from the ologs directory."""
    filepath = OLOG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Olog file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Load all ologs at module initialization
try:
    SUBGENRES_OLOG = load_olog("subgenres.yaml")
    ERAS_OLOG = load_olog("eras.yaml")
    CONTEXTS_OLOG = load_olog("contexts.yaml")
    ARCHETYPES_OLOG = load_olog("archetypes.yaml")
    INTENTIONALITY_OLOG = load_olog("intentionality.yaml")
except FileNotFoundError as e:
    # Fallback for testing without ologs
    print(f"Warning: {e} - Using empty defaults")
    SUBGENRES_OLOG = {"subgenres": {}, "keywords": {}}
    ERAS_OLOG = {"eras": {}, "keywords": {}}
    CONTEXTS_OLOG = {"contexts": {}, "keywords": {}}
    ARCHETYPES_OLOG = {"archetypes": {}, "keywords": {}, "band_configurations": {}}
    INTENTIONALITY_OLOG = {"intentionality": {}, "intensity_levels": {}, "calculation_rules": {}}

# Extract taxonomy data from ologs
SUBGENRE_TAXONOMY = SUBGENRES_OLOG.get("subgenres", {})
ERA_TAXONOMY = ERAS_OLOG.get("eras", {})
CONTEXT_TAXONOMY = CONTEXTS_OLOG.get("contexts", {})
ARCHETYPE_TAXONOMY = ARCHETYPES_OLOG.get("archetypes", {})
INTENSITY_MAPPING = INTENTIONALITY_OLOG.get("intensity_levels", {})

# Extract keyword mappings
SUBGENRE_KEYWORDS = SUBGENRES_OLOG.get("keywords", {})
ERA_KEYWORDS = ERAS_OLOG.get("keywords", {})
CONTEXT_KEYWORDS = CONTEXTS_OLOG.get("keywords", {})
ARCHETYPE_KEYWORDS = ARCHETYPES_OLOG.get("keywords", {})


# =============================================================================
# ENUMS - Valid taxonomy values
# =============================================================================

class Subgenre(str, Enum):
    """Visual Kei subgenre classification."""
    KOTE_KEI = "kote_kei"
    OSHARE_KEI = "oshare_kei"
    NAGOYA_KEI = "nagoya_kei"
    ANGURA_KEI = "angura_kei"
    EROGURO_KEI = "eroguro_kei"
    LOLITA_KEI = "lolita_kei"
    IRYOU_KEI = "iryou_kei"


class Era(str, Enum):
    """Visual Kei historical era."""
    PIONEER_1980S = "pioneer_1980s"
    GOLDEN_AGE_1990S = "golden_age_1990s"
    DIVERSIFICATION_2000S = "diversification_2000s"
    NEO_REVIVAL_2010S = "neo_revival_2010s"


class Context(str, Enum):
    """Visual context/setting for the character."""
    STAGE_PERFORMANCE = "stage_performance"
    PHOTOSHOOT = "photoshoot"
    STREET_FASHION = "street_fashion"
    MUSIC_VIDEO = "music_video"
    ALBUM_COVER = "album_cover"


class Archetype(str, Enum):
    """Band member role archetype."""
    VOCALIST = "vocalist"
    GUITARIST = "guitarist"
    BASSIST = "bassist"
    DRUMMER = "drummer"
    KEYBOARDIST = "keyboardist"


class IntensityLevel(str, Enum):
    """Styling intensity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


# =============================================================================
# PYDANTIC MODELS - Input validation
# =============================================================================

class SubgenreInput(BaseModel):
    """Input for retrieving subgenre taxonomy."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    subgenre: Subgenre = Field(
        ...,
        description="Visual Kei subgenre to retrieve"
    )


class EraInput(BaseModel):
    """Input for retrieving era taxonomy."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    era: Era = Field(
        ...,
        description="Visual Kei era to retrieve"
    )


class IntentAnalysisInput(BaseModel):
    """Input for analyzing user intent from natural language."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    prompt: str = Field(
        ...,
        description="Natural language description of desired Visual Kei aesthetic",
        min_length=3,
        max_length=2000
    )


class ParameterMappingInput(BaseModel):
    """Input for deterministic parameter mapping."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    subgenre: Subgenre = Field(..., description="Visual Kei subgenre")
    era: Optional[Era] = Field(default=Era.GOLDEN_AGE_1990S, description="Historical era")
    context: Optional[Context] = Field(default=Context.PHOTOSHOOT, description="Visual context")
    archetype: Optional[Archetype] = Field(default=None, description="Band member archetype")
    intensity: Optional[IntensityLevel] = Field(default=IntensityLevel.HIGH, description="Styling intensity")


class EnhancePromptInput(BaseModel):
    """Input for the complete enhancement workflow."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    base_prompt: str = Field(
        ...,
        description="Original prompt to enhance with Visual Kei aesthetics",
        min_length=3,
        max_length=2000
    )
    subgenre: Optional[Subgenre] = Field(default=None, description="Specific subgenre (auto-detected if not provided)")
    era: Optional[Era] = Field(default=None, description="Historical era (defaults to golden_age_1990s)")
    context: Optional[Context] = Field(default=None, description="Visual context (defaults to photoshoot)")
    archetype: Optional[Archetype] = Field(default=None, description="Band member archetype (optional)")
    intensity: Optional[IntensityLevel] = Field(default=IntensityLevel.HIGH, description="Styling intensity level")


class BandGenerationInput(BaseModel):
    """Input for generating complete band parameters."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    subgenre: Subgenre = Field(..., description="Visual Kei subgenre for the band")
    era: Optional[Era] = Field(default=Era.GOLDEN_AGE_1990S, description="Historical era")
    context: Optional[Context] = Field(default=Context.PHOTOSHOOT, description="Visual context")
    member_count: int = Field(default=5, description="Number of band members", ge=2, le=7)
    include_roles: Optional[List[Archetype]] = Field(default=None, description="Specific roles to include")


class CompareSubgenresInput(BaseModel):
    """Input for comparing two subgenres."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    subgenre_a: Subgenre = Field(..., description="First subgenre to compare")
    subgenre_b: Subgenre = Field(..., description="Second subgenre to compare")


# =============================================================================
# HELPER FUNCTIONS - Deterministic operations (Layer 2)
# =============================================================================

def detect_subgenre_from_text(text: str) -> tuple[str, float]:
    """Detect subgenre from natural language text using keyword matching."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    
    for subgenre, keywords in SUBGENRE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[subgenre] = score
    
    if not scores:
        return "kote_kei", 0.5  # Default to classic VK
    
    best_match = max(scores, key=scores.get)
    max_possible = len(SUBGENRE_KEYWORDS.get(best_match, [1]))
    confidence = min(0.5 + (scores[best_match] / max(max_possible, 1)) * 0.5, 0.99)
    
    return best_match, confidence


def detect_era_from_text(text: str) -> tuple[str, float]:
    """Detect era from natural language text."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    
    for era, keywords in ERA_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[era] = score
    
    if not scores:
        return "golden_age_1990s", 0.6
    
    best_match = max(scores, key=scores.get)
    confidence = min(0.6 + (scores[best_match] / 5) * 0.4, 0.99)
    
    return best_match, confidence


def detect_context_from_text(text: str) -> tuple[str, float]:
    """Detect context from natural language text."""
    text_lower = text.lower()
    
    for context, keywords in CONTEXT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return context, 0.9
    
    return "photoshoot", 0.5


def detect_archetype_from_text(text: str) -> tuple[Optional[str], float]:
    """Detect band member archetype from text."""
    text_lower = text.lower()
    
    for archetype, keywords in ARCHETYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return archetype, 0.95
    
    return None, 0.0


def detect_intensity_from_text(text: str) -> str:
    """Detect intensity level from text."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["subtle", "minimal", "casual", "everyday"]):
        return "low"
    elif any(word in text_lower for word in ["moderate", "medium", "balanced"]):
        return "medium"
    elif any(word in text_lower for word in ["maximum", "extreme", "theatrical", "iconic"]):
        return "maximum"
    
    return "high"


def calculate_intensity_percentage(
    base_intensity: str,
    archetype_modifier: float,
    context_modifier: float
) -> int:
    """Calculate final intensity percentage."""
    calc_rules = INTENTIONALITY_OLOG.get("calculation_rules", {}).get("intensity_percentage", {})
    base_values = calc_rules.get("base_values", {"low": 35, "medium": 60, "high": 85, "maximum": 97})
    
    base = base_values.get(base_intensity, 75)
    modified = base * archetype_modifier * context_modifier
    return max(30, min(100, int(modified)))


def build_hair_description(subgenre_data: Dict, era_data: Dict, intensity_modifier: float) -> str:
    """Build hair description from taxonomy."""
    hair = subgenre_data.get("hair", {})
    
    volume = hair.get("volume_inches", "6-8")
    if intensity_modifier < 0.6:
        volume = "3-5"
    
    textures = ", ".join(hair.get("texture", ["styled"]))
    colors = hair.get("colors", ["black"])[0] if hair.get("colors") else "black"
    styles = hair.get("styles", ["dramatic"])[0] if hair.get("styles") else "dramatic"
    
    return f"{volume} inches volume, {textures}, {colors}, {styles}"


def build_makeup_description(subgenre_data: Dict, intensity_modifier: float) -> str:
    """Build makeup description from taxonomy."""
    makeup = subgenre_data.get("makeup", {})
    
    intensity_word = makeup.get("intensity", "medium")
    if intensity_modifier < 0.5:
        intensity_word = "minimal"
    elif intensity_modifier > 0.9:
        intensity_word = "theatrical"
    
    foundation = makeup.get("foundation", "pale")
    eyes = ", ".join(makeup.get("eyes", ["defined"])[:2])
    lips = makeup.get("lips", ["natural"])[0] if makeup.get("lips") else "natural"
    features = makeup.get("features", "")
    
    return f"{intensity_word} intensity, {foundation}, eyes: {eyes}, lips: {lips}, {features}"


def build_garment_description(subgenre_data: Dict, intensity_modifier: float) -> str:
    """Build garment description from taxonomy."""
    garments = subgenre_data.get("garments", {})
    
    types = garments.get("types", ["styled clothing"])[:3]
    fabrics = garments.get("fabrics", ["fabric"])[:2]
    silhouette = garments.get("silhouette", "styled")
    
    num_items = max(1, int(len(types) * intensity_modifier))
    selected_types = types[:num_items]
    
    return f"{', '.join(selected_types)}, fabrics: {', '.join(fabrics)}, silhouette: {silhouette}"


def build_complete_prompt_parameters(
    subgenre: str,
    era: str,
    context: str,
    archetype: Optional[str],
    intensity: str
) -> Dict[str, Any]:
    """
    Build complete prompt parameters from taxonomy (Layer 2).
    This is the core deterministic mapping function.
    """
    subgenre_data = SUBGENRE_TAXONOMY.get(subgenre, SUBGENRE_TAXONOMY.get("kote_kei", {}))
    era_data = ERA_TAXONOMY.get(era, ERA_TAXONOMY.get("golden_age_1990s", {}))
    context_data = CONTEXT_TAXONOMY.get(context, CONTEXT_TAXONOMY.get("photoshoot", {}))
    intensity_data = INTENSITY_MAPPING.get(intensity, INTENSITY_MAPPING.get("high", {}))
    
    archetype_data = None
    archetype_modifier = 1.0
    if archetype:
        archetype_data = ARCHETYPE_TAXONOMY.get(archetype)
        if archetype_data:
            archetype_modifier = archetype_data.get("intensity_modifier", 1.0)
    
    context_modifier = context_data.get("intensity_modifier", 1.0)
    intensity_modifiers = intensity_data.get("modifiers", {"hair": 0.85, "makeup": 0.85, "garment": 0.85})
    intensity_modifier = intensity_modifiers.get("hair", 0.85)
    
    final_intensity = calculate_intensity_percentage(intensity, archetype_modifier, context_modifier)
    
    # Build component descriptions
    hair_desc = build_hair_description(subgenre_data, era_data, intensity_modifier)
    makeup_desc = build_makeup_description(subgenre_data, intensity_modifier)
    garment_desc = build_garment_description(subgenre_data, intensity_modifier)
    
    # Build color palette
    palette = subgenre_data.get("color_palette", {})
    primary_colors = palette.get("primary", ["black"])
    accent_colors = palette.get("accent", ["silver"])
    
    # Build accessories list
    accessories = subgenre_data.get("accessories", [])
    num_accessories = max(1, int(len(accessories) * intensity_modifier))
    selected_accessories = accessories[:num_accessories]
    
    return {
        "subgenre": {
            "key": subgenre,
            "name": subgenre_data.get("name"),
            "mood": subgenre_data.get("mood")
        },
        "era": {
            "key": era,
            "name": era_data.get("name"),
            "production_value": era_data.get("production_value")
        },
        "context": {
            "key": context,
            "name": context_data.get("name"),
            "lighting": context_data.get("lighting"),
            "prompt_additions": context_data.get("prompt_additions", [])
        },
        "archetype": {
            "key": archetype,
            "name": archetype_data.get("name") if archetype_data else None,
            "prompt_additions": archetype_data.get("prompt_additions", []) if archetype_data else []
        } if archetype else None,
        "intensity": {
            "level": intensity,
            "percentage": final_intensity,
            "description": intensity_data.get("description")
        },
        "visual_parameters": {
            "hair": hair_desc,
            "makeup": makeup_desc,
            "garments": garment_desc,
            "color_palette": {
                "primary": primary_colors,
                "accent": accent_colors,
                "scheme": palette.get("scheme", "high_contrast")
            },
            "accessories": selected_accessories
        },
        "exemplar_bands": subgenre_data.get("exemplar_bands", []),
        "synthesis_instruction": (
            f"Synthesize a Visual Kei character prompt using {subgenre_data.get('name')} aesthetic "
            f"from the {era_data.get('name')}. Apply {final_intensity}% intensity. "
            f"Mood should be: {subgenre_data.get('mood')}. "
            f"Context is {context_data.get('name')} with {context_data.get('lighting')} lighting."
        )
    }


# =============================================================================
# MCP TOOLS - Layer 1: Taxonomy Enumeration
# =============================================================================

@mcp.tool(
    name="list_subgenres",
    annotations={
        "title": "List Visual Kei Subgenres",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_subgenres() -> str:
    """
    List all available Visual Kei subgenres with brief descriptions.
    
    LAYER 1: Pure taxonomy enumeration (zero LLM cost).
    
    Returns:
        str: JSON containing all subgenres with names, descriptions, and moods.
    """
    result = []
    for key, data in SUBGENRE_TAXONOMY.items():
        result.append({
            "key": key,
            "name": data.get("name"),
            "description": data.get("description"),
            "mood": data.get("mood"),
            "exemplar_bands": data.get("exemplar_bands", [])[:3]
        })
    
    return json.dumps({"subgenres": result}, indent=2)


@mcp.tool(
    name="list_eras",
    annotations={
        "title": "List Visual Kei Historical Eras",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_eras() -> str:
    """
    List all available Visual Kei historical eras with characteristics.
    
    LAYER 1: Pure taxonomy enumeration (zero LLM cost).
    
    Returns:
        str: JSON containing all eras with names, descriptions, and key bands.
    """
    result = []
    for key, data in ERA_TAXONOMY.items():
        result.append({
            "key": key,
            "name": data.get("name"),
            "description": data.get("description"),
            "characteristics": data.get("characteristics", []),
            "production_value": data.get("production_value"),
            "key_bands": data.get("key_bands", [])
        })
    
    return json.dumps({"eras": result}, indent=2)


@mcp.tool(
    name="list_contexts",
    annotations={
        "title": "List Visual Contexts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_contexts() -> str:
    """
    List all available visual contexts (settings) for character generation.
    
    LAYER 1: Pure taxonomy enumeration (zero LLM cost).
    
    Returns:
        str: JSON containing all contexts with requirements and considerations.
    """
    result = []
    for key, data in CONTEXT_TAXONOMY.items():
        result.append({
            "key": key,
            "name": data.get("name"),
            "intensity_modifier": data.get("intensity_modifier"),
            "lighting": data.get("lighting"),
            "requirements": data.get("requirements", []),
            "considerations": data.get("considerations", [])
        })
    
    return json.dumps({"contexts": result}, indent=2)


@mcp.tool(
    name="list_archetypes",
    annotations={
        "title": "List Band Member Archetypes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_archetypes() -> str:
    """
    List all available band member archetypes with their visual characteristics.
    
    LAYER 1: Pure taxonomy enumeration (zero LLM cost).
    
    Returns:
        str: JSON containing all archetypes with characteristics and modifiers.
    """
    result = []
    for key, data in ARCHETYPE_TAXONOMY.items():
        result.append({
            "key": key,
            "name": data.get("name"),
            "visual_priority": data.get("visual_priority"),
            "intensity_modifier": data.get("intensity_modifier"),
            "characteristics": data.get("characteristics", {})
        })
    
    return json.dumps({"archetypes": result}, indent=2)


@mcp.tool(
    name="get_subgenre_details",
    annotations={
        "title": "Get Subgenre Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_subgenre_details(params: SubgenreInput) -> str:
    """
    Get complete taxonomy details for a specific Visual Kei subgenre.
    
    LAYER 1: Pure taxonomy retrieval (zero LLM cost).
    
    Args:
        params: SubgenreInput containing the subgenre key.
    
    Returns:
        str: JSON with complete subgenre specification.
    """
    subgenre_key = params.subgenre.value
    data = SUBGENRE_TAXONOMY.get(subgenre_key)
    
    if not data:
        return json.dumps({"error": f"Subgenre '{subgenre_key}' not found"})
    
    return json.dumps({"subgenre": subgenre_key, "details": data}, indent=2)


@mcp.tool(
    name="get_era_details",
    annotations={
        "title": "Get Era Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_era_details(params: EraInput) -> str:
    """
    Get complete taxonomy details for a specific Visual Kei era.
    
    LAYER 1: Pure taxonomy retrieval (zero LLM cost).
    
    Args:
        params: EraInput containing the era key.
    
    Returns:
        str: JSON with complete era specification.
    """
    era_key = params.era.value
    data = ERA_TAXONOMY.get(era_key)
    
    if not data:
        return json.dumps({"error": f"Era '{era_key}' not found"})
    
    return json.dumps({"era": era_key, "details": data}, indent=2)


@mcp.tool(
    name="get_intensity_guide",
    annotations={
        "title": "Get Intensity Level Guide",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_intensity_guide() -> str:
    """
    Get complete guide to intensity levels and their applications.
    
    LAYER 1: Pure taxonomy reference (zero LLM cost).
    
    Returns:
        str: JSON with all intensity levels, percentages, and use cases.
    """
    return json.dumps({
        "intensity_levels": INTENSITY_MAPPING,
        "usage_guide": {
            "low": "Use for everyday wear, subtle VK influence, fan fashion",
            "medium": "Use for concerts as audience, themed events, casual shoots",
            "high": "Use for professional photos, music videos, performances",
            "maximum": "Use for album covers, major events, iconic looks"
        }
    }, indent=2)


# =============================================================================
# MCP TOOLS - Layer 2: Parameter Mapping
# =============================================================================

@mcp.tool(
    name="analyze_intent",
    annotations={
        "title": "Analyze User Intent",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def analyze_intent(params: IntentAnalysisInput) -> str:
    """
    Analyze natural language prompt to detect Visual Kei parameters.
    
    LAYER 2: Deterministic keyword matching (zero LLM cost).
    
    Args:
        params: IntentAnalysisInput containing the user's natural language prompt.
    
    Returns:
        str: JSON with detected subgenre, era, context, archetype, and confidence scores.
    """
    text = params.prompt
    
    subgenre, subgenre_conf = detect_subgenre_from_text(text)
    era, era_conf = detect_era_from_text(text)
    context, context_conf = detect_context_from_text(text)
    archetype, archetype_conf = detect_archetype_from_text(text)
    intensity = detect_intensity_from_text(text)
    
    return json.dumps({
        "original_prompt": text,
        "detected_parameters": {
            "subgenre": {"value": subgenre, "confidence": round(subgenre_conf, 2)},
            "era": {"value": era, "confidence": round(era_conf, 2)},
            "context": {"value": context, "confidence": round(context_conf, 2)},
            "archetype": {"value": archetype, "confidence": round(archetype_conf, 2)} if archetype else None,
            "intensity": intensity
        },
        "overall_confidence": round((subgenre_conf + era_conf + context_conf) / 3, 2)
    }, indent=2)


@mcp.tool(
    name="map_parameters",
    annotations={
        "title": "Map Visual Kei Parameters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def map_parameters(params: ParameterMappingInput) -> str:
    """
    Map Visual Kei selections to complete visual parameters.
    
    LAYER 2: Deterministic parameter assembly (zero LLM cost).
    
    Args:
        params: ParameterMappingInput with subgenre, era, context, archetype, intensity.
    
    Returns:
        str: JSON with complete visual parameters for synthesis.
    """
    result = build_complete_prompt_parameters(
        subgenre=params.subgenre.value,
        era=params.era.value if params.era else "golden_age_1990s",
        context=params.context.value if params.context else "photoshoot",
        archetype=params.archetype.value if params.archetype else None,
        intensity=params.intensity.value if params.intensity else "high"
    )
    
    return json.dumps(result, indent=2)


@mcp.tool(
    name="compare_subgenres",
    annotations={
        "title": "Compare Two Subgenres",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def compare_subgenres(params: CompareSubgenresInput) -> str:
    """
    Compare two Visual Kei subgenres side by side.
    
    LAYER 1: Pure taxonomy comparison (zero LLM cost).
    
    Args:
        params: CompareSubgenresInput with two subgenres to compare.
    
    Returns:
        str: JSON with side-by-side comparison.
    """
    a_key = params.subgenre_a.value
    b_key = params.subgenre_b.value
    
    a_data = SUBGENRE_TAXONOMY.get(a_key, {})
    b_data = SUBGENRE_TAXONOMY.get(b_key, {})
    
    comparison = {
        "subgenre_a": {
            "key": a_key,
            "name": a_data.get("name"),
            "mood": a_data.get("mood"),
            "color_scheme": a_data.get("color_palette", {}).get("scheme"),
            "primary_colors": a_data.get("color_palette", {}).get("primary", []),
            "makeup_intensity": a_data.get("makeup", {}).get("intensity"),
            "hair_volume": a_data.get("hair", {}).get("volume"),
            "exemplar_bands": a_data.get("exemplar_bands", [])[:3]
        },
        "subgenre_b": {
            "key": b_key,
            "name": b_data.get("name"),
            "mood": b_data.get("mood"),
            "color_scheme": b_data.get("color_palette", {}).get("scheme"),
            "primary_colors": b_data.get("color_palette", {}).get("primary", []),
            "makeup_intensity": b_data.get("makeup", {}).get("intensity"),
            "hair_volume": b_data.get("hair", {}).get("volume"),
            "exemplar_bands": b_data.get("exemplar_bands", [])[:3]
        },
        "key_differences": []
    }
    
    # Identify key differences
    if a_data.get("mood") != b_data.get("mood"):
        comparison["key_differences"].append(
            f"Mood: {a_data.get('name')} is '{a_data.get('mood')}' vs {b_data.get('name')} is '{b_data.get('mood')}'"
        )
    
    a_scheme = a_data.get("color_palette", {}).get("scheme", "")
    b_scheme = b_data.get("color_palette", {}).get("scheme", "")
    if a_scheme != b_scheme:
        comparison["key_differences"].append(f"Color scheme: {a_scheme} vs {b_scheme}")
    
    return json.dumps(comparison, indent=2)


# =============================================================================
# MCP TOOLS - Layer 3: Enhancement Interface
# =============================================================================

@mcp.tool(
    name="enhance_with_visual_kei",
    annotations={
        "title": "Enhance Prompt with Visual Kei Aesthetic",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def enhance_with_visual_kei(params: EnhancePromptInput) -> str:
    """
    Complete enhancement workflow: analyze intent + map parameters.
    
    LAYER 3 INTERFACE: Combines Layers 1 & 2 into structured data
    ready for Claude to synthesize into an enhanced prompt.
    
    Args:
        params: EnhancePromptInput with base_prompt and optional parameter overrides.
    
    Returns:
        str: JSON with complete visual vocabulary and synthesis instructions.
    """
    text = params.base_prompt
    
    # Use provided values or detect from text
    if params.subgenre:
        subgenre = params.subgenre.value
        subgenre_conf = 1.0
    else:
        subgenre, subgenre_conf = detect_subgenre_from_text(text)
    
    era = params.era.value if params.era else detect_era_from_text(text)[0]
    context = params.context.value if params.context else detect_context_from_text(text)[0]
    archetype = params.archetype.value if params.archetype else detect_archetype_from_text(text)[0]
    intensity = params.intensity.value if params.intensity else "high"
    
    visual_params = build_complete_prompt_parameters(
        subgenre=subgenre,
        era=era,
        context=context,
        archetype=archetype,
        intensity=intensity
    )
    
    # Get synthesis guidance from intentionality olog
    synthesis_guidance = INTENTIONALITY_OLOG.get("intentionality", {}).get("synthesis_guidance", {})
    
    return json.dumps({
        "original_prompt": text,
        "detected_confidence": round(subgenre_conf, 2),
        "parameters_used": {
            "subgenre": subgenre,
            "era": era,
            "context": context,
            "archetype": archetype,
            "intensity": intensity
        },
        "visual_vocabulary": visual_params,
        "synthesis_guidance": {
            "instruction": synthesis_guidance.get("general_approach", 
                "Synthesize Visual Kei aesthetic parameters into a cohesive image generation prompt."),
            "priority_elements": synthesis_guidance.get("priority_elements", []),
            "avoid": synthesis_guidance.get("avoid", [])
        }
    }, indent=2)


@mcp.tool(
    name="generate_band_parameters",
    annotations={
        "title": "Generate Complete Band Parameters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def generate_band_parameters(params: BandGenerationInput) -> str:
    """
    Generate parameters for a complete Visual Kei band with multiple members.
    
    LAYER 2: Deterministic multi-character assembly (zero LLM cost).
    
    Args:
        params: BandGenerationInput with subgenre, era, context, member count.
    
    Returns:
        str: JSON with band overview and individual member specifications.
    """
    subgenre = params.subgenre.value
    era = params.era.value if params.era else "golden_age_1990s"
    context = params.context.value if params.context else "photoshoot"
    
    # Get band configurations from olog
    band_configs = ARCHETYPES_OLOG.get("band_configurations", {})
    config_keys = {2: "duo", 3: "trio", 4: "quartet", 5: "quintet", 6: "sextet", 7: "sextet"}
    config = band_configs.get(config_keys.get(params.member_count, "quintet"), {})
    
    default_roles = config.get("members", ["vocalist", "guitarist", "guitarist", "bassist", "drummer"])
    roles = [Archetype(r) if isinstance(r, str) else r for r in (params.include_roles or default_roles)]
    
    subgenre_data = SUBGENRE_TAXONOMY.get(subgenre, {})
    era_data = ERA_TAXONOMY.get(era, {})
    
    band_overview = {
        "subgenre": subgenre_data.get("name"),
        "era": era_data.get("name"),
        "shared_color_palette": subgenre_data.get("color_palette"),
        "shared_mood": subgenre_data.get("mood"),
        "cohesion_notes": "All members share the same color palette and subgenre characteristics."
    }
    
    members = []
    for i, role in enumerate(roles[:params.member_count]):
        role_value = role.value if isinstance(role, Archetype) else role
        member_params = build_complete_prompt_parameters(
            subgenre=subgenre,
            era=era,
            context=context,
            archetype=role_value,
            intensity="high"
        )
        
        members.append({
            "member_number": i + 1,
            "role": role_value,
            "intensity_percentage": member_params["intensity"]["percentage"],
            "visual_parameters": member_params["visual_parameters"],
            "archetype_notes": member_params.get("archetype", {})
        })
    
    return json.dumps({
        "band_overview": band_overview,
        "member_count": params.member_count,
        "members": members,
        "synthesis_instruction": (
            f"Generate a cohesive {params.member_count}-member {subgenre_data.get('name')} band "
            f"from the {era_data.get('name')}. Each member should be individually distinct "
            f"while maintaining visual cohesion through shared color palette and mood."
        )
    }, indent=2)


# =============================================================================
# PHASE 2.6: RHYTHMIC COMPOSITION — 5D Normalized Morphospace
# =============================================================================
#
# Visual Kei aesthetic space is parameterized by 5 normalized dimensions [0,1]:
#
#   theatrical_intensity  — subtle everyday (0.0) to maximum stage theatrics (1.0)
#   darkness_quotient     — bright/colorful oshare (0.0) to darkest angura/eroguro (1.0)
#   androgyny_level       — conventional presentation (0.0) to maximally androgynous (1.0)
#   ornamentation_density — stripped minimal (0.0) to maximum decoration (1.0)
#   era_modernity         — raw pioneer 1980s (0.0) to polished neo-revival (1.0)
#
# Canonical states anchored at the 7 subgenres. Rhythmic presets oscillate
# between pairs of these canonical states via forced orbit integration.
#
# Period strategy:
#   14 — fills gap 12–15 (novel)
#   11 — fills gap 10–12 (novel)
#   26 — fills gap 25–30 (novel)
#   32 — above-30 novel harmonic region
#   22 — shared with catastrophe + heraldic (synchronization)
#   18 — shared with nuclear + catastrophe + diatom (synchronization)
# =============================================================================

VISUAL_KEI_PARAMETER_NAMES = [
    "theatrical_intensity",
    "darkness_quotient",
    "androgyny_level",
    "ornamentation_density",
    "era_modernity",
]

VISUAL_KEI_CANONICAL_STATES = {
    "kote_kei": {
        "theatrical_intensity": 0.85,
        "darkness_quotient": 0.70,
        "androgyny_level": 0.90,
        "ornamentation_density": 0.80,
        "era_modernity": 0.45,
    },
    "oshare_kei": {
        "theatrical_intensity": 0.60,
        "darkness_quotient": 0.10,
        "androgyny_level": 0.75,
        "ornamentation_density": 0.85,
        "era_modernity": 0.80,
    },
    "nagoya_kei": {
        "theatrical_intensity": 0.70,
        "darkness_quotient": 0.80,
        "androgyny_level": 0.55,
        "ornamentation_density": 0.30,
        "era_modernity": 0.35,
    },
    "angura_kei": {
        "theatrical_intensity": 0.95,
        "darkness_quotient": 0.90,
        "androgyny_level": 0.85,
        "ornamentation_density": 0.60,
        "era_modernity": 0.25,
    },
    "eroguro_kei": {
        "theatrical_intensity": 1.00,
        "darkness_quotient": 0.95,
        "androgyny_level": 0.80,
        "ornamentation_density": 0.70,
        "era_modernity": 0.50,
    },
    "lolita_kei": {
        "theatrical_intensity": 0.75,
        "darkness_quotient": 0.40,
        "androgyny_level": 0.95,
        "ornamentation_density": 1.00,
        "era_modernity": 0.60,
    },
    "iryou_kei": {
        "theatrical_intensity": 0.80,
        "darkness_quotient": 0.65,
        "androgyny_level": 0.70,
        "ornamentation_density": 0.45,
        "era_modernity": 0.90,
    },
}

VISUAL_KEI_RHYTHMIC_PRESETS = {
    "subgenre_morph": {
        "state_a": "kote_kei",
        "state_b": "oshare_kei",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 14,
        "description": "Gothic aristocrat ↔ neon pop morph (period 14, gap-filler 12–15)",
    },
    "darkness_oscillation": {
        "state_a": "oshare_kei",
        "state_b": "angura_kei",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "description": "Bright pop ↔ underground darkness (period 22, syncs catastrophe+heraldic)",
    },
    "era_drift": {
        "state_a": "nagoya_kei",
        "state_b": "iryou_kei",
        "pattern": "triangular",
        "num_cycles": 3,
        "steps_per_cycle": 18,
        "description": "Raw retro aggression ↔ clinical modernity (period 18, syncs nuclear+diatom)",
    },
    "theatrical_crescendo": {
        "state_a": "lolita_kei",
        "state_b": "eroguro_kei",
        "pattern": "sinusoidal",
        "num_cycles": 2,
        "steps_per_cycle": 26,
        "description": "Porcelain elegance ↔ grotesque excess (period 26, gap-filler 25–30)",
    },
    "androgyny_pulse": {
        "state_a": "nagoya_kei",
        "state_b": "lolita_kei",
        "pattern": "square",
        "num_cycles": 5,
        "steps_per_cycle": 11,
        "description": "Masculine raw ↔ feminine ornate binary toggle (period 11, gap-filler 10–12)",
    },
    "clinical_bloom": {
        "state_a": "iryou_kei",
        "state_b": "oshare_kei",
        "pattern": "sinusoidal",
        "num_cycles": 2,
        "steps_per_cycle": 32,
        "description": "Sterile clinical ↔ exuberant color bloom (period 32, novel above-30)",
    },
}


# =============================================================================
# PHASE 2.7: VISUAL VOCABULARY — Attractor Visualization Prompt Generation
# =============================================================================
#
# 7 visual types anchored at subgenre canonical states. Each carries
# image-generation-ready keywords. Nearest-neighbor matching (Euclidean
# distance, weight cutoff ~0.15) maps any 5D coordinate to keywords.
# =============================================================================

VISUAL_KEI_VISUAL_TYPES = {
    "gothic_elegance": {
        "coords": {
            "theatrical_intensity": 0.85,
            "darkness_quotient": 0.70,
            "androgyny_level": 0.90,
            "ornamentation_density": 0.80,
            "era_modernity": 0.45,
        },
        "keywords": [
            "gothic aristocrat Visual Kei",
            "elaborate dark lace and velvet",
            "silver cross jewelry and chains",
            "dramatic smoky eye with white foundation",
            "towering teased black hair with colored streaks",
            "Victorian ruffled blouse under corseted vest",
            "theatrical androgynous beauty",
        ],
    },
    "neon_kawaii": {
        "coords": {
            "theatrical_intensity": 0.60,
            "darkness_quotient": 0.10,
            "androgyny_level": 0.75,
            "ornamentation_density": 0.85,
            "era_modernity": 0.80,
        },
        "keywords": [
            "oshare kei neon pop aesthetic",
            "candy-bright layered clothing",
            "rainbow-dyed voluminous hair",
            "glitter eye makeup with star decals",
            "plastic jewelry and novelty accessories",
            "plaid layered over neon mesh",
            "cheerful kawaii Visual Kei",
        ],
    },
    "underground_raw": {
        "coords": {
            "theatrical_intensity": 0.70,
            "darkness_quotient": 0.80,
            "androgyny_level": 0.55,
            "ornamentation_density": 0.30,
            "era_modernity": 0.35,
        },
        "keywords": [
            "nagoya kei raw aggressive aesthetic",
            "stripped-back dark clothing",
            "smeared kohl eyeliner on pale skin",
            "tangled unwashed-look hair",
            "torn fabric and safety-pin detailing",
            "industrial metal accessories",
            "gritty lo-fi visual noise",
        ],
    },
    "ancestral_theatre": {
        "coords": {
            "theatrical_intensity": 0.95,
            "darkness_quotient": 0.90,
            "androgyny_level": 0.85,
            "ornamentation_density": 0.60,
            "era_modernity": 0.25,
        },
        "keywords": [
            "angura kei avant-garde kabuki fusion",
            "traditional Japanese theatrical makeup",
            "kimono elements deconstructed into punk silhouette",
            "bone-white face paint with crimson accents",
            "ritual ceremonial staging",
            "butoh-influenced body posture",
            "dark shamanic presence",
        ],
    },
    "grotesque_beauty": {
        "coords": {
            "theatrical_intensity": 1.00,
            "darkness_quotient": 0.95,
            "androgyny_level": 0.80,
            "ornamentation_density": 0.70,
            "era_modernity": 0.50,
        },
        "keywords": [
            "eroguro kei grotesque glamour",
            "horror-inspired prosthetic details",
            "blood-red contact lenses and veined makeup",
            "bandage wrapping over couture garments",
            "decayed elegance and ruined beauty",
            "surgical aesthetic mixed with haute couture",
            "visceral theatrical shock beauty",
        ],
    },
    "porcelain_doll": {
        "coords": {
            "theatrical_intensity": 0.75,
            "darkness_quotient": 0.40,
            "androgyny_level": 0.95,
            "ornamentation_density": 1.00,
            "era_modernity": 0.60,
        },
        "keywords": [
            "lolita kei porcelain doll aesthetic",
            "elaborate lace headdress and ribbon cascades",
            "voluminous petticoat silhouette",
            "delicate rosebud lip and doll-eye circle lenses",
            "pastel and black layered textiles",
            "miniature crown and cameo brooch details",
            "ornate bisque-figure precision",
        ],
    },
    "clinical_avant_garde": {
        "coords": {
            "theatrical_intensity": 0.80,
            "darkness_quotient": 0.65,
            "androgyny_level": 0.70,
            "ornamentation_density": 0.45,
            "era_modernity": 0.90,
        },
        "keywords": [
            "iryou kei clinical avant-garde",
            "sterile white and surgical green palette",
            "medical bandage and IV-tube accessories",
            "asymmetric precision-cut hair",
            "contact lenses with clinical iris patterns",
            "PVC and latex fabric panels",
            "futuristic antiseptic beauty",
        ],
    },
}


# =============================================================================
# PHASE 2.6 CORE — Deterministic Oscillation & Trajectory Generation (Layer 2)
# =============================================================================

def _vk_generate_oscillation(num_steps: int, num_cycles: float, pattern: str) -> np.ndarray:
    """Generate oscillation envelope in [0, 1]."""
    t = np.linspace(0, 2 * np.pi * num_cycles, num_steps, endpoint=False)
    if pattern == "sinusoidal":
        return 0.5 * (1.0 + np.sin(t))
    elif pattern == "triangular":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 2.0 * t_norm, 2.0 * (1.0 - t_norm))
    elif pattern == "square":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown oscillation pattern: {pattern}")


def _vk_generate_preset_trajectory(preset_config: dict) -> np.ndarray:
    """
    Generate a single preset trajectory via forced orbit integration.

    Returns ndarray of shape (total_steps, 5) — one full cycle set,
    guaranteed periodic by construction (zero drift).
    """
    state_a = VISUAL_KEI_CANONICAL_STATES[preset_config["state_a"]]
    state_b = VISUAL_KEI_CANONICAL_STATES[preset_config["state_b"]]

    num_cycles = preset_config["num_cycles"]
    steps_per_cycle = preset_config["steps_per_cycle"]
    total_steps = num_cycles * steps_per_cycle

    alpha = _vk_generate_oscillation(total_steps, num_cycles, preset_config["pattern"])

    vec_a = np.array([state_a[p] for p in VISUAL_KEI_PARAMETER_NAMES])
    vec_b = np.array([state_b[p] for p in VISUAL_KEI_PARAMETER_NAMES])

    trajectory = np.outer(1.0 - alpha, vec_a) + np.outer(alpha, vec_b)
    return trajectory


def _vk_extract_visual_vocabulary(state: dict, strength: float = 1.0) -> dict:
    """
    Nearest-neighbor vocabulary extraction from 5D coordinate.

    Returns the closest visual type, its Euclidean distance, and
    weighted keywords suitable for image generation prompts.
    """
    point = np.array([state.get(p, 0.5) for p in VISUAL_KEI_PARAMETER_NAMES])

    best_type = None
    best_dist = float("inf")
    best_keywords = []

    for vtype, vdata in VISUAL_KEI_VISUAL_TYPES.items():
        ref = np.array([vdata["coords"][p] for p in VISUAL_KEI_PARAMETER_NAMES])
        dist = float(np.linalg.norm(point - ref))
        if dist < best_dist:
            best_dist = dist
            best_type = vtype
            best_keywords = vdata["keywords"]

    return {
        "nearest_type": best_type,
        "distance": round(best_dist, 4),
        "strength": round(strength, 3),
        "keywords": best_keywords,
    }


# =============================================================================
# PHASE 2.6 MCP TOOLS — Rhythmic Composition
# =============================================================================

@mcp.tool(
    name="get_visual_kei_coordinates",
    annotations={
        "title": "Get Visual Kei 5D Coordinates",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_visual_kei_coordinates(state_name: str) -> str:
    """
    Return normalized 5D morphospace coordinates for a canonical Visual Kei state.

    LAYER 2: Pure deterministic lookup (zero LLM cost).

    Valid state names: kote_kei, oshare_kei, nagoya_kei, angura_kei,
    eroguro_kei, lolita_kei, iryou_kei.
    """
    coords = VISUAL_KEI_CANONICAL_STATES.get(state_name)
    if coords is None:
        return json.dumps({
            "error": f"Unknown state '{state_name}'",
            "valid_states": list(VISUAL_KEI_CANONICAL_STATES.keys()),
        })
    return json.dumps({
        "state": state_name,
        "coordinates": coords,
        "parameter_names": VISUAL_KEI_PARAMETER_NAMES,
    }, indent=2)


@mcp.tool(
    name="list_visual_kei_rhythmic_presets",
    annotations={
        "title": "List Visual Kei Rhythmic Presets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_visual_kei_rhythmic_presets() -> str:
    """
    List all Phase 2.6 rhythmic presets with their periods and descriptions.

    LAYER 1: Pure enumeration (zero LLM cost).

    Each preset defines a forced-orbit oscillation between two canonical
    Visual Kei subgenre states. Periods chosen strategically for
    cross-domain LCM resonance and gap-filling.
    """
    presets = []
    for name, cfg in VISUAL_KEI_RHYTHMIC_PRESETS.items():
        presets.append({
            "name": name,
            "period": cfg["steps_per_cycle"],
            "pattern": cfg["pattern"],
            "num_cycles": cfg["num_cycles"],
            "total_steps": cfg["num_cycles"] * cfg["steps_per_cycle"],
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "description": cfg["description"],
        })
    return json.dumps({
        "domain": "visual_kei",
        "phase": "2.6",
        "preset_count": len(presets),
        "periods": sorted(set(cfg["steps_per_cycle"] for cfg in VISUAL_KEI_RHYTHMIC_PRESETS.values())),
        "presets": presets,
    }, indent=2)


@mcp.tool(
    name="apply_visual_kei_rhythmic_preset",
    annotations={
        "title": "Apply Visual Kei Rhythmic Preset",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def apply_visual_kei_rhythmic_preset(preset_name: str) -> str:
    """
    Generate a complete rhythmic oscillation trajectory for a preset.

    LAYER 2: Deterministic forced orbit integration (zero LLM cost).

    Returns the full trajectory as a list of 5D parameter states,
    guaranteed periodic by construction.
    """
    cfg = VISUAL_KEI_RHYTHMIC_PRESETS.get(preset_name)
    if cfg is None:
        return json.dumps({
            "error": f"Unknown preset '{preset_name}'",
            "valid_presets": list(VISUAL_KEI_RHYTHMIC_PRESETS.keys()),
        })

    trajectory = _vk_generate_preset_trajectory(cfg)
    total_steps = trajectory.shape[0]

    # Convert to list of dicts
    states = []
    for i in range(total_steps):
        state = {p: round(float(trajectory[i, j]), 4) for j, p in enumerate(VISUAL_KEI_PARAMETER_NAMES)}
        states.append(state)

    return json.dumps({
        "preset": preset_name,
        "period": cfg["steps_per_cycle"],
        "pattern": cfg["pattern"],
        "num_cycles": cfg["num_cycles"],
        "total_steps": total_steps,
        "state_a": cfg["state_a"],
        "state_b": cfg["state_b"],
        "trajectory": states,
    }, indent=2)


@mcp.tool(
    name="compute_visual_kei_trajectory",
    annotations={
        "title": "Compute Trajectory Between Visual Kei States",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def compute_visual_kei_trajectory(
    state_a: str,
    state_b: str,
    steps: int = 24,
    pattern: str = "sinusoidal",
    num_cycles: int = 1,
) -> str:
    """
    Compute a smooth interpolation trajectory between two canonical states.

    LAYER 2: Deterministic forced orbit (zero LLM cost).

    Args:
        state_a: Starting canonical state name.
        state_b: Ending canonical state name.
        steps: Steps per cycle (period). Default 24.
        pattern: Oscillation pattern — sinusoidal, triangular, or square.
        num_cycles: Number of full oscillation cycles. Default 1.
    """
    if state_a not in VISUAL_KEI_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown state_a '{state_a}'", "valid": list(VISUAL_KEI_CANONICAL_STATES.keys())})
    if state_b not in VISUAL_KEI_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown state_b '{state_b}'", "valid": list(VISUAL_KEI_CANONICAL_STATES.keys())})
    if pattern not in ("sinusoidal", "triangular", "square"):
        return json.dumps({"error": f"Unknown pattern '{pattern}'", "valid": ["sinusoidal", "triangular", "square"]})

    cfg = {
        "state_a": state_a,
        "state_b": state_b,
        "pattern": pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps,
    }
    trajectory = _vk_generate_preset_trajectory(cfg)
    total = trajectory.shape[0]

    states = []
    for i in range(total):
        state = {p: round(float(trajectory[i, j]), 4) for j, p in enumerate(VISUAL_KEI_PARAMETER_NAMES)}
        states.append(state)

    return json.dumps({
        "state_a": state_a,
        "state_b": state_b,
        "period": steps,
        "pattern": pattern,
        "num_cycles": num_cycles,
        "total_steps": total,
        "trajectory": states,
    }, indent=2)


# =============================================================================
# PHASE 2.7 MCP TOOLS — Attractor Visualization Prompt Generation
# =============================================================================

@mcp.tool(
    name="get_visual_kei_visual_types",
    annotations={
        "title": "List Visual Kei Visual Types",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_visual_kei_visual_types() -> str:
    """
    List all visual vocabulary types with their 5D coordinates and keywords.

    LAYER 1: Pure enumeration (zero LLM cost).

    Each visual type anchors a region of the Visual Kei morphospace and
    carries image-generation-ready keywords for prompt composition.
    """
    types = []
    for vtype, vdata in VISUAL_KEI_VISUAL_TYPES.items():
        types.append({
            "type": vtype,
            "coordinates": vdata["coords"],
            "keywords": vdata["keywords"],
        })
    return json.dumps({
        "domain": "visual_kei",
        "phase": "2.7",
        "visual_type_count": len(types),
        "visual_types": types,
    }, indent=2)


@mcp.tool(
    name="generate_visual_kei_attractor_prompt",
    annotations={
        "title": "Generate Attractor Visualization Prompt",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def generate_visual_kei_attractor_prompt(
    theatrical_intensity: float = 0.5,
    darkness_quotient: float = 0.5,
    androgyny_level: float = 0.5,
    ornamentation_density: float = 0.5,
    era_modernity: float = 0.5,
    mode: str = "composite",
) -> str:
    """
    Generate image-generation-ready prompt from a 5D morphospace coordinate.

    LAYER 2: Deterministic nearest-neighbor vocabulary extraction (zero LLM cost).

    Three generation modes:
      composite  — Blended prompt from nearest visual type keywords.
      split_view — Separate keyword sets for primary + secondary types.
      sequence   — Trajectory of keywords across parameter interpolation.

    Args:
        theatrical_intensity: 0.0 (subtle) to 1.0 (maximum theatrics).
        darkness_quotient:    0.0 (bright) to 1.0 (darkest).
        androgyny_level:      0.0 (conventional) to 1.0 (maximally androgynous).
        ornamentation_density: 0.0 (minimal) to 1.0 (maximum decoration).
        era_modernity:        0.0 (pioneer retro) to 1.0 (polished neo-revival).
        mode: Generation mode — composite, split_view, or sequence.
    """
    state = {
        "theatrical_intensity": max(0.0, min(1.0, theatrical_intensity)),
        "darkness_quotient": max(0.0, min(1.0, darkness_quotient)),
        "androgyny_level": max(0.0, min(1.0, androgyny_level)),
        "ornamentation_density": max(0.0, min(1.0, ornamentation_density)),
        "era_modernity": max(0.0, min(1.0, era_modernity)),
    }

    point = np.array([state[p] for p in VISUAL_KEI_PARAMETER_NAMES])

    # Compute distances to all visual types
    ranked = []
    for vtype, vdata in VISUAL_KEI_VISUAL_TYPES.items():
        ref = np.array([vdata["coords"][p] for p in VISUAL_KEI_PARAMETER_NAMES])
        dist = float(np.linalg.norm(point - ref))
        ranked.append((vtype, dist, vdata["keywords"]))
    ranked.sort(key=lambda x: x[1])

    if mode == "composite":
        primary = ranked[0]
        secondary = ranked[1] if len(ranked) > 1 else None

        # Weight: primary gets full keywords, secondary filtered by proximity
        prompt_keywords = list(primary[2])
        if secondary and secondary[1] < 0.45:
            # Blend in secondary keywords (skip duplicates)
            blend_count = max(1, int(len(secondary[2]) * (1.0 - secondary[1] / 0.45)))
            for kw in secondary[2][:blend_count]:
                if kw not in prompt_keywords:
                    prompt_keywords.append(kw)

        return json.dumps({
            "mode": "composite",
            "input_state": state,
            "primary_type": primary[0],
            "primary_distance": round(primary[1], 4),
            "secondary_type": secondary[0] if secondary else None,
            "secondary_distance": round(secondary[1], 4) if secondary else None,
            "prompt_keywords": prompt_keywords,
            "composite_prompt": ", ".join(prompt_keywords),
        }, indent=2)

    elif mode == "split_view":
        primary = ranked[0]
        secondary = ranked[1] if len(ranked) > 1 else None

        result = {
            "mode": "split_view",
            "input_state": state,
            "primary": {
                "type": primary[0],
                "distance": round(primary[1], 4),
                "keywords": primary[2],
                "prompt": ", ".join(primary[2]),
            },
        }
        if secondary:
            result["secondary"] = {
                "type": secondary[0],
                "distance": round(secondary[1], 4),
                "keywords": secondary[2],
                "prompt": ", ".join(secondary[2]),
            }
        return json.dumps(result, indent=2)

    elif mode == "sequence":
        # Generate 5-keyframe sequence interpolating through morphospace
        keyframes = []
        for i in range(5):
            t = i / 4.0
            interp_state = {}
            center = np.array([0.5] * 5)
            interp_point = center + t * (point - center)
            for j, p in enumerate(VISUAL_KEI_PARAMETER_NAMES):
                interp_state[p] = round(float(np.clip(interp_point[j], 0.0, 1.0)), 4)

            vocab = _vk_extract_visual_vocabulary(interp_state)
            keyframes.append({
                "keyframe": i + 1,
                "t": round(t, 2),
                "state": interp_state,
                "nearest_type": vocab["nearest_type"],
                "distance": vocab["distance"],
                "keywords": vocab["keywords"],
            })

        return json.dumps({
            "mode": "sequence",
            "input_state": state,
            "keyframe_count": 5,
            "keyframes": keyframes,
        }, indent=2)

    else:
        return json.dumps({
            "error": f"Unknown mode '{mode}'",
            "valid_modes": ["composite", "split_view", "sequence"],
        })


# =============================================================================
# DOMAIN REGISTRY CONFIG — Tier 4D Integration
# =============================================================================

@mcp.tool(
    name="get_visual_kei_domain_registry_config",
    annotations={
        "title": "Get Domain Registry Config for Tier 4D",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_visual_kei_domain_registry_config() -> str:
    """
    Return domain registration config for the Tier 4D compositional system.

    LAYER 2: Deterministic export (zero LLM cost).

    Provides everything needed to register visual_kei in domain_registry.py:
    parameter names, canonical state coordinates, preset configs, visual
    vocabulary, and predicted emergent attractor interactions.
    """
    preset_configs = {}
    for name, cfg in VISUAL_KEI_RHYTHMIC_PRESETS.items():
        preset_configs[name] = {
            "period": cfg["steps_per_cycle"],
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "pattern": cfg["pattern"],
            "description": cfg["description"],
        }

    periods = sorted(set(cfg["steps_per_cycle"] for cfg in VISUAL_KEI_RHYTHMIC_PRESETS.values()))

    # Predicted emergent attractors based on period interactions
    predicted_attractors = [
        {
            "mechanism": "gap_filler",
            "predicted_period": 13,
            "gap": "12–14",
            "confidence": "medium",
            "note": "Period 14 (subgenre_morph) and period 12 (diatom/heraldic) bracket this gap",
        },
        {
            "mechanism": "lcm_sync",
            "predicted_period": 22,
            "partners": ["catastrophe (22)", "heraldic (22)"],
            "confidence": "high",
            "note": "Three-domain period lock at 22 steps",
        },
        {
            "mechanism": "lcm_sync",
            "predicted_period": 18,
            "partners": ["nuclear (18)", "catastrophe (18)", "diatom (18)"],
            "confidence": "high",
            "note": "Four-domain period lock at 18 steps",
        },
        {
            "mechanism": "composite_beat",
            "predicted_period": 40,
            "formula": "LCM(8, 10) from half-periods of 16 and 20",
            "confidence": "low",
            "note": "Speculative interaction with microscopy periods",
        },
        {
            "mechanism": "gap_filler",
            "predicted_period": 29,
            "gap": "26–32",
            "confidence": "medium",
            "note": "Period 26 (theatrical_crescendo) and 32 (clinical_bloom) create large gap",
        },
    ]

    return json.dumps({
        "domain_id": "visual_kei",
        "display_name": "Visual Kei Aesthetic",
        "mcp_server": "visual-kei-mcp",
        "phase_2_6_status": "complete",
        "phase_2_7_status": "complete",
        "parameter_names": VISUAL_KEI_PARAMETER_NAMES,
        "canonical_states": VISUAL_KEI_CANONICAL_STATES,
        "preset_count": len(preset_configs),
        "presets": preset_configs,
        "periods": periods,
        "visual_type_count": len(VISUAL_KEI_VISUAL_TYPES),
        "visual_types": {
            vtype: {
                "coords": vdata["coords"],
                "keywords": vdata["keywords"],
            }
            for vtype, vdata in VISUAL_KEI_VISUAL_TYPES.items()
        },
        "predicted_emergent_attractors": predicted_attractors,
        "tier_4d_integration": {
            "estimated_basin_sizes": {
                "period_22_sync": 0.08,
                "period_18_sync": 0.06,
                "period_14_gap": 0.04,
                "period_26_gap": 0.03,
            },
            "recommended_combinations": [
                "visual_kei × catastrophe (shared period 22)",
                "visual_kei × nuclear (shared period 18)",
                "visual_kei × diatom (shared period 18)",
                "visual_kei × heraldic (shared period 22)",
                "visual_kei × microscopy (novel interactions via periods 11, 14, 26, 32)",
            ],
        },
    }, indent=2)


# =============================================================================
# SERVER INFO UPDATE — Reflect Phase 2.6 + 2.7 Status
# =============================================================================

@mcp.tool(
    name="get_server_info",
    annotations={
        "title": "Get Visual Kei MCP Server Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_server_info() -> str:
    """
    Return server metadata, capabilities, and Phase 2.6/2.7 status.

    LAYER 1: Pure metadata (zero LLM cost).
    """
    return json.dumps({
        "server": "visual-kei-mcp",
        "version": "2.7.0",
        "description": (
            "Visual Kei aesthetic vocabulary server with Phase 2.6 rhythmic "
            "composition and Phase 2.7 attractor visualization prompt generation."
        ),
        "three_layer_architecture": {
            "layer_1": "Deterministic taxonomy lookup from YAML ologs (zero LLM cost)",
            "layer_2": "Structured parameter assembly + Phase 2.6/2.7 dynamics (zero LLM cost)",
            "layer_3": "LLM synthesis interface (returns data for Claude)",
        },
        "subgenres": list(VISUAL_KEI_CANONICAL_STATES.keys()),
        "eras": ["pioneer_1980s", "golden_age_1990s", "diversification_2000s", "neo_revival_2010s"],
        "phase_2_6_enhancements": {
            "rhythmic_composition": True,
            "preset_count": len(VISUAL_KEI_RHYTHMIC_PRESETS),
            "periods": sorted(set(c["steps_per_cycle"] for c in VISUAL_KEI_RHYTHMIC_PRESETS.values())),
            "forced_orbit_integration": True,
            "patterns": ["sinusoidal", "triangular", "square"],
            "parameter_count": len(VISUAL_KEI_PARAMETER_NAMES),
            "canonical_state_count": len(VISUAL_KEI_CANONICAL_STATES),
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_type_count": len(VISUAL_KEI_VISUAL_TYPES),
            "prompt_modes": ["composite", "split_view", "sequence"],
            "nearest_neighbor_matching": True,
        },
        "tier_4d_ready": True,
    }, indent=2)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mcp.run()
