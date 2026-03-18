# Visual Kei Aesthetic Vocabulary MCP Server

A Lushy brick that translates natural language descriptions into authentic Visual Kei aesthetic parameters for image generation.

## Overview

Visual Kei (ビジュアル系) is a Japanese music movement characterized by elaborate visual aesthetics combining rock music with theatrical presentation. This MCP server provides comprehensive taxonomic mapping from natural language to specific visual parameters, enabling authentic character and styling generation.

## Architecture

### Three-Layer Design (Lushy Pattern)

```
User Intent → [Layer 1: Taxonomy] → [Layer 2: Mapping] → [Layer 3: Synthesis Interface]
                   ↓                      ↓                        ↓
              Zero LLM cost         Zero LLM cost          Returns data for Claude
```

- **Layer 1**: Pure taxonomy lookup - enumerates available options (from YAML ologs)
- **Layer 2**: Deterministic parameter assembly - maps selections to visual vocabulary  
- **Layer 3**: Synthesis interface - provides structured data for Claude to create final prompts

### Cost Optimization

- **Layers 1 & 2**: 100% deterministic, zero LLM tokens
- **Layer 3**: Single Claude API call for creative synthesis
- **Estimated savings**: 60-70% vs pure LLM approach

## Taxonomy Coverage

### Subgenres (7)
- **Kote Kei**: Classic Visual Kei - dramatic, theatrical, dark glamour
- **Oshare Kei**: Bright, colorful, pop-influenced - cheerful, playful
- **Nagoya Kei**: Dark, minimal, "gothic businessman" - brooding, mysterious
- **Angura Kei**: Traditional Japanese with dark themes - cultural, mystical
- **Eroguro Kei**: Erotic grotesque, disturbing imagery - shocking, provocative
- **Lolita Kei**: Elegant Gothic, Victorian/Rococo - aristocratic, refined
- **Iryou Kei**: Medical/hospital aesthetic - clinical, disturbing

### Historical Eras (4)
- **Pioneer (1980s)**: Raw, maximum drama, Western glam influence
- **Golden Age (1990s)**: Peak refinement, highest production value
- **Diversification (2000s)**: Multiple substyles, international awareness
- **Neo-Revival (2010s+)**: Modern techniques, social media aware

## Installation

### Development Setup
```bash
cd visual-kei-mcp
pip install -e ".[dev]"
```

### Run Tests
```bash
./tests/run_tests.sh
```

### FastMCP Cloud Deployment
```bash
fastmcp deploy src/visual_kei_mcp/server.py:mcp
```

## Available Tools

### Layer 1: Taxonomy Enumeration
- `list_subgenres` - List all 7 subgenres with descriptions
- `list_eras` - List all 4 historical eras
- `list_contexts` - List all visual contexts
- `list_archetypes` - List band member archetypes
- `get_subgenre_details` - Complete taxonomy for one subgenre
- `get_era_details` - Complete taxonomy for one era

### Layer 2: Parameter Mapping
- `analyze_intent` - Detect VK parameters from natural language
- `map_parameters` - Map selections to visual vocabulary
- `compare_subgenres` - Side-by-side subgenre comparison

### Layer 3: Enhancement Interface
- `enhance_with_visual_kei` - Complete enhancement workflow
- `generate_band_parameters` - Multi-member band generation

## License

Proprietary - Lushy Platform
