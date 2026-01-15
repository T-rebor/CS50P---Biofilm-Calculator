"""
project.py - Biofilm Property Calculator (Public Educational Version)
CS50P Final Project

A calculator for estimating biofilm/bioplastic properties based on formulation
composition. Uses data from published scientific literature.

Author: [Your Name]
Date: 2025
Course: CS50's Introduction to Programming with Python

EDUCATIONAL PURPOSE NOTICE:
This calculator is designed for learning programming concepts and understanding
biofilm property relationships. Results are estimates based on simplified models
and should NOT be used for actual material development without experimental
validation.

Features:
- Property calculation using Modified Rule of Mixtures
- Formulation optimization for target applications
- Comparison with commercial materials
- Report generation (text/markdown)
- Interactive CLI and command-line arguments
"""

import sys
import argparse
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# Import local database
from fiber_database import (
    FiberType, MatrixType, PlasticizerType,
    Fiber, Matrix, Plasticizer,
    FIBER_DATABASE, MATRIX_DATABASE, PLASTICIZER_DATABASE,
    REFERENCE_MATERIALS, ReferenceMaterial,
    get_fiber, get_matrix, get_plasticizer,
    list_available_fibers, list_available_matrices, list_available_plasticizers
)


# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "1.0.0-public"
HISTORY_FILE = "formulation_history.json"
DEFAULT_REFERENCES = ["LDPE", "PLA", "PBAT"]


# =============================================================================
# FORMULATION DATA CLASS
# =============================================================================

@dataclass
class Formulation:
    """
    Represents a biofilm formulation with all components and their proportions.

    Attributes:
        name: Identifier for this formulation
        fiber_type: Type of natural fiber reinforcement
        matrix_type: Type of biopolymer matrix
        plasticizer_type: Type of plasticizer
        fiber_content: Fiber percentage of total dry weight
        matrix_content: Matrix percentage of total dry weight
        plasticizer_content: Plasticizer percentage relative to matrix
    """
    name: str

    # Components
    fiber_type: FiberType
    matrix_type: MatrixType
    plasticizer_type: PlasticizerType

    # Proportions (weight percentages)
    fiber_content: float      # % of total dry weight
    matrix_content: float     # % of total dry weight
    plasticizer_content: float  # % relative to matrix

    # Optional parameters
    processing_temp: float = 70.0  # °C
    drying_time: float = 24.0      # hours

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the formulation parameters.

        Returns:
            Tuple of (is_valid, message)
        """
        fiber = get_fiber(self.fiber_type)

        # Check fiber concentration range
        min_conc, max_conc = fiber.recommended_concentration
        if not (min_conc <= self.fiber_content <= max_conc):
            return False, (f"Fiber content {self.fiber_content}% outside recommended "
                          f"range ({min_conc}-{max_conc}%) for {fiber.name}")

        # Check plasticizer range
        plasticizer = get_plasticizer(self.plasticizer_type)
        p_min, p_max = plasticizer.recommended_concentration
        if not (p_min <= self.plasticizer_content <= p_max):
            return False, (f"Plasticizer content {self.plasticizer_content}% outside "
                          f"recommended range ({p_min}-{p_max}%)")

        # Check total doesn't exceed 100%
        total = self.fiber_content + self.matrix_content
        if total > 100:
            return False, f"Total dry content ({total}%) exceeds 100%"

        return True, "Formulation is valid"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "fiber_type": self.fiber_type.value,
            "matrix_type": self.matrix_type.value,
            "plasticizer_type": self.plasticizer_type.value,
            "fiber_content": self.fiber_content,
            "matrix_content": self.matrix_content,
            "plasticizer_content": self.plasticizer_content,
            "processing_temp": self.processing_temp,
            "drying_time": self.drying_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Formulation':
        """Create Formulation from dictionary."""
        return cls(
            name=data["name"],
            fiber_type=FiberType(data["fiber_type"]),
            matrix_type=MatrixType(data["matrix_type"]),
            plasticizer_type=PlasticizerType(data["plasticizer_type"]),
            fiber_content=data["fiber_content"],
            matrix_content=data["matrix_content"],
            plasticizer_content=data["plasticizer_content"],
            processing_temp=data.get("processing_temp", 70.0),
            drying_time=data.get("drying_time", 24.0)
        )


# =============================================================================
# CALCULATED PROPERTIES DATA CLASS
# =============================================================================

@dataclass
class CalculatedProperties:
    """
    Stores calculated/estimated properties of a biofilm formulation.

    All mechanical properties in MPa, barrier in g·mm/m²·day·kPa.
    """
    # Mechanical properties
    tensile_strength: float      # MPa
    elongation_at_break: float   # %
    youngs_modulus: float        # MPa
    toughness_index: float       # Dimensionless

    # Barrier properties
    wvp: float                   # g·mm/m²·day·kPa
    water_absorption_24h: float  # %

    # Thermal properties
    degradation_onset: float     # °C

    # Sustainability metrics
    biodegradation_days: int
    bio_content: float           # % bio-based content

    # Quality indicators
    confidence_level: str        # "High", "Medium", "Low"
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tensile_strength": round(self.tensile_strength, 2),
            "elongation_at_break": round(self.elongation_at_break, 2),
            "youngs_modulus": round(self.youngs_modulus, 2),
            "toughness_index": round(self.toughness_index, 3),
            "wvp": round(self.wvp, 2),
            "water_absorption_24h": round(self.water_absorption_24h, 2),
            "degradation_onset": round(self.degradation_onset, 1),
            "biodegradation_days": self.biodegradation_days,
            "bio_content": round(self.bio_content, 1),
            "confidence_level": self.confidence_level,
            "warnings": self.warnings
        }


# =============================================================================
# CORE FUNCTION 1: calculate_properties()
# =============================================================================

def calculate_properties(formulation: Formulation) -> CalculatedProperties:
    """
    Calculate estimated properties for a biofilm formulation.

    Uses Modified Rule of Mixtures for composite materials, with empirical
    correction factors from literature for natural fiber composites.

    Args:
        formulation: A Formulation object with component specifications

    Returns:
        CalculatedProperties object with estimated values

    Raises:
        ValueError: If formulation components are not in database

    Example:
        >>> form = Formulation("Test", FiberType.SISAL, MatrixType.STARCH_CORN,
        ...                    PlasticizerType.GLYCEROL, 10.0, 90.0, 25.0)
        >>> props = calculate_properties(form)
        >>> print(f"Tensile Strength: {props.tensile_strength:.2f} MPa")
    """
    # Get component data
    fiber = get_fiber(formulation.fiber_type)
    matrix = get_matrix(formulation.matrix_type)
    plasticizer = get_plasticizer(formulation.plasticizer_type)

    # Convert percentages to fractions
    v_f = formulation.fiber_content / 100  # Fiber volume fraction
    v_m = formulation.matrix_content / 100  # Matrix volume fraction
    p_content = formulation.plasticizer_content / 100  # Plasticizer fraction

    warnings = []

    # =========================================================================
    # MECHANICAL PROPERTIES - Modified Rule of Mixtures
    # =========================================================================

    # Efficiency factor for natural fibers (typically 0.2-0.5)
    # Accounts for: random orientation, poor adhesion, aspect ratio
    eta_l = 0.35  # Length efficiency
    eta_o = 0.375  # Orientation factor (2D random = 3/8)
    fiber_efficiency = eta_l * eta_o

    # Tensile Strength (MPa)
    # Modified ROM: σc = η × Vf × σf + Vm × σm
    # Then adjust for plasticizer effect
    ts_composite = (fiber_efficiency * v_f * fiber.mechanical.tensile_strength +
                   v_m * matrix.mechanical.tensile_strength)
    ts_plasticized = ts_composite * (plasticizer.strength_factor ** p_content)

    # Ensure minimum realistic value
    tensile_strength = max(1.0, ts_plasticized)

    # Elongation at Break (%)
    # Plasticizer significantly increases elongation
    elong_base = (v_f * fiber.mechanical.elongation_at_break +
                 v_m * matrix.mechanical.elongation_at_break)
    elongation = elong_base * (plasticizer.flexibility_factor ** p_content)

    # Limit to realistic range
    elongation = min(elongation, 100.0)

    # Young's Modulus (MPa)
    # Halpin-Tsai simplified for short fibers
    E_f = fiber.mechanical.youngs_modulus
    E_m = matrix.mechanical.youngs_modulus

    # Simplified mixing rule with fiber efficiency
    youngs_modulus = (fiber_efficiency * v_f * E_f + v_m * E_m) * \
                     (1 - 0.3 * p_content)  # Plasticizer reduces stiffness

    # Toughness Index
    toughness_index = (tensile_strength * elongation) / 200

    # =========================================================================
    # BARRIER PROPERTIES
    # =========================================================================

    # Water Vapor Permeability (g·mm/m²·day·kPa)
    # Fibers generally increase WVP due to hydrophilicity
    wvp_matrix = matrix.barrier.wvp
    wvp_increase_factor = 1 + (v_f * 0.5)  # 50% increase per 100% fiber (linear approx)
    wvp_plasticizer = 1 + (p_content * (plasticizer.water_sensitivity_factor - 1))

    wvp = wvp_matrix * wvp_increase_factor * wvp_plasticizer

    # Water Absorption (24h, %)
    wa_matrix = matrix.barrier.water_absorption_24h
    wa_fiber_contribution = v_f * 100 * 0.8  # Cellulose fibers absorb water

    water_absorption = wa_matrix + wa_fiber_contribution
    water_absorption *= plasticizer.water_sensitivity_factor ** p_content

    # =========================================================================
    # THERMAL PROPERTIES
    # =========================================================================

    # Degradation onset - limiting component
    degradation_onset = min(
        fiber.thermal.degradation_onset,
        matrix.thermal.degradation_onset
    ) - (p_content * 20)  # Plasticizer slightly reduces thermal stability

    # =========================================================================
    # SUSTAINABILITY METRICS
    # =========================================================================

    # Biodegradation time (days) - slowest component dominates
    biodeg_fiber = fiber.biodegradation_days or 180
    biodeg_matrix = 90  # Assume starch-based ~90 days
    biodegradation_days = max(biodeg_fiber, biodeg_matrix)

    # Bio-based content (%)
    bio_content = 100.0  # All components are bio-based in this calculator

    # =========================================================================
    # CONFIDENCE ASSESSMENT
    # =========================================================================

    is_valid, validation_msg = formulation.validate()
    if not is_valid:
        warnings.append(validation_msg)

    # Check if within well-studied ranges
    if v_f < 0.05:
        confidence_level = "High"  # Low fiber, matrix-dominated
    elif v_f > 0.15:
        confidence_level = "Low"
        warnings.append("High fiber content may cause processing issues")
    else:
        confidence_level = "Medium"

    # Check plasticizer level
    if p_content > 0.35:
        warnings.append("High plasticizer may cause migration")
        confidence_level = "Low"

    return CalculatedProperties(
        tensile_strength=tensile_strength,
        elongation_at_break=elongation,
        youngs_modulus=youngs_modulus,
        toughness_index=toughness_index,
        wvp=wvp,
        water_absorption_24h=water_absorption,
        degradation_onset=degradation_onset,
        biodegradation_days=biodegradation_days,
        bio_content=bio_content,
        confidence_level=confidence_level,
        warnings=warnings
    )


# =============================================================================
# CORE FUNCTION 2: optimize_formulation()
# =============================================================================

@dataclass
class TargetProperties:
    """Target properties for optimization."""
    name: str
    min_tensile_strength: float = 0.0
    max_tensile_strength: float = 100.0
    min_elongation: float = 0.0
    max_elongation: float = 100.0
    max_wvp: float = 10.0
    priority: str = "balanced"  # "strength", "flexibility", "barrier", "balanced"


# Preset targets for common applications
APPLICATION_TARGETS = {
    "packaging": TargetProperties(
        name="Food Packaging",
        min_tensile_strength=5.0,
        max_wvp=3.0,
        min_elongation=10.0,
        priority="barrier"
    ),
    "agricultural": TargetProperties(
        name="Agricultural Mulch",
        min_tensile_strength=3.0,
        min_elongation=20.0,
        max_wvp=8.0,
        priority="flexibility"
    ),
    "rigid": TargetProperties(
        name="Rigid Container",
        min_tensile_strength=15.0,
        max_elongation=10.0,
        max_wvp=2.0,
        priority="strength"
    ),
    "flexible": TargetProperties(
        name="Flexible Film",
        min_tensile_strength=3.0,
        min_elongation=30.0,
        max_wvp=5.0,
        priority="flexibility"
    ),
}


def optimize_formulation(
    fiber_type: FiberType,
    target: TargetProperties,
    matrix_types: Optional[List[MatrixType]] = None,
    plasticizer_types: Optional[List[PlasticizerType]] = None,
    n_results: int = 5
) -> List[Tuple[Formulation, CalculatedProperties, float]]:
    """
    Find optimal formulation for target properties using grid search.

    Explores combinations of matrix, plasticizer, and concentrations
    to find formulations that best meet the target specifications.

    Args:
        fiber_type: Type of fiber to use (fixed)
        target: TargetProperties defining desired outcomes
        matrix_types: List of matrices to consider (None = all)
        plasticizer_types: List of plasticizers to consider (None = all)
        n_results: Number of top results to return

    Returns:
        List of (Formulation, CalculatedProperties, fitness_score) tuples,
        sorted by fitness score (higher is better)

    Example:
        >>> target = APPLICATION_TARGETS["packaging"]
        >>> results = optimize_formulation(FiberType.SISAL, target)
        >>> best = results[0]
        >>> print(f"Best formulation: {best[0].name}, Score: {best[2]:.2f}")
    """
    if matrix_types is None:
        matrix_types = list(MatrixType)
    if plasticizer_types is None:
        plasticizer_types = list(PlasticizerType)

    fiber = get_fiber(fiber_type)
    fiber_min, fiber_max = fiber.recommended_concentration

    results = []

    # Grid search over parameters
    fiber_range = [fiber_min, (fiber_min + fiber_max) / 2, fiber_max]
    plast_range = [20, 25, 30, 35]  # Common plasticizer levels

    for matrix_type in matrix_types:
        for plast_type in plasticizer_types:
            for fiber_pct in fiber_range:
                for plast_pct in plast_range:
                    # Create formulation
                    formulation = Formulation(
                        name=f"{fiber.name}_{matrix_type.name}_{fiber_pct}%",
                        fiber_type=fiber_type,
                        matrix_type=matrix_type,
                        plasticizer_type=plast_type,
                        fiber_content=fiber_pct,
                        matrix_content=100 - fiber_pct,
                        plasticizer_content=plast_pct
                    )

                    # Skip invalid formulations
                    is_valid, _ = formulation.validate()
                    if not is_valid:
                        continue

                    # Calculate properties
                    try:
                        props = calculate_properties(formulation)
                        score = _calculate_fitness_score(props, target)
                        results.append((formulation, props, score))
                    except ValueError:
                        continue

    # Sort by fitness score (descending)
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:n_results]


def _calculate_fitness_score(props: CalculatedProperties,
                            target: TargetProperties) -> float:
    """
    Calculate fitness score for optimization.

    Score is 0-100, with higher being better match to target.
    """
    score = 100.0

    # Tensile strength check
    if props.tensile_strength < target.min_tensile_strength:
        score -= (target.min_tensile_strength - props.tensile_strength) * 5
    if props.tensile_strength > target.max_tensile_strength:
        score -= (props.tensile_strength - target.max_tensile_strength) * 2

    # Elongation check
    if props.elongation_at_break < target.min_elongation:
        score -= (target.min_elongation - props.elongation_at_break) * 2
    if props.elongation_at_break > target.max_elongation:
        score -= (props.elongation_at_break - target.max_elongation) * 1

    # WVP check (barrier)
    if props.wvp > target.max_wvp:
        score -= (props.wvp - target.max_wvp) * 10

    # Priority bonuses
    if target.priority == "strength":
        score += props.tensile_strength * 0.5
    elif target.priority == "flexibility":
        score += props.elongation_at_break * 0.3
    elif target.priority == "barrier":
        score += max(0, (5 - props.wvp)) * 5

    # Confidence penalty
    if props.confidence_level == "Low":
        score *= 0.8
    elif props.confidence_level == "Medium":
        score *= 0.9

    return max(0, score)


# =============================================================================
# CORE FUNCTION 3: compare_materials()
# =============================================================================

def compare_materials(
    properties: CalculatedProperties,
    reference_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare calculated properties with reference commercial materials.

    Args:
        properties: Calculated properties of the biofilm
        reference_names: List of reference material names to compare
                        (None = use DEFAULT_REFERENCES)

    Returns:
        Dictionary with comparison data for each reference material.
        Values are percentages relative to reference (100% = equal).

    Example:
        >>> props = calculate_properties(formulation)
        >>> comparison = compare_materials(props, ["LDPE", "PLA"])
        >>> print(f"vs LDPE strength: {comparison['LDPE']['tensile_strength']}%")
    """
    if reference_names is None:
        reference_names = DEFAULT_REFERENCES

    comparisons = {}

    for ref_name in reference_names:
        if ref_name not in REFERENCE_MATERIALS:
            continue

        ref = REFERENCE_MATERIALS[ref_name]

        # Calculate percentage comparisons
        comparisons[ref_name] = {
            "tensile_strength": round(
                (properties.tensile_strength / ref.tensile_strength) * 100, 1
            ),
            "elongation": round(
                (properties.elongation_at_break / ref.elongation) * 100, 1
            ),
            "wvp": round(
                (ref.wvp / properties.wvp) * 100, 1  # Inverted - lower is better
            ) if properties.wvp > 0 else 0,
            "biodegradable_advantage": ref.biodegradable == False,
            "reference_biodegradable": ref.biodegradable
        }

    return comparisons


# =============================================================================
# CORE FUNCTION 4: generate_report()
# =============================================================================

def generate_report(
    formulation: Formulation,
    properties: CalculatedProperties,
    comparisons: Optional[Dict] = None,
    format: str = "text"
) -> str:
    """
    Generate a formatted report of the calculation results.

    Args:
        formulation: The input formulation
        properties: Calculated properties
        comparisons: Optional comparison data from compare_materials()
        format: Output format ("text" or "markdown")

    Returns:
        Formatted report string

    Example:
        >>> report = generate_report(formulation, props, comparisons)
        >>> print(report)
    """
    fiber = get_fiber(formulation.fiber_type)
    matrix = get_matrix(formulation.matrix_type)
    plasticizer = get_plasticizer(formulation.plasticizer_type)

    if format == "markdown":
        return _generate_markdown_report(formulation, properties,
                                        comparisons, fiber, matrix, plasticizer)

    # Text format (default)
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"  BIOFILM PROPERTY CALCULATOR - RESULTS")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # Formulation section - clarify composition basis
    lines.append(f"\n📋 FORMULATION: {formulation.name}")
    lines.append("-" * 40)
    lines.append(f"  Dry Basis Composition (Fiber + Matrix = 100%):")
    lines.append(f"    Fiber:       {fiber.name} ({formulation.fiber_content}%)")
    lines.append(f"    Matrix:      {matrix.name} ({formulation.matrix_content}%)")
    lines.append(f"  Plasticizer (relative to matrix):")
    lines.append(f"    {plasticizer.name}: {formulation.plasticizer_content}% w/w of matrix")

    # Properties section
    lines.append(f"\n📊 ESTIMATED PROPERTIES")
    lines.append("-" * 40)
    lines.append(f"  Tensile Strength:    {properties.tensile_strength:>8.2f} MPa")
    lines.append(f"  Elongation:          {properties.elongation_at_break:>8.2f} %")
    lines.append(f"  Young's Modulus:     {properties.youngs_modulus:>8.2f} MPa")
    lines.append(f"  Toughness Index:     {properties.toughness_index:>8.3f}")
    lines.append(f"  WVP:                 {properties.wvp:>8.2f} g·mm/m²·day·kPa")
    lines.append(f"  Water Absorption:    {properties.water_absorption_24h:>8.2f} %")
    lines.append(f"  Thermal Stability:   {properties.degradation_onset:>8.1f} °C")
    lines.append(f"  Biodegradation:      {properties.biodegradation_days:>8d} days")

    # Confidence
    lines.append(f"\n🎯 CONFIDENCE: {properties.confidence_level}")
    if properties.warnings:
        for warning in properties.warnings:
            lines.append(f"  ⚠️  {warning}")

    # Comparisons
    if comparisons:
        lines.append(f"\n📈 COMPARISON WITH COMMERCIAL MATERIALS")
        lines.append("-" * 40)

        for ref_name, data in comparisons.items():
            bio_icon = "🌱" if not data.get("reference_biodegradable", True) else "📦"
            lines.append(f"\n  vs {ref_name} {bio_icon}")
            lines.append(f"    Strength:   {data['tensile_strength']:>6.1f}% of reference")
            lines.append(f"    Elongation: {data['elongation']:>6.1f}% of reference")
            lines.append(f"    Barrier:    {data['wvp']:>6.1f}% of reference")

            if data.get("biodegradable_advantage"):
                lines.append(f"    ✓ Biodegradable advantage over non-bio reference")

    lines.append(f"\n{sep}")
    lines.append("Note: These are ESTIMATES for educational purposes.")
    lines.append("Validate experimentally before any practical application.")
    lines.append(sep)

    return "\n".join(lines)


def _generate_markdown_report(formulation, properties, comparisons,
                             fiber, matrix, plasticizer) -> str:
    """Generate markdown format report."""
    lines = []

    lines.append(f"# Biofilm Property Report: {formulation.name}")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    lines.append("## Formulation")
    lines.append("\n**Dry Basis Composition** (Fiber + Matrix = 100%)")
    lines.append("| Component | Material | Content |")
    lines.append("|-----------|----------|---------|")
    lines.append(f"| Fiber | {fiber.name} | {formulation.fiber_content}% of dry weight |")
    lines.append(f"| Matrix | {matrix.name} | {formulation.matrix_content}% of dry weight |")
    lines.append("")
    lines.append("**Plasticizer** (relative to matrix weight)")
    lines.append(f"| Plasticizer | {plasticizer.name} | {formulation.plasticizer_content}% w/w of matrix |")

    lines.append("\n## Estimated Properties")
    lines.append("| Property | Value | Unit |")
    lines.append("|----------|-------|------|")
    lines.append(f"| Tensile Strength | {properties.tensile_strength:.2f} | MPa |")
    lines.append(f"| Elongation | {properties.elongation_at_break:.2f} | % |")
    lines.append(f"| Young's Modulus | {properties.youngs_modulus:.2f} | MPa |")
    lines.append(f"| WVP | {properties.wvp:.2f} | g·mm/m²·day·kPa |")
    lines.append(f"| Biodegradation | {properties.biodegradation_days} | days |")

    lines.append(f"\n**Confidence Level:** {properties.confidence_level}")

    if properties.warnings:
        lines.append("\n### Warnings")
        for w in properties.warnings:
            lines.append(f"- ⚠️ {w}")

    lines.append("\n---")
    lines.append("*Educational estimate only. Validate experimentally.*")

    return "\n".join(lines)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_comparison_chart(
    properties: CalculatedProperties,
    comparisons: Dict[str, Dict]
) -> str:
    """Generate ASCII bar chart comparing properties."""
    lines = []
    lines.append("\n📊 PROPERTY COMPARISON CHART")
    lines.append("-" * 50)

    # Find max value for scaling
    all_ts = [properties.tensile_strength] + \
             [REFERENCE_MATERIALS[r].tensile_strength
              for r in comparisons if r in REFERENCE_MATERIALS]
    max_ts = max(all_ts)

    # Scale factor (50 chars max)
    scale = 40 / max_ts if max_ts > 0 else 1

    lines.append("\nTensile Strength (MPa):")

    # Biofilm
    bar_len = int(properties.tensile_strength * scale)
    bar = "█" * bar_len
    lines.append(f"  Biofilm : {bar} {properties.tensile_strength:.1f}")

    # References
    for ref_name in comparisons:
        if ref_name in REFERENCE_MATERIALS:
            ref = REFERENCE_MATERIALS[ref_name]
            bar_len = int(ref.tensile_strength * scale)
            bar = "▒" * bar_len
            lines.append(f"  {ref_name:8}: {bar} {ref.tensile_strength:.1f}")

    return "\n".join(lines)


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def save_to_history(formulation: Formulation,
                   properties: CalculatedProperties) -> bool:
    """Save calculation to history file."""
    try:
        history = load_history()
    except:
        history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "formulation": formulation.to_dict(),
        "properties": properties.to_dict()
    }

    history.append(entry)

    # Keep only last 50 entries
    history = history[-50:]

    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False


def load_history() -> List[dict]:
    """Load calculation history from file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def display_history():
    """Display calculation history."""
    history = load_history()

    if not history:
        print("\n📜 No calculation history found.")
        return

    print("\n" + "=" * 60)
    print("  CALCULATION HISTORY")
    print("=" * 60)

    for i, entry in enumerate(history[-10:], 1):  # Last 10
        form = entry["formulation"]
        props = entry["properties"]
        timestamp = entry.get("timestamp", "Unknown")

        print(f"\n{i}. {form['name']} ({timestamp[:10]})")
        print(f"   TS: {props['tensile_strength']:.1f} MPa | "
              f"Elong: {props['elongation_at_break']:.1f}% | "
              f"WVP: {props['wvp']:.1f}")


# =============================================================================
# INTERACTIVE FUNCTIONS
# =============================================================================

def interactive_calculation():
    """Interactive mode for new calculation."""
    print("\n" + "=" * 50)
    print("  NEW FORMULATION CALCULATION")
    print("=" * 50)

    # Get formulation name
    name = input("\nFormulation name: ").strip() or "Unnamed"

    # Select fiber
    print("\n🌿 Available Fibers:")
    fibers = list(FiberType)
    for i, ft in enumerate(fibers, 1):
        fiber = FIBER_DATABASE[ft]
        print(f"  {i}. {fiber.name} ({ft.value})")

    fiber_idx = int(input("\nSelect fiber (number): ")) - 1
    selected_fiber = fibers[fiber_idx]
    fiber = get_fiber(selected_fiber)

    # Get fiber content (dry basis: fiber + matrix = 100%)
    min_f, max_f = fiber.recommended_concentration
    print(f"\n  [Dry basis: Fiber + Matrix = 100%]")
    print(f"  Recommended fiber range: {min_f}-{max_f}%")
    fiber_content = float(input(f"  Fiber content (% of dry weight): ") or str((min_f + max_f) / 2))

    # Select matrix
    print("\n📦 Available Matrices:")
    matrices = list(MatrixType)
    for i, mt in enumerate(matrices, 1):
        matrix = MATRIX_DATABASE[mt]
        print(f"  {i}. {matrix.name}")

    matrix_idx = int(input("\nSelect matrix (number): ")) - 1
    selected_matrix = matrices[matrix_idx]
    matrix_content = 100 - fiber_content
    print(f"  Matrix content: {matrix_content}% (calculated as 100% - fiber)")

    # Select plasticizer
    print("\n💧 Available Plasticizers:")
    print("  [Plasticizer is expressed as % w/w relative to matrix weight]")
    plasticizers = list(PlasticizerType)
    for i, pt in enumerate(plasticizers, 1):
        plast = PLASTICIZER_DATABASE[pt]
        print(f"  {i}. {plast.name}")

    plast_idx = int(input("\nSelect plasticizer (number): ")) - 1
    selected_plasticizer = plasticizers[plast_idx]
    plast = get_plasticizer(selected_plasticizer)

    # Get plasticizer content (relative to matrix weight)
    min_p, max_p = plast.recommended_concentration
    print(f"\n  Recommended range: {min_p}-{max_p}% (w/w of matrix)")
    plast_content = float(input(f"  Plasticizer content (% w/w of matrix): ") or "25")

    # Create formulation
    formulation = Formulation(
        name=name,
        fiber_type=selected_fiber,
        matrix_type=selected_matrix,
        plasticizer_type=selected_plasticizer,
        fiber_content=fiber_content,
        matrix_content=matrix_content,
        plasticizer_content=plast_content
    )

    # Validate
    is_valid, message = formulation.validate()
    if not is_valid:
        print(f"\n⚠️ Warning: {message}")
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            return

    # Calculate
    print("\n⏳ Calculating properties...")
    properties = calculate_properties(formulation)
    comparisons = compare_materials(properties)
    report = generate_report(formulation, properties, comparisons)

    print(report)
    print(generate_comparison_chart(properties, comparisons))

    # Save to history
    save_to_history(formulation, properties)
    print("\n✓ Saved to history")


def interactive_optimization():
    """Interactive mode for formulation optimization."""
    print("\n" + "=" * 50)
    print("  FORMULATION OPTIMIZATION")
    print("=" * 50)

    # Select fiber
    print("\n🌿 Select Fiber to Optimize:")
    fibers = list(FiberType)
    for i, ft in enumerate(fibers, 1):
        fiber = FIBER_DATABASE[ft]
        print(f"  {i}. {fiber.name}")

    fiber_idx = int(input("\nSelect fiber (number): ")) - 1
    selected_fiber = fibers[fiber_idx]

    # Select application target
    print("\n🎯 Select Target Application:")
    targets = list(APPLICATION_TARGETS.keys())
    for i, target_name in enumerate(targets, 1):
        target = APPLICATION_TARGETS[target_name]
        print(f"  {i}. {target.name} ({target_name})")
        print(f"      Min Strength: {target.min_tensile_strength} MPa, "
              f"Max WVP: {target.max_wvp}, Priority: {target.priority}")

    target_idx = int(input("\nSelect application (number): ")) - 1
    selected_target_name = targets[target_idx]
    selected_target = APPLICATION_TARGETS[selected_target_name]

    # Number of results
    n_results = int(input("\nNumber of top results to show (default 5): ") or "5")

    # Run optimization
    print("\n⏳ Optimizing formulations...")
    print("   Testing combinations of matrices and plasticizers...")

    results = optimize_formulation(
        fiber_type=selected_fiber,
        target=selected_target,
        n_results=n_results
    )

    if not results:
        print("\n❌ No valid formulations found for these constraints.")
        return

    # Display results
    print("\n" + "=" * 60)
    print(f"  🏆 TOP {len(results)} OPTIMIZED FORMULATIONS")
    print(f"  Target: {selected_target.name}")
    print("=" * 60)

    for rank, (formulation, props, score) in enumerate(results, 1):
        print(f"\n{'─' * 55}")
        print(f"  #{rank} - Score: {score:.1f}/100")
        print(f"{'─' * 55}")
        print(f"  Composition (dry basis: fiber + matrix = 100%):")
        print(f"    Fiber: {formulation.fiber_type.value} @ {formulation.fiber_content}%")
        print(f"    Matrix: {formulation.matrix_type.value} @ {formulation.matrix_content}%")
        print(f"  Plasticizer (w/w of matrix):")
        print(f"    {formulation.plasticizer_type.value} @ {formulation.plasticizer_content}%")
        print(f"\n  Properties:")
        print(f"    Tensile Strength: {props.tensile_strength:.2f} MPa")
        print(f"    Young's Modulus:  {props.youngs_modulus:.2f} GPa")
        print(f"    Elongation:       {props.elongation_at_break:.1f}%")
        print(f"    WVP:              {props.wvp:.2f} g·mm/m²·day·kPa")

        # Show if meets targets
        meets = []
        if props.tensile_strength >= selected_target.min_tensile_strength:
            meets.append("✓ Strength")
        else:
            meets.append("✗ Strength")
        if props.wvp <= selected_target.max_wvp:
            meets.append("✓ Barrier")
        else:
            meets.append("✗ Barrier")
        if props.elongation_at_break >= selected_target.min_elongation:
            meets.append("✓ Elongation")
        else:
            meets.append("✗ Elongation")

        print(f"\n  Target Compliance: {' | '.join(meets)}")

    # Ask to save best
    save = input("\n\nSave best formulation to history? (y/n): ").lower()
    if save == 'y' and results:
        best_form, best_props, _ = results[0]
        save_to_history(best_form, best_props)
        print("✓ Best formulation saved to history")

    # Ask to generate detailed report
    report_opt = input("Generate detailed report for best formulation? (y/n): ").lower()
    if report_opt == 'y' and results:
        best_form, best_props, _ = results[0]
        comparisons = compare_materials(best_props)
        report = generate_report(best_form, best_props, comparisons, format_type="markdown")

        filename = f"optimized_{selected_target_name}_{selected_fiber.value}.md"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {filename}")


def view_materials():
    """Display available materials."""
    print("\n" + "=" * 50)
    print("  MATERIALS DATABASE")
    print("=" * 50)

    print("\n🌿 FIBERS")
    print("-" * 40)
    for fiber in FIBER_DATABASE.values():
        print(f"\n  {fiber.name} ({fiber.scientific_name})")
        print(f"    Cellulose: {fiber.composition.cellulose}%")
        print(f"    Tensile Strength: {fiber.mechanical.tensile_strength} MPa")
        print(f"    Range: {fiber.recommended_concentration[0]}-"
              f"{fiber.recommended_concentration[1]}%")

    print("\n\n📦 MATRICES")
    print("-" * 40)
    for matrix in MATRIX_DATABASE.values():
        print(f"\n  {matrix.name}")
        print(f"    Tensile Strength: {matrix.mechanical.tensile_strength} MPa")
        print(f"    Elongation: {matrix.mechanical.elongation_at_break}%")

    print("\n\n💧 PLASTICIZERS")
    print("-" * 40)
    for plast in PLASTICIZER_DATABASE.values():
        print(f"\n  {plast.name}")
        print(f"    Flexibility: ×{plast.flexibility_factor}")
        print(f"    Range: {plast.recommended_concentration[0]}-"
              f"{plast.recommended_concentration[1]}%")

    input("\n\nPress Enter to continue...")


def show_about():
    """Show project information."""
    print("\n" + "=" * 60)
    print("  ABOUT THIS PROJECT")
    print("=" * 60)
    print(f"""
    🧬 Biofilm Property Calculator v{VERSION}

    CS50's Introduction to Programming with Python
    Harvard University - Final Project

    This calculator estimates properties of biofilms/bioplastics
    based on formulation composition using the Modified Rule of
    Mixtures for composite materials.

    📊 Data Sources (Published Literature):
    - Faruk et al. (2012) - Prog Polym Sci
    - Satyanarayana et al. (2009) - Prog Polym Sci
    - Siqueira et al. (2010) - Polymers
    - Jawaid & Abdul Khalil (2011) - Carbohydr Polym

    🌿 Fibers Available:
    - Sisal (Agave sisalana)
    - Jute (Corchorus capsularis)
    - Coir (Cocos nucifera)
    - Hemp (Cannabis sativa)
    - Flax (Linum usitatissimum)
    - Banana (Musa acuminata)

    ⚠️ EDUCATIONAL PURPOSE:
    This tool is for learning programming and understanding
    composite material relationships. Results are estimates
    and should NOT be used for actual material development
    without experimental validation.

    📝 License: Educational use (CS50P)
    """)
    input("\nPress Enter to continue...")


# =============================================================================
# CLI FUNCTIONS
# =============================================================================

def parse_fiber_type(value: str) -> FiberType:
    """Parse fiber type from string (CLI argument)."""
    value_lower = value.lower().replace("-", "_").replace(" ", "_")

    # Direct enum value match
    for ft in FiberType:
        if ft.value.lower() == value_lower or ft.name.lower() == value_lower:
            return ft

    # Partial match
    for ft in FiberType:
        if value_lower in ft.value.lower() or value_lower in ft.name.lower():
            return ft

    raise ValueError(f"Unknown fiber type: {value}. "
                    f"Available: {', '.join(ft.value for ft in FiberType)}")


def parse_matrix_type(value: str) -> MatrixType:
    """Parse matrix type from string (CLI argument)."""
    value_lower = value.lower().replace("-", "_").replace(" ", "_")

    for mt in MatrixType:
        if mt.value.lower() == value_lower or mt.name.lower() == value_lower:
            return mt

    # Partial match
    for mt in MatrixType:
        if value_lower in mt.value.lower() or value_lower in mt.name.lower():
            return mt

    raise ValueError(f"Unknown matrix type: {value}. "
                    f"Available: {', '.join(mt.value for mt in MatrixType)}")


def parse_plasticizer_type(value: str) -> PlasticizerType:
    """Parse plasticizer type from string (CLI argument)."""
    value_lower = value.lower().replace("-", "_").replace(" ", "_")

    for pt in PlasticizerType:
        if pt.value.lower() == value_lower or pt.name.lower() == value_lower:
            return pt

    # Partial match
    for pt in PlasticizerType:
        if value_lower in pt.value.lower() or value_lower in pt.name.lower():
            return pt

    raise ValueError(f"Unknown plasticizer type: {value}. "
                    f"Available: {', '.join(pt.value for pt in PlasticizerType)}")


def cli_quick_calculate(fiber: str, matrix: str, plasticizer: str,
                       fiber_pct: float, plast_pct: float, show_chart: bool = False):
    """Quick calculation from CLI arguments."""
    try:
        fiber_type = parse_fiber_type(fiber)
        matrix_type = parse_matrix_type(matrix)
        plast_type = parse_plasticizer_type(plasticizer)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    formulation = Formulation(
        name=f"CLI_{fiber}_{matrix}",
        fiber_type=fiber_type,
        matrix_type=matrix_type,
        plasticizer_type=plast_type,
        fiber_content=fiber_pct,
        matrix_content=100 - fiber_pct,
        plasticizer_content=plast_pct
    )

    is_valid, message = formulation.validate()
    if not is_valid:
        print(f"⚠️ Warning: {message}")

    properties = calculate_properties(formulation)
    comparisons = compare_materials(properties)
    report = generate_report(formulation, properties, comparisons)

    print(report)

    if show_chart:
        print(generate_comparison_chart(properties, comparisons))

    save_to_history(formulation, properties)


def cli_optimize(application: str, fiber: str = None, show_chart: bool = False):
    """Optimization from CLI arguments."""
    # Validate application
    if application.lower() not in APPLICATION_TARGETS:
        print(f"❌ Unknown application: {application}")
        print(f"   Available: {', '.join(APPLICATION_TARGETS.keys())}")
        sys.exit(1)

    target = APPLICATION_TARGETS[application.lower()]

    # Parse fiber if provided
    if fiber:
        try:
            fiber_type = parse_fiber_type(fiber)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        fiber_type = FiberType.SISAL  # Default

    print(f"\n⏳ Optimizing for {target.name} with {fiber_type.value}...")

    results = optimize_formulation(
        fiber_type=fiber_type,
        target=target,
        n_results=5
    )

    if not results:
        print("❌ No valid formulations found.")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 60)
    print(f"  🏆 TOP {len(results)} OPTIMIZED FORMULATIONS")
    print(f"  Application: {target.name}")
    print("=" * 60)

    for rank, (formulation, props, score) in enumerate(results, 1):
        print(f"\n#{rank} [Score: {score:.1f}/100]")
        print(f"   Dry basis (fiber + matrix = 100%):")
        print(f"     Fiber: {formulation.fiber_type.value} @ {formulation.fiber_content}%")
        print(f"     Matrix: {formulation.matrix_type.value} @ {formulation.matrix_content}%")
        print(f"   Plasticizer ({formulation.plasticizer_content}% w/w of matrix):")
        print(f"     {formulation.plasticizer_type.value}")
        print(f"   ├─ Tensile: {props.tensile_strength:.2f} MPa")
        print(f"   ├─ Elongation: {props.elongation_at_break:.1f}%")
        print(f"   └─ WVP: {props.wvp:.2f} g·mm/m²·day·kPa")

    if show_chart:
        best_form, best_props, _ = results[0]
        comparisons = compare_materials(best_props)
        print(generate_comparison_chart(best_props, comparisons))


def cli_list_materials(category: str = "all"):
    """List available materials from CLI."""
    category = category.lower()

    if category in ["fiber", "fibers", "all"]:
        print("\n🌿 AVAILABLE FIBERS")
        print("-" * 40)
        for ft in FiberType:
            fiber = FIBER_DATABASE[ft]
            print(f"  {ft.value:20} - {fiber.name}")
            print(f"    {'':20}   TS: {fiber.mechanical.tensile_strength} MPa, "
                  f"Range: {fiber.recommended_concentration[0]}-{fiber.recommended_concentration[1]}%")

    if category in ["matrix", "matrices", "all"]:
        print("\n📦 AVAILABLE MATRICES")
        print("-" * 40)
        for mt in MatrixType:
            matrix = MATRIX_DATABASE[mt]
            print(f"  {mt.value:20} - {matrix.name}")

    if category in ["plasticizer", "plasticizers", "all"]:
        print("\n💧 AVAILABLE PLASTICIZERS")
        print("-" * 40)
        for pt in PlasticizerType:
            plast = PLASTICIZER_DATABASE[pt]
            print(f"  {pt.value:20} - {plast.name}")

    if category in ["application", "applications", "targets", "all"]:
        print("\n🎯 APPLICATION TARGETS")
        print("-" * 40)
        for name, target in APPLICATION_TARGETS.items():
            print(f"  {name:15} - {target.name}")
            print(f"    {'':15}   Min TS: {target.min_tensile_strength} MPa, "
                  f"Max WVP: {target.max_wvp}, Priority: {target.priority}")


def cli_show_history(n: int = 10):
    """Show calculation history from CLI."""
    history = load_history()

    if not history:
        print("\n📋 No calculation history found.")
        return

    print(f"\n📋 CALCULATION HISTORY (Last {min(n, len(history))} entries)")
    print("=" * 60)

    for entry in history[-n:]:
        timestamp = entry.get("timestamp", "Unknown")
        form = entry.get("formulation", {})
        props = entry.get("properties", {})

        print(f"\n{timestamp}")
        print(f"  Name: {form.get('name', 'Unnamed')}")
        print(f"  Dry basis: Fiber {form.get('fiber_content', '?')}% + Matrix {form.get('matrix_content', '?')}%")
        print(f"  Fiber type: {form.get('fiber_type', '?')}")
        print(f"  Plasticizer: {form.get('plasticizer_content', '?')}% w/w of matrix")
        print(f"  Tensile: {props.get('tensile_strength', 0):.2f} MPa, "
              f"WVP: {props.get('wvp', 0):.2f}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="🧬 Biofilm Property Calculator - CS50P Final Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project.py                           # Interactive mode
  python project.py --quick sisal corn glycerol 10 25
  python project.py --optimize packaging --fiber sisal --chart
  python project.py --list fibers
  python project.py --history
        """
    )

    # Quick calculation
    parser.add_argument(
        "--quick", "-q",
        nargs=5,
        metavar=("FIBER", "MATRIX", "PLASTICIZER", "FIBER%", "PLAST%"),
        help="Quick calculation: fiber matrix plasticizer fiber%% plasticizer%%"
    )

    # Optimization
    parser.add_argument(
        "--optimize", "-o",
        metavar="APPLICATION",
        help="Optimize for application: packaging, agricultural, rigid, flexible"
    )

    parser.add_argument(
        "--fiber", "-f",
        metavar="FIBER",
        help="Fiber type for optimization (default: sisal)"
    )

    # List materials
    parser.add_argument(
        "--list", "-l",
        nargs="?",
        const="all",
        metavar="CATEGORY",
        help="List available materials: fibers, matrices, plasticizers, applications, all"
    )

    # History
    parser.add_argument(
        "--history", "-H",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Show last N calculations (default: 10)"
    )

    # Chart option
    parser.add_argument(
        "--chart", "-c",
        action="store_true",
        help="Show comparison chart"
    )

    # Version
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Biofilm Property Calculator v{VERSION}"
    )

    return parser


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function - Entry point for Biofilm Property Calculator.

    Supports both CLI arguments and interactive menu.
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle CLI arguments
    if args.quick:
        fiber, matrix, plast, fiber_pct, plast_pct = args.quick
        cli_quick_calculate(
            fiber, matrix, plast,
            float(fiber_pct), float(plast_pct),
            show_chart=args.chart
        )
        return

    if args.optimize:
        cli_optimize(args.optimize, args.fiber, show_chart=args.chart)
        return

    if args.list:
        cli_list_materials(args.list)
        return

    if args.history is not None:
        cli_show_history(args.history)
        return

    # Interactive mode
    print("\n" + "=" * 60)
    print("  🧬 BIOFILM PROPERTY CALCULATOR")
    print("  CS50P Final Project - Educational Version")
    print(f"  Version {VERSION}")
    print("=" * 60)
    print("\n  Tip: Use --help for command line options")

    while True:
        print("\n📋 MAIN MENU")
        print("-" * 40)
        print("1. Calculate properties for new formulation")
        print("2. Optimize formulation for application")
        print("3. View available materials")
        print("4. View calculation history")
        print("5. About this project")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            interactive_calculation()
        elif choice == "2":
            interactive_optimization()
        elif choice == "3":
            view_materials()
        elif choice == "4":
            display_history()
        elif choice == "5":
            show_about()
        elif choice == "0":
            print("\n👋 Thank you for using Biofilm Property Calculator!")
            print("   Developed for CS50P - Harvard University")
            print("   Remember: Validate experimentally!\n")
            sys.exit(0)
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
