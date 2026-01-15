"""
fiber_database.py - Public Version
Database of natural fiber properties for biofilm calculations.

All data sourced from PUBLISHED peer-reviewed literature only.
No proprietary or unpublished research data.

Literature Sources:
- Siqueira et al. (2010) - Cellulosic bionanocomposites: A review
- Satyanarayana et al. (2009) - Natural fiber-reinforced composites
- Faruk et al. (2012) - Biocomposites reinforced with natural fibers
- Jawaid & Abdul Khalil (2011) - Cellulosic/synthetic fibre reinforced composites
- Gurunathan et al. (2015) - A review of natural fiber composites

This database is intended for EDUCATIONAL purposes (CS50P Final Project).
Values represent typical ranges from literature and should be validated
experimentally before any practical application.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


# =============================================================================
# ENUMS - Material Types
# =============================================================================

class FiberType(Enum):
    """Types of natural fibers available in the database."""
    SISAL = "agave_sisalana"
    JUTE = "corchorus_capsularis"
    COIR = "cocos_nucifera"
    HEMP = "cannabis_sativa"
    FLAX = "linum_usitatissimum"
    BANANA = "musa_acuminata"


class MatrixType(Enum):
    """Types of biopolymer matrices."""
    STARCH_CORN = "corn_starch"
    STARCH_POTATO = "potato_starch"
    PLA = "polylactic_acid"
    CHITOSAN = "chitosan"
    GELATIN = "gelatin"


class PlasticizerType(Enum):
    """Types of plasticizers."""
    GLYCEROL = "glycerol"
    SORBITOL = "sorbitol"
    PEG_400 = "polyethylene_glycol_400"
    CITRIC_ACID = "citric_acid"


# =============================================================================
# DATA CLASSES - Property Structures
# =============================================================================

@dataclass
class ChemicalComposition:
    """
    Chemical composition of a natural fiber (percentages).

    Standard components for lignocellulosic fibers.
    """
    cellulose: float          # Main structural component
    hemicellulose: float      # Secondary polysaccharide
    lignin: float             # Aromatic polymer
    extractives: float = 0.0  # Waxes, oils, etc.
    ash: float = 0.0          # Mineral content
    moisture: float = 0.0     # Water content

    def validate(self) -> bool:
        """Validate that composition sums to approximately 100%."""
        total = self.get_total()
        return 90.0 <= total <= 110.0  # Allow 10% tolerance for literature variation

    def get_total(self) -> float:
        """Return total percentage."""
        return (self.cellulose + self.hemicellulose + self.lignin +
                self.extractives + self.ash + self.moisture)


@dataclass
class MechanicalProperties:
    """
    Mechanical properties of a material.

    All values in standard SI units.
    """
    tensile_strength: float      # MPa
    elongation_at_break: float   # %
    youngs_modulus: float        # MPa (or GPa for fibers, converted)
    flexural_strength: Optional[float] = None  # MPa

    def get_toughness_index(self) -> float:
        """
        Calculate a simple toughness index.

        Toughness ≈ (Tensile Strength × Elongation) / 2
        Provides relative comparison between materials.
        """
        return (self.tensile_strength * self.elongation_at_break) / 200


@dataclass
class BarrierProperties:
    """
    Barrier properties of a biofilm.

    Critical for packaging applications.
    """
    wvp: float                    # Water Vapor Permeability (g·mm/m²·day·kPa)
    wvtr: Optional[float] = None  # Water Vapor Transmission Rate (g/m²·day)
    oxygen_permeability: Optional[float] = None  # cm³·μm/m²·day·kPa
    water_absorption_24h: Optional[float] = None  # % mass gain


@dataclass
class ThermalProperties:
    """
    Thermal properties of a material.

    From TGA/DSC analysis.
    """
    degradation_onset: float      # °C (TGA - 5% mass loss)
    max_degradation_temp: float   # °C (peak DTG)
    glass_transition: Optional[float] = None  # °C (Tg from DSC)
    melting_point: Optional[float] = None     # °C (Tm)


# =============================================================================
# MAIN DATA CLASSES
# =============================================================================

@dataclass
class Fiber:
    """
    Complete fiber data structure.

    Contains all properties needed for biofilm calculation.
    """
    name: str
    scientific_name: str
    fiber_type: FiberType
    composition: ChemicalComposition
    mechanical: MechanicalProperties
    thermal: ThermalProperties
    density: float  # g/cm³

    # Processing characteristics
    recommended_concentration: Tuple[float, float]  # (min%, max%)
    treatment_required: bool = True
    treatment_type: str = "alkaline"

    # Sustainability metrics
    biodegradation_days: Optional[int] = None
    availability: str = "Worldwide"

    # Literature source
    source: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.scientific_name})"


@dataclass
class Matrix:
    """
    Biopolymer matrix data structure.

    Base material for the biofilm.
    """
    name: str
    matrix_type: MatrixType
    mechanical: MechanicalProperties
    barrier: BarrierProperties
    thermal: ThermalProperties
    density: float  # g/cm³

    # Processing parameters
    gelatinization_temp: float  # °C
    recommended_concentration: Tuple[float, float]  # (min%, max%) in water

    # Cost and availability
    cost_per_kg: float  # USD (approximate)
    source: str = ""


@dataclass
class Plasticizer:
    """
    Plasticizer data structure.

    Modifies flexibility and processability.
    """
    name: str
    plasticizer_type: PlasticizerType
    density: float  # g/cm³

    # Effect on properties (multipliers relative to base)
    flexibility_factor: float      # >1 increases flexibility
    strength_factor: float         # <1 typically reduces strength
    water_sensitivity_factor: float  # >1 increases water absorption

    # Recommended usage
    recommended_concentration: Tuple[float, float]  # (min%, max%) relative to matrix

    cost_per_kg: float  # USD (approximate)
    source: str = ""


@dataclass
class ReferenceMaterial:
    """Reference commercial material for comparison."""
    name: str
    tensile_strength: float  # MPa
    elongation: float        # %
    wvp: float              # g·mm/m²·day·kPa
    biodegradable: bool
    source: str = ""


# =============================================================================
# FIBER DATABASE - Literature Values Only
# =============================================================================

FIBER_DATABASE: Dict[FiberType, Fiber] = {

    FiberType.SISAL: Fiber(
        name="Sisal",
        scientific_name="Agave sisalana",
        fiber_type=FiberType.SISAL,
        composition=ChemicalComposition(
            cellulose=65.0,      # Range: 60-70%
            hemicellulose=12.0,  # Range: 10-15%
            lignin=10.0,         # Range: 8-12%
            extractives=2.0,
            ash=1.0,
            moisture=10.0
        ),
        mechanical=MechanicalProperties(
            tensile_strength=510.0,   # Range: 400-700 MPa
            elongation_at_break=2.5,  # Range: 2-3%
            youngs_modulus=9400.0     # Range: 9-22 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=220.0,
            max_degradation_temp=340.0,
            glass_transition=None
        ),
        density=1.45,
        recommended_concentration=(5.0, 20.0),
        treatment_required=True,
        treatment_type="alkaline_5%_NaOH",
        biodegradation_days=120,
        availability="Brazil, Mexico, Africa",
        source="Faruk O, Bledzki AK, Fink HP, Sain M. Biocomposites reinforced with natural fibers: 2000-2010. Prog Polym Sci. 2012;37(11):1552-1596. doi:10.1016/j.progpolymsci.2012.04.003"
    ),

    FiberType.JUTE: Fiber(
        name="Jute",
        scientific_name="Corchorus capsularis",
        fiber_type=FiberType.JUTE,
        composition=ChemicalComposition(
            cellulose=64.0,      # Range: 61-71%
            hemicellulose=22.0,  # Range: 14-20%
            lignin=12.0,         # Range: 12-13%
            extractives=0.5,
            ash=0.5,
            moisture=1.0
        ),
        mechanical=MechanicalProperties(
            tensile_strength=450.0,   # Range: 393-773 MPa
            elongation_at_break=1.7,  # Range: 1.5-1.8%
            youngs_modulus=15000.0    # Range: 10-30 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=210.0,
            max_degradation_temp=330.0,
            glass_transition=None
        ),
        density=1.46,
        recommended_concentration=(5.0, 25.0),
        treatment_required=True,
        treatment_type="alkaline_2%_NaOH",
        biodegradation_days=90,
        availability="Bangladesh, India",
        source="Jawaid M, Abdul Khalil HPS. Cellulosic/synthetic fibre reinforced polymer hybrid composites: A review. Carbohydr Polym. 2011;86(1):1-18. doi:10.1016/j.carbpol.2011.04.043"
    ),

    FiberType.COIR: Fiber(
        name="Coir (Coconut)",
        scientific_name="Cocos nucifera",
        fiber_type=FiberType.COIR,
        composition=ChemicalComposition(
            cellulose=43.0,      # Range: 36-43%
            hemicellulose=0.3,   # Very low
            lignin=45.0,         # High lignin: 41-45%
            extractives=4.0,
            ash=2.0,
            moisture=5.7
        ),
        mechanical=MechanicalProperties(
            tensile_strength=175.0,   # Range: 131-220 MPa
            elongation_at_break=25.0, # High elongation: 15-40%
            youngs_modulus=4500.0     # Range: 4-6 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=190.0,
            max_degradation_temp=320.0,
            glass_transition=None
        ),
        density=1.25,
        recommended_concentration=(5.0, 15.0),
        treatment_required=True,
        treatment_type="alkaline_5%_NaOH",
        biodegradation_days=180,
        availability="Tropical regions worldwide",
        source="Satyanarayana KG, Arizaga GGC, Wypych F. Biodegradable composites based on lignocellulosic fibers—An overview. Prog Polym Sci. 2009;34(9):982-1021. doi:10.1016/j.progpolymsci.2008.12.002"
    ),

    FiberType.HEMP: Fiber(
        name="Hemp",
        scientific_name="Cannabis sativa",
        fiber_type=FiberType.HEMP,
        composition=ChemicalComposition(
            cellulose=70.0,      # Range: 68-74%
            hemicellulose=18.0,  # Range: 15-22%
            lignin=4.0,          # Low lignin: 3.7-5.7%
            extractives=1.0,
            ash=0.8,
            moisture=6.2
        ),
        mechanical=MechanicalProperties(
            tensile_strength=690.0,   # Range: 550-900 MPa
            elongation_at_break=1.6,  # Range: 1.6%
            youngs_modulus=30000.0    # Range: 30-60 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=250.0,
            max_degradation_temp=360.0,
            glass_transition=None
        ),
        density=1.48,
        recommended_concentration=(5.0, 20.0),
        treatment_required=True,
        treatment_type="alkaline_2%_NaOH",
        biodegradation_days=90,
        availability="Europe, China, Canada",
        source="Gurunathan T, Mohanty S, Nayak SK. A review of the recent developments in biocomposites based on natural fibres and their application perspectives. Compos Part B Eng. 2015;99:293-307. doi:10.1016/j.compositesb.2015.08.005"
    ),

    FiberType.FLAX: Fiber(
        name="Flax",
        scientific_name="Linum usitatissimum",
        fiber_type=FiberType.FLAX,
        composition=ChemicalComposition(
            cellulose=71.0,      # Range: 64-71%
            hemicellulose=19.0,  # Range: 18.6-20.6%
            lignin=2.2,          # Low lignin: 2-2.2%
            extractives=1.5,
            ash=1.5,
            moisture=4.8
        ),
        mechanical=MechanicalProperties(
            tensile_strength=800.0,   # Range: 345-1500 MPa
            elongation_at_break=2.7,  # Range: 2.7-3.2%
            youngs_modulus=27000.0    # Range: 27-38 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=240.0,
            max_degradation_temp=350.0,
            glass_transition=None
        ),
        density=1.50,
        recommended_concentration=(5.0, 25.0),
        treatment_required=True,
        treatment_type="alkaline_1%_NaOH",
        biodegradation_days=100,
        availability="Europe, Canada",
        source="Siqueira G, Bras J, Dufresne A. Cellulosic bionanocomposites: A review of preparation, properties and applications. Polymers. 2010;2(4):728-765. doi:10.3390/polym2040728"
    ),

    FiberType.BANANA: Fiber(
        name="Banana (Pseudostem)",
        scientific_name="Musa acuminata",
        fiber_type=FiberType.BANANA,
        composition=ChemicalComposition(
            cellulose=63.0,      # Range: 60-65%
            hemicellulose=19.0,  # Range: 6-19%
            lignin=5.0,          # Range: 5-10%
            extractives=4.0,
            ash=1.0,
            moisture=8.0
        ),
        mechanical=MechanicalProperties(
            tensile_strength=355.0,   # Range: 300-600 MPa
            elongation_at_break=3.0,  # Range: 3-10%
            youngs_modulus=8000.0     # Range: 8-20 GPa
        ),
        thermal=ThermalProperties(
            degradation_onset=200.0,
            max_degradation_temp=330.0,
            glass_transition=None
        ),
        density=1.35,
        recommended_concentration=(5.0, 15.0),
        treatment_required=True,
        treatment_type="alkaline_1%_NaOH",
        biodegradation_days=60,
        availability="Tropical regions worldwide",
        source="Faruk O, Bledzki AK, Fink HP, Sain M. Biocomposites reinforced with natural fibers: 2000-2010. Prog Polym Sci. 2012;37(11):1552-1596. doi:10.1016/j.progpolymsci.2012.04.003"
    ),
}


# =============================================================================
# MATRIX DATABASE - Literature Values Only
# =============================================================================

MATRIX_DATABASE: Dict[MatrixType, Matrix] = {

    MatrixType.STARCH_CORN: Matrix(
        name="Corn Starch",
        matrix_type=MatrixType.STARCH_CORN,
        mechanical=MechanicalProperties(
            tensile_strength=5.0,     # Range: 2-10 MPa (plasticized)
            elongation_at_break=30.0, # High with plasticizer
            youngs_modulus=100.0
        ),
        barrier=BarrierProperties(
            wvp=3.0,                  # Typical: 2-5 g·mm/m²·day·kPa
            water_absorption_24h=45.0
        ),
        thermal=ThermalProperties(
            degradation_onset=280.0,
            max_degradation_temp=320.0,
            glass_transition=60.0
        ),
        density=1.50,
        gelatinization_temp=70.0,
        recommended_concentration=(3.0, 8.0),
        cost_per_kg=1.50,
        source="Jiménez A, Fabra MJ, Talens P, Chiralt A. Edible and biodegradable starch films: A review. Food Bioprocess Technol. 2012;5(6):2058-2076. doi:10.1007/s11947-012-0835-4"
    ),

    MatrixType.STARCH_POTATO: Matrix(
        name="Potato Starch",
        matrix_type=MatrixType.STARCH_POTATO,
        mechanical=MechanicalProperties(
            tensile_strength=4.5,
            elongation_at_break=35.0,
            youngs_modulus=80.0
        ),
        barrier=BarrierProperties(
            wvp=3.5,
            water_absorption_24h=50.0
        ),
        thermal=ThermalProperties(
            degradation_onset=275.0,
            max_degradation_temp=315.0,
            glass_transition=55.0
        ),
        density=1.48,
        gelatinization_temp=65.0,
        recommended_concentration=(3.0, 8.0),
        cost_per_kg=1.80,
        source="Mali S, Grossmann MVE, García MA, Martino MN, Zaritzky NE. Barrier, mechanical and optical properties of plasticized yam starch films. Carbohydr Polym. 2004;56(2):129-135. doi:10.1016/j.carbpol.2004.01.004"
    ),

    MatrixType.PLA: Matrix(
        name="Polylactic Acid",
        matrix_type=MatrixType.PLA,
        mechanical=MechanicalProperties(
            tensile_strength=50.0,    # Range: 40-60 MPa
            elongation_at_break=6.0,  # Brittle
            youngs_modulus=3500.0
        ),
        barrier=BarrierProperties(
            wvp=0.5,                  # Good barrier
            water_absorption_24h=1.0
        ),
        thermal=ThermalProperties(
            degradation_onset=300.0,
            max_degradation_temp=370.0,
            glass_transition=60.0,
            melting_point=170.0
        ),
        density=1.24,
        gelatinization_temp=180.0,  # Processing temp
        recommended_concentration=(100.0, 100.0),  # Used pure
        cost_per_kg=3.50,
        source="Auras R, Harte B, Selke S. An overview of polylactides as packaging materials. Macromol Biosci. 2004;4(9):835-864. doi:10.1002/mabi.200400043"
    ),

    MatrixType.CHITOSAN: Matrix(
        name="Chitosan",
        matrix_type=MatrixType.CHITOSAN,
        mechanical=MechanicalProperties(
            tensile_strength=30.0,    # Range: 20-50 MPa
            elongation_at_break=8.0,
            youngs_modulus=1200.0
        ),
        barrier=BarrierProperties(
            wvp=1.5,
            water_absorption_24h=30.0
        ),
        thermal=ThermalProperties(
            degradation_onset=250.0,
            max_degradation_temp=300.0,
            glass_transition=140.0
        ),
        density=1.40,
        gelatinization_temp=25.0,  # Room temp (acid solution)
        recommended_concentration=(1.0, 3.0),
        cost_per_kg=25.00,
        source="Elsabee MZ, Abdou ES. Chitosan based edible films and coatings: A review. Mater Sci Eng C. 2013;33(4):1819-1841. doi:10.1016/j.msec.2013.01.010"
    ),

    MatrixType.GELATIN: Matrix(
        name="Gelatin",
        matrix_type=MatrixType.GELATIN,
        mechanical=MechanicalProperties(
            tensile_strength=25.0,    # Range: 20-40 MPa
            elongation_at_break=15.0,
            youngs_modulus=800.0
        ),
        barrier=BarrierProperties(
            wvp=2.5,
            water_absorption_24h=60.0
        ),
        thermal=ThermalProperties(
            degradation_onset=200.0,
            max_degradation_temp=320.0,
            glass_transition=80.0,
            melting_point=35.0
        ),
        density=1.30,
        gelatinization_temp=40.0,
        recommended_concentration=(5.0, 15.0),
        cost_per_kg=12.00,
        source="Gómez-Guillén MC, Giménez B, López-Caballero ME, Montero MP. Functional and bioactive properties of collagen and gelatin from alternative sources: A review. Food Hydrocoll. 2011;25(8):1813-1827. doi:10.1016/j.foodhyd.2011.02.007"
    ),
}


# =============================================================================
# PLASTICIZER DATABASE - Literature Values
# =============================================================================

PLASTICIZER_DATABASE: Dict[PlasticizerType, Plasticizer] = {

    PlasticizerType.GLYCEROL: Plasticizer(
        name="Glycerol",
        plasticizer_type=PlasticizerType.GLYCEROL,
        density=1.26,
        flexibility_factor=2.0,        # Significant flexibility increase
        strength_factor=0.7,           # 30% strength reduction typical
        water_sensitivity_factor=1.5,  # Increases hydrophilicity
        recommended_concentration=(15.0, 40.0),
        cost_per_kg=1.20,
        source="Vieira MGA, da Silva MA, dos Santos LO, Beppu MM. Natural-based plasticizers and biopolymer films: A review. Eur Polym J. 2011;47(3):254-263. doi:10.1016/j.eurpolymj.2010.12.011"
    ),

    PlasticizerType.SORBITOL: Plasticizer(
        name="Sorbitol",
        plasticizer_type=PlasticizerType.SORBITOL,
        density=1.49,
        flexibility_factor=1.5,        # Moderate flexibility
        strength_factor=0.85,          # Less strength reduction
        water_sensitivity_factor=1.2,  # Lower hygroscopicity
        recommended_concentration=(15.0, 35.0),
        cost_per_kg=2.00,
        source="Müller CMO, Laurindo JB, Yamashita F. Effect of cellulose fibers addition on the mechanical properties and water vapor barrier of starch-based films. Food Hydrocoll. 2009;23(5):1328-1333. doi:10.1016/j.foodhyd.2008.09.002"
    ),

    PlasticizerType.PEG_400: Plasticizer(
        name="PEG 400",
        plasticizer_type=PlasticizerType.PEG_400,
        density=1.13,
        flexibility_factor=1.8,
        strength_factor=0.75,
        water_sensitivity_factor=1.3,
        recommended_concentration=(10.0, 30.0),
        cost_per_kg=3.50,
        source="Sanyang ML, Sapuan SM, Jawaid M, Ishak MR, Sahari J. Effect of plasticizer type and concentration on physical properties of biodegradable films based on sugar palm (Arenga pinnata) starch for food packaging. J Food Sci Technol. 2016;53(1):326-336. doi:10.1007/s13197-015-2009-7"
    ),

    PlasticizerType.CITRIC_ACID: Plasticizer(
        name="Citric Acid",
        plasticizer_type=PlasticizerType.CITRIC_ACID,
        density=1.67,
        flexibility_factor=1.3,        # Lower plasticization
        strength_factor=0.90,          # Minimal strength loss
        water_sensitivity_factor=0.9,  # Can crosslink, reduce water sensitivity
        recommended_concentration=(5.0, 20.0),
        cost_per_kg=2.50,
        source="Reddy N, Yang Y. Citric acid cross-linking of starch films. Food Chem. 2010;118(3):702-711. doi:10.1016/j.foodchem.2009.05.050"
    ),
}


# =============================================================================
# REFERENCE MATERIALS - Commercial Comparison
# =============================================================================

REFERENCE_MATERIALS: Dict[str, ReferenceMaterial] = {

    "LDPE": ReferenceMaterial(
        name="Low-Density Polyethylene",
        tensile_strength=10.0,    # Range: 8-15 MPa
        elongation=400.0,         # High elongation
        wvp=0.08,                 # Excellent barrier
        biodegradable=False,
        source="ASTM D882 typical values"
    ),

    "HDPE": ReferenceMaterial(
        name="High-Density Polyethylene",
        tensile_strength=25.0,
        elongation=150.0,
        wvp=0.04,
        biodegradable=False,
        source="ASTM D882 typical values"
    ),

    "PP": ReferenceMaterial(
        name="Polypropylene",
        tensile_strength=35.0,
        elongation=200.0,
        wvp=0.06,
        biodegradable=False,
        source="ASTM D882 typical values"
    ),

    "PLA": ReferenceMaterial(
        name="Polylactic Acid",
        tensile_strength=50.0,
        elongation=6.0,
        wvp=0.5,
        biodegradable=True,
        source="NatureWorks LLC datasheet"
    ),

    "PBAT": ReferenceMaterial(
        name="Polybutylene Adipate Terephthalate",
        tensile_strength=20.0,
        elongation=700.0,
        wvp=1.2,
        biodegradable=True,
        source="BASF Ecoflex datasheet"
    ),

    "Cellophane": ReferenceMaterial(
        name="Regenerated Cellulose",
        tensile_strength=80.0,
        elongation=20.0,
        wvp=7.0,
        biodegradable=True,
        source="Innovia Films datasheet"
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_fiber(fiber_type: FiberType) -> Fiber:
    """Get fiber data by type."""
    if fiber_type not in FIBER_DATABASE:
        raise ValueError(f"Fiber type {fiber_type} not found in database")
    return FIBER_DATABASE[fiber_type]


def get_matrix(matrix_type: MatrixType) -> Matrix:
    """Get matrix data by type."""
    if matrix_type not in MATRIX_DATABASE:
        raise ValueError(f"Matrix type {matrix_type} not found in database")
    return MATRIX_DATABASE[matrix_type]


def get_plasticizer(plasticizer_type: PlasticizerType) -> Plasticizer:
    """Get plasticizer data by type."""
    if plasticizer_type not in PLASTICIZER_DATABASE:
        raise ValueError(f"Plasticizer type {plasticizer_type} not found in database")
    return PLASTICIZER_DATABASE[plasticizer_type]


def list_available_fibers() -> List[str]:
    """Return list of available fiber names."""
    return [f"{f.name} ({f.fiber_type.value})" for f in FIBER_DATABASE.values()]


def list_available_matrices() -> List[str]:
    """Return list of available matrix names."""
    return [f"{m.name} ({m.matrix_type.value})" for m in MATRIX_DATABASE.values()]


def list_available_plasticizers() -> List[str]:
    """Return list of available plasticizer names."""
    return [f"{p.name} ({p.plasticizer_type.value})" for p in PLASTICIZER_DATABASE.values()]


def get_fiber_by_name(name: str) -> Optional[Fiber]:
    """Find fiber by partial name match (case-insensitive)."""
    name_lower = name.lower()
    for fiber in FIBER_DATABASE.values():
        if name_lower in fiber.name.lower() or name_lower in fiber.fiber_type.value:
            return fiber
    return None


def get_matrix_by_name(name: str) -> Optional[Matrix]:
    """Find matrix by partial name match (case-insensitive)."""
    name_lower = name.lower()
    for matrix in MATRIX_DATABASE.values():
        if name_lower in matrix.name.lower() or name_lower in matrix.matrix_type.value:
            return matrix
    return None


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FIBER DATABASE - PUBLIC VERSION")
    print("All data from published literature")
    print("=" * 60)

    print("\n🌿 Available Fibers:")
    for fiber in FIBER_DATABASE.values():
        print(f"  - {fiber.name}: Cellulose {fiber.composition.cellulose}%, "
              f"TS {fiber.mechanical.tensile_strength} MPa")

    print("\n📦 Available Matrices:")
    for matrix in MATRIX_DATABASE.values():
        print(f"  - {matrix.name}: TS {matrix.mechanical.tensile_strength} MPa, "
              f"Elong {matrix.mechanical.elongation_at_break}%")

    print("\n💧 Available Plasticizers:")
    for plast in PLASTICIZER_DATABASE.values():
        print(f"  - {plast.name}: Flexibility ×{plast.flexibility_factor}")

    print("\n📊 Reference Materials:")
    for ref in REFERENCE_MATERIALS.values():
        bio = "🌱" if ref.biodegradable else "🛢️"
        print(f"  {bio} {ref.name}: TS {ref.tensile_strength} MPa")


