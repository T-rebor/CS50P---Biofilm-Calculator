# Biofilm Calculator
#### Video Demo: <https://youtu.be/HhIvnqbF0CI>
#### Description:

## Overview

The **Biofilm Calculator** is an educational tool for predicting the mechanical and barrier properties of biodegradable biofilm/biocomposite materials. This project was developed as the final project for CS50's Introduction to Programming with Python, demonstrating proficiency in Python programming concepts including object-oriented programming, file I/O, data structures, argument parsing, and unit testing.

## What It Does

This calculator allows users to:

1. **Calculate Properties**: Predict tensile strength, elongation, Young's modulus, and water vapor permeability for custom biofilm formulations using natural fibers (sisal, jute, hemp, coir, flax, banana) combined with biopolymer matrices (corn starch, potato starch, cassava starch, PLA, PBAT, cellulose acetate) and plasticizers (glycerol, sorbitol, citric acid, PEG-400).

2. **Optimize Formulations**: Find the best material combinations for specific application targets like food packaging, agricultural films, rigid containers, or flexible films.

3. **Compare Materials**: Benchmark predicted properties against reference materials like LDPE, PLA, and PBAT bioplastics.

4. **Generate Reports**: Create detailed text or markdown reports documenting formulation specifications and predicted performance.

## Files Description

### project.py

This is the main module containing all core functionality. It includes:

- **`Formulation` dataclass**: Represents a biofilm composition with fiber type, matrix type, plasticizer type, and their respective concentrations.

- **`CalculatedProperties` dataclass**: Stores the predicted mechanical and barrier properties including tensile strength, elongation at break, Young's modulus, water vapor permeability, biodegradation time, and bio-content percentage.

- **`TargetProperties` dataclass**: Defines optimization targets for different applications.

- **`calculate_properties(formulation)`**: The core calculation function that uses a Modified Rule of Mixtures approach to predict biofilm properties based on the formulation components. It considers fiber-matrix interactions, plasticizer effects on flexibility, and environmental factors affecting barrier properties.

- **`optimize_formulation(fiber_type, target, ...)`**: Searches through possible material combinations to find formulations that best match specified target properties. Returns ranked results with fitness scores.

- **`compare_materials(properties)`**: Compares calculated properties against reference materials (LDPE, PLA, PBAT) to contextualize the biofilm's performance.

- **`generate_report(formulation, properties, ...)`**: Creates comprehensive text or markdown reports suitable for documentation or academic purposes.

- **CLI argument parsing**: Full command-line interface with `--quick`, `--optimize`, `--list`, `--history`, and `--chart` options.

- **`main()`**: Provides both an interactive command-line interface with a menu system and CLI argument handling.

### fiber_database.py

The materials database module containing:

- **Enum classes**: `FiberType`, `MatrixType`, `PlasticizerType` for type-safe material selection.

- **Dataclasses**: `MechanicalProperties`, `ThermalProperties`, `FiberComposition`, `Fiber`, `Matrix`, `Plasticizer`, `ReferenceMaterial` for structured material data.

- **Material databases**: `FIBER_DATABASE`, `MATRIX_DATABASE`, `PLASTICIZER_DATABASE`, `REFERENCE_MATERIALS` containing properties from peer-reviewed literature (Faruk et al. 2012, Satyanarayana et al. 2009, Siqueira et al. 2010, Jawaid & Abdul Khalil 2011, Gurunathan et al. 2015).

- **Accessor functions**: `get_fiber()`, `get_matrix()`, `get_plasticizer()`, `list_available_fibers()`, etc. for querying the database.

### test_project.py

Comprehensive test suite with 40 tests organized into logical groups:

- **TestDatabaseIntegrity**: Verifies all materials have complete data and literature citations.
- **TestFormulation**: Tests formulation creation and validation.
- **TestCalculateProperties**: Validates calculation outputs for all material combinations.
- **TestOptimizeFormulation**: Ensures optimization returns appropriate results.
- **TestCompareMaterials**: Checks comparison functionality.
- **TestGenerateReport**: Verifies report generation.
- **TestEdgeCases**: Tests boundary conditions and edge cases.
- **TestMetadata**: Confirms version information.

### requirements.txt

Lists the project dependencies. The core functionality uses only Python standard library, with pytest required for testing.

## Design Decisions

### Why Natural Fibers?

The project focuses on globally-available natural fibers (sisal, jute, hemp, coir, flax, banana) rather than proprietary materials. All property values come from published peer-reviewed literature, making the project educational and reproducible.

### Calculation Methodology

The Modified Rule of Mixtures was chosen because it:
1. Has strong theoretical foundations in materials science
2. Can be implemented with reasonable accuracy using published data
3. Demonstrates programming concepts without requiring complex numerical libraries

### Interactive vs. CLI Use

The project supports both modes:
- Interactive menu for casual exploration and step-by-step guidance
- Command-line arguments for quick calculations and scripting

This design allows the code to be both user-friendly and automatable.

## How to Run

### Interactive Mode
```bash
python project.py
```

### Command Line Interface
```bash
# Show help
python project.py --help

# Quick calculation
python project.py --quick sisal starch_corn glycerol 10 25

# Optimize for food packaging
python project.py --optimize packaging --fiber sisal --chart

# Optimize for agricultural mulch
python project.py --optimize agricultural --fiber jute

# List available materials
python project.py --list fibers
python project.py --list matrices
python project.py --list applications

# View calculation history
python project.py --history
```

### Run Tests
```bash
pytest test_project.py -v
```

## CLI Options Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message |
| `--quick` | `-q` | Quick calculation (fiber matrix plasticizer fiber% plast%) |
| `--optimize` | `-o` | Optimize for application (packaging, agricultural, rigid, flexible) |
| `--fiber` | `-f` | Fiber type for optimization |
| `--list` | `-l` | List materials (fibers, matrices, plasticizers, applications) |
| `--history` | `-H` | Show calculation history |
| `--chart` | `-c` | Show ASCII comparison chart |
| `--version` | `-v` | Show version number |

## Composition Convention

The biofilm composition follows standard practice in biopolymer literature:

**Dry Basis (Fiber + Matrix = 100%)**:
- Fiber content: expressed as percentage of total dry weight
- Matrix content: calculated as `100% - fiber%`
- Example: 10% sisal + 90% corn starch

**Plasticizer (relative to matrix weight)**:
- Expressed as % w/w (weight/weight) relative to matrix
- Example: "25% glycerol" means 25 g glycerol per 100 g of matrix
- This convention is common in biopolymer literature (often called "phr" - parts per hundred resin)

This means a formulation with 10% fiber, 90% matrix, and 25% plasticizer contains:
- 10 g fiber
- 90 g matrix
- 22.5 g plasticizer (25% of 90g)
- **Total: 122.5 g**

This convention reflects how biofilms are typically prepared in laboratories.

## Educational Disclaimer

This calculator provides theoretical estimates based on published literature values. Results should be validated experimentally before industrial application. It is designed as an educational tool demonstrating programming concepts within a biotechnology context.

## Literature Sources (Vancouver Format)

### Fiber Database References

1. Faruk O, Bledzki AK, Fink HP, Sain M. Biocomposites reinforced with natural fibers: 2000-2010. Prog Polym Sci. 2012;37(11):1552-1596. doi:10.1016/j.progpolymsci.2012.04.003

2. Jawaid M, Abdul Khalil HPS. Cellulosic/synthetic fibre reinforced polymer hybrid composites: A review. Carbohydr Polym. 2011;86(1):1-18. doi:10.1016/j.carbpol.2011.04.043

3. Satyanarayana KG, Arizaga GGC, Wypych F. Biodegradable composites based on lignocellulosic fibers—An overview. Prog Polym Sci. 2009;34(9):982-1021. doi:10.1016/j.progpolymsci.2008.12.002

4. Gurunathan T, Mohanty S, Nayak SK. A review of the recent developments in biocomposites based on natural fibres and their application perspectives. Compos Part B Eng. 2015;99:293-307. doi:10.1016/j.compositesb.2015.08.005

5. Siqueira G, Bras J, Dufresne A. Cellulosic bionanocomposites: A review of preparation, properties and applications. Polymers. 2010;2(4):728-765. doi:10.3390/polym2040728

### Matrix Database References

6. Jiménez A, Fabra MJ, Talens P, Chiralt A. Edible and biodegradable starch films: A review. Food Bioprocess Technol. 2012;5(6):2058-2076. doi:10.1007/s11947-012-0835-4

7. Mali S, Grossmann MVE, García MA, Martino MN, Zaritzky NE. Barrier, mechanical and optical properties of plasticized yam starch films. Carbohydr Polym. 2004;56(2):129-135. doi:10.1016/j.carbpol.2004.01.004

8. Auras R, Harte B, Selke S. An overview of polylactides as packaging materials. Macromol Biosci. 2004;4(9):835-864. doi:10.1002/mabi.200400043

9. Elsabee MZ, Abdou ES. Chitosan based edible films and coatings: A review. Mater Sci Eng C. 2013;33(4):1819-1841. doi:10.1016/j.msec.2013.01.010

10. Gómez-Guillén MC, Giménez B, López-Caballero ME, Montero MP. Functional and bioactive properties of collagen and gelatin from alternative sources: A review. Food Hydrocoll. 2011;25(8):1813-1827. doi:10.1016/j.foodhyd.2011.02.007

### Plasticizer Database References

11. Vieira MGA, da Silva MA, dos Santos LO, Beppu MM. Natural-based plasticizers and biopolymer films: A review. Eur Polym J. 2011;47(3):254-263. doi:10.1016/j.eurpolymj.2010.12.011

12. Müller CMO, Laurindo JB, Yamashita F. Effect of cellulose fibers addition on the mechanical properties and water vapor barrier of starch-based films. Food Hydrocoll. 2009;23(5):1328-1333. doi:10.1016/j.foodhyd.2008.09.002

13. Sanyang ML, Sapuan SM, Jawaid M, Ishak MR, Sahari J. Effect of plasticizer type and concentration on physical properties of biodegradable films based on sugar palm (Arenga pinnata) starch for food packaging. J Food Sci Technol. 2016;53(1):326-336. doi:10.1007/s13197-015-2009-7

14. Reddy N, Yang Y. Citric acid cross-linking of starch films. Food Chem. 2010;118(3):702-711. doi:10.1016/j.foodchem.2009.05.050
