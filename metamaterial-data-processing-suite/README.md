# Mechanical Testing Data Processing Suite

## Overview
This project is an automated pipeline for processing mechanical testing data and extracting material properties from raw experimental datasets. It is specifically designed for analyzing 3D-printed components.

Test types supported:
- Tension  (ISO 527)
- Compression (ASTM D695) 
- Shear (ASTM D5379)  
- 4-Point Bending (ASTM D6272)  

The system converts raw machine data into stress–strain curves, extracts mechanical properties, and performs statistical validation.

---

## Features

### 🔹 Data Processing
- Reads `.xlsx` files from testing machines  
- Cleans noisy data and removes invalid regions  
- Handles multiple displacement measurement sources  

### 🔹 Signal Stitching
Automatically selects the most reliable displacement signal:
1. Extensometer  
2. LVDT  
3. Encoder 
4. Strain gauge (for shear only) 

### 🔹 Mechanical Properties
- Elastic Modulus (sliding window method)  
- Yield Strength (0.2% offset method)  
- Ultimate Strength (maximum stress)
- Fracture Strenght (for tension only)  

### 🔹 Statistical Analysis
- Mean, Standard Deviation, Coefficient of Variation (CV)  
- Subgroup-based validation  
- Flags unreliable datasets (CV > 10%)  

---
## File Naming Convention
All sample files follow a standardized naming convention to encode the material, orientation, test type, infill percentage, and sample number. The general format is:

`<Material><Orientation><TestType><Infill> – <Sample Number>`

- Material: P = PETG, N = Nylon  
- Orientation: 0 = short axis aligned, 1 = long axis aligned, 2 = flipped  
- TestType: TN = Tension, BE = Bending, CP = Compression, SH = Shear  
- Infill: Percentage of infill (e.g., 100)  
- Sample Number: Index of the specimen  

### Example

P1TN100 – 3

- PETG, long axis aligned, tension test, 100% infill, sample #3
---

## How It Works

### 1. Pre-Test Measurements 
Before testing, all samples must have their physical dimensions measured. These measurements are recorded in the Excel file located in the `resources` folder.

The sample name must also be entered exactly as defined by the file naming convention, and must match the corresponding "Sample Name" in the master Excel file. This ensures that each dataset is correctly linked to its corresponding physical dimensions.

For accuracy, each dimension is measured three times and averaged.

- **Tension:** Width, Depth, and Length (3 measurements each)  
- **Compression:** Width, Depth, and Length (3 measurements each)  
- **Bending:** Width and Depth (3 measurements each)  
- **Shear:** Width and Length (3 measurements each)  

These measurements are used to compute the cross-sectional area and other required geometric properties for subsequent analysis. 

#### Example of Master Dimensions File

<style>
  table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 90%;
    text-align: center;
  }

  th, td {
    border: 1px solid #ccc;
    padding: 2px;
  }
</style>

<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>J</th>
      <th>K</th>
      <th>L</th>
    </tr>
    <tr>
      <td>Sample Name</td>
      <td>W1</td>
      <td>W2</td>
      <td>W3</td>
      <td>D1</td>
      <td>D2</td>
      <td>D3</td>
      <td>Avg W</td>
      <td>Avg D</td>
      <td>Length</td>
      <td>Area (W × D)</td>
      <td>P/F</td>
    </tr>
    <tr>
      <td>P1TN100 - 1</td>
      <td>12.0</td>
      <td>11.9</td>
      <td>12.1</td>
      <td>12.2</td>
      <td>12.1</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>12.1</td>
      <td>120</td>
      <td>145.2</td>
      <td>P</td>
    </tr>
  </table>
</div>

- **P/F:** Indicates whether the sample is valid (`P`) or failed (`F`). Failed samples are excluded from analysis.  
- **Avg W / Avg D:** Average of the three width and depth measurements.  
- **Length:** Gauge length of the specimen.  
- **Area:** Cross-sectional area calculated as:

Area = (Average Width) × (Average Depth)

The full template can be found here:  
[Download Master Dimensions File](resources/PETG_sample_dimensions.xlsx)
 
---

### 2. Post Testing

After completing a test, the raw data file must be saved using the naming convention defined earlier.

Each sample must also be recorded in the **master dimensions file**, where its validity is specified in the **P/F** column:
- `P` → Valid test (used in analysis)
- `F` → Invalid test (excluded due to testing issues or incorrect failure mode)

⚠️ Limitation:

The script only accepts raw data files that meet BOTH of the following conditions:

#### File Format Requirement
The raw data file MUST:
- Be an `.xlsx` file
- Contain the exact column headers shown below (case-sensitive, no extra spaces) 

<style>
  table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 100%;
    text-align: center;
  }

  th, td {
    border: 1px solid #ccc;
    padding: 5px;
  }
</style>

<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
    <tr>
      <td>Encoder [mm]</td>
      <td>LVDT [mm]</td>
      <td>Extensometer [mm]</td>
      <td>Load [kN]</td>
      <td>Lateral Strain [µStrain]</td>
      <td>Axial Strain [µStrain]</td>
    </tr>
  </table>
</div>

Any deviation (missing columns, renamed headers, different file type) will cause the script to fail or ignore the file.

Once validated, the raw data file must be placed in the appropriate folder based on:
- Material type (e.g., PETG, NYLON)
- Test type (e.g., TENSION, COMPRESSION, SHEAR, 4PT_BENDING) 

 
#### Data Folder Structure
```bash
DATA/
├── NYLON/
│   ├── NYLON_TENSION/
│   │   └── N1TN100 - 1.xlsx
│   ├── NYLON_COMPRESSION/
│   │   └── N1CP100 - 1.xlsx
│   └── NYLON_SHEAR/
│       └── N1SH100 - 1.xlsx
│
└── PETG/
    ├── PETG_TENSION/
    │   └── P1TN100 - 1.xlsx
    ├── PETG_COMPRESSION/
    │   └── P1CP100 - 1.xlsx
    ├── PETG_SHEAR/
    │   └── P1SH100 - 1.xlsx
    └── PETG_4PT_BENDING/
        └── P1BE100 - 1.xlsx 

```


#### Data Workflow

1. Perform experiment → Generate raw data file  
2. Name file according to naming convention  
3. Record sample in master dimensions file  
4. Assign validity (P/F)  
5. Place file in correct folder:

   DATA / Material / Test Type / Raw File


### 3. Test Results and anlysis 
The results generated from each experiment are automatically processed and stored in structured output folders. These outputs include both graphical visualizations and statistical summaries.

#### Graphical Results
- All processed stress–strain curves are saved as interactive HTML files.
- These graphs allow visualization of:
  - Linear region detection
  - Offset yield point
  - Ultimate strength


- Graph files are located in:

```bash
GRAPHS/
├── NYLON/
│   ├── TENSION/
│   ├── COMPRESSION/
│   └── SHEAR/
└── PETG/
    ├── TENSION/
    ├── COMPRESSION/
    ├── SHEAR/
    └── 4PT_BENDING/
```

Each file corresponds to an individual test sample and can be opened in a web browser.

#### Statistical Results
- Summary statistics for each test type are compiled into Excel files.
- These include:
- Young’s Modulus (E)
- Yield Strength (0.2% offset)
- Ultimate Strength
- Coefficient of Variation (CV)
- Results are aggregated per material and test type.

Statistical output location:

```bash
STATS/
├── NYLON/
│   └── RESULTS.xlsx
└── PETG/
    └── RESULTS.xlsx
```

#### Analysis Overview
- Raw data is filtered to remove:
- Initial machine settling (non-physical ramp)
- Post-failure instability
- Linear regions are automatically detected using regression criteria.
- Yield points are calculated using the 0.2% offset method.
- All results are validated before being included in the final statistics.

#### Notes
Each graph is linked directly in the results Excel file for quick access.
Ensure that input data follows the required format to guarantee accurate processing.

---


## Project Structure

```bash
project/
├── DATA/                # Raw experimental data (`.xlsx`)
├── GRAPHS/              # Generated plots (stress–strain curves)
├── STATS/               # The physical properties of subgroups of tests and the statisical analysys of them 
├── docs/                # Documentation of the code 
├── resources/           # Physical dimentions needed for calculation
│
├── tension_core/        # Tension analysis pipeline
├── compression_core/    # Compression analysis pipeline
├── shear_core/          # Shear analysis pipeline
├── bending_core/        # 4-point bending analysis
│
├── petg_tension_core/   # Specialized PETG analysis
├── shared_core/         # Shared utilities by all 4 test (fitting, stitching, linear region)
├── reporting_core/      # Statistical analysis and reporting
├── app_core/            # High-level orchestration logic
│
├── tests/               # Testing scripts
├── verification/        # Validation tools and debugging
│
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
└── README.md
```
---



### Installation


1. Clone the repository:
```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
```

2. Install dependencies:
```bash

    pip install -r requirements.txt
```

3. Run the main processing script:
```bash 

 python main.py
```