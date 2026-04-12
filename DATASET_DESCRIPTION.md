You will find all the data in the data directory in this workspace.

Dataset Description
📂 Dataset Description
The dataset consists of drug pairs enriched with extensive pharmacological, chemical, and clinical text data. The goal is to predict the Severity, Side Effects, and PRR Risk of the interaction between Drug A and Drug B.

File Descriptions
train.csv
The main training set containing drug pairs, their rich feature set, and ground truth labels.

Rows: 15,284 interaction pairs.
Columns: Includes Identifiers, Features (A & B), and Targets.
test.csv
The test set containing 3,974 drug pairs.

Contains all feature columns found in Train (Drug_A_Name, SMILES_A, etc.).
Does not contain the Target columns.
🧬 Feature Column Details
The dataset provides dual features for every interaction: one set for Drug A (suffix _A) and one for Drug B (suffix _B).

1. Identifiers & Chemistry
Pair_ID: Unique identifier for the interaction row.
Drug_A_Name, Drug_B_Name: Generic international non-proprietary name (INN).
SMILES_A, SMILES_B: Simplified Molecular Input Line Entry System. A text string representing the exact 2D chemical structure and topology of the molecule.
2. Pharmacology (Mechanism)
High-dimensional text fields describing how the drug works biologically.

Mechanism_A, Mechanism_B: Description of the specific molecular targets (receptors, enzymes) the drug activates or inhibits.
Pharmacodynamics_A, Pharmacodynamics_B: Describes the biochemical and physiologic effects of the drug on the body.
3. Pharmacokinetics (ADME)
Data describing how the drug moves through the body (Absorption, Distribution, Metabolism, Excretion). These fields may contain mixed text and numeric data.

Absorption_A, Absorption_B: Details on bioavailability and absorption rates.
Metabolism_A, Metabolism_B: General description of metabolic breakdown locations (e.g., Liver).
Elimination_Route_A, Elimination_Route_B: How the drug exits the body (Renal, Biliary, etc.).
Half_Life_A, Half_Life_B: The time required for the drug's concentration in the body to reduce by half.
Protein_Binding_A, Protein_Binding_B: Percentage of the drug bound to plasma proteins (affects distribution).
4. Metabolic Enzymes (Crucial for DDIs)
CYP450_Enzymes_A, CYP450_Enzymes_B: Specific Cytochrome P450 isozymes involved in the drug's metabolism (e.g., "CYP3A4, CYP2D6"). Note: Overlap in these fields is a strong predictor of metabolic interactions.
5. Clinical & Safety Profile
Indication_A, Indication_B: The approved diseases or conditions the drug treats.
Warning_A, Warning_B: FDA Black Box warnings or general precautions.
Toxicity_A, Toxicity_B: Known toxic effects, LD50 values, or overdose symptoms.
🎯 Target Variables (Train Only)
1. Severity (Classification)
Severity: The clinical impact of the interaction.
Classes: Major, Moderate, Minor.
2. Specific Side Effects (Multi-Label)
Target_Binary_[SideEffect] (50 columns):
1: The side effect occurs.
0: The side effect does not occur.
3. Risk Quantification (Regression)
Target_PRR_[SideEffect] (50 columns):
Proportional Reporting Ratio (PRR): A float value indicating the strength of the statistical signal.
Values > 0 indicate a signal.
Values == 0 indicate no signal/no side effect.
