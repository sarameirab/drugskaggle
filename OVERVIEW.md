
Overview
💊 Deep Pharma Challenge 2026: Drug-Drug Interaction Prediction

Can your model save lives by predicting dangerous drug combinations before they are prescribed?

Context Modern medicine relies on polypharmacy—prescribing multiple drugs to treat complex conditions. However, this comes with a hidden cost: Drug-Drug Interactions (DDIs).

30% of adverse drug events are caused by DDIs.

$30 Billion+ is lost annually in healthcare costs due to preventable interactions.

The Gap: Clinical trials cannot test every possible pair of drugs. We rely on post-market surveillance (Pharmacovigilance) to find these risks, often too late.

The Mission Your goal is to build a Multi-Task Machine Learning Model that predicts the safety profile of drug pairs. You must predict not just if an interaction occurs, but how severe it is, what specific side effects will happen, and how likely they are compared to the general population.

The Tasks This is a triple-prediction challenge. For every Drug Pair (A + B), you must output:

Severity Classification: Is the interaction Minor, Moderate, Major, or a Contraindication?

Side Effect Prediction (Binary): Which of the 50 specific adverse events (e.g., Nausea, Arrhythmia) will occur?

Risk Quantification (Regression): What is the Proportional Reporting Ratio (PRR) for each side effect? (A measure of statistical alerting used by the FDA).

The Data You are provided with a rich, multimodal dataset derived from real FDA Adverse Event Reporting System (FAERS) data:

Chemistry: SMILES strings representing the exact molecular structure of each drug.

Pharmacology: Text descriptions of Mechanisms of Action (MoA), Metabolism, and Transporters.

Scale:

Training Set: ~15,000 confirmed interacting pairs.

Test Set: ~4,000 held-out pairs (Cold Start / New Scaffolds).

Why Join?

Real-World Impact: Your code could be the foundation for next-gen Clinical Decision Support Systems (CDSS).

Scientific Discovery: Help uncover hidden patterns between molecular structure and biological toxicity.

Start

5 days ago
Close
a year to go
Evaluation
📊 Evaluation Metric
This competition uses a Hardcore Clinical Score designed to rigorously test your model's ability to detect rare signals. It penalizes "lazy" predictions (predicting nothing) and rewards precision in risk estimation.

The final score ranges from 0.0 to 1.0, where higher is better.

🏆 Final Score Formula



Where:







Detailed Breakdown
1. Severity Classification (40%)
Metric: Macro F1-Score.
Description: Multi-class classification of the clinical severity level (Minor, Moderate, Major).
Why: We treat all severity levels as equally important. Detecting a rare "Contraindication" is just as critical as detecting a common "Moderate" case. The Macro average ensures the minority classes are not drowned out.
2. Side Effect Prediction (30%)
Metric: Micro F1-Score (Global).
Description: Binary multi-label classification for the 50 side effects.
Why Micro F1?
Unlike Jaccard or Accuracy, Micro F1 does not reward predicting "Nothing".
You only gain points by actively detecting real side effects (True Positives).
3. PRR Regression (30%)
Metric: Inverse RMSE on Masked Data.

Formula:

Description: We calculate the Root Mean Squared Error (RMSE) between your predicted PRR and the true PRR.

The "Masked" Rule: The error is calculated ONLY for pairs where a side effect actually exists in the ground truth (True PRR > 0). You are not penalized for predicting PRR values on empty rows, nor are you rewarded for predicting 0s correctly.

Why Inverse RMSE?

It is a "Hardcore" metric. Large errors are penalized heavily by the square term in RMSE.

A perfect model gets 1.0. A model with high error will see its score drop rapidly towards 0.

📝 Submission Format
You must submit a CSV file containing a header and the predictions for all test rows. The file format should look like this:

Pair_ID,Severity,Target_Binary_Abdominal_pain,Target_Binary_Anaemia,...,Target_PRR_Abdominal_pain,...
Test_1,Moderate,1,0,...,2.5,...
Test_2,Major,0,1,...,1.8,...
Test_3,Minor,0,0,...,0.0,...
Column Definitions
Pair_ID: Must match the Test Set IDs exactly.
Severity: String value (Major, Moderate, Minor, Contraindication).
Target_Binary_...: Integer (0 or 1).
Target_PRR_...: Float value (typically 0.0 to 10.0).