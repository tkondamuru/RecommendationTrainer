# Matrix Factorization in Recommendation Engine

This document provides a technical deep-dive into the Matrix Factorization algorithm used in the Recommendation Trainer and explains the specific configuration options chosen for the automotive accessory engine.

---

## 1. What is Matrix Factorization?

Matrix Factorization (MF) is a collaborative filtering algorithm used to discover latent (hidden) relationships between two entities—in our case, **Glass Parts (Users)** and **Accessories (Items)**.

Imagine a massive table where:
*   **Rows** represent every unique Glass Part (e.g., Windshields).
*   **Columns** represent every unique Accessory (e.g., Urethanes, Clips).
*   **Cells** represent the "score" or likelihood of them being sold together.

Most of this table is empty (Sparse). Matrix Factorization breaks this giant, sparse table into two much smaller, dense matrices. By multiplying these two matrices back together, the algorithm can "predict" values for the empty cells, effectively suggesting accessories that have never been seen with a specific glass part before but "fit the pattern" of similar parts.

---

## 2. Why use it for Automotive Recommendations?

While traditional "If this, then that" rules work for direct fitment, Matrix Factorization excels at:
*   **Discovering Market Baskets**: It identifies that technicians who buy "Premium Windshield A" also tend to buy "Acoustic Dampening Tape B," even if there is no explicit catalog rule linking them.
*   **Generalization**: It can recommend accessories for a new glass part by comparing its sales pattern to existing, similar glass parts.
*   **Scalability**: It handles thousands of parts and millions of transactions efficiently.

---

## 3. Configuration Options Explained

The following parameters are used in [RecommendationModelTrainer.cs](file:///c:/Development/labs/marketanalysis/RecommendationTrainer/RecommendationModelTrainer.cs):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **MatrixColumnIndexColumnName** | `"G"` | Specifies the column representing the "User" (Target Glass Part). |
| **MatrixRowIndexColumnName** | `"S"` | Specifies the column representing the "Item" (Accessory). |
| **LabelColumnName** | `"Label"` | The target value (1.0 for sold together, 0.0 for not sold). |
| **NumberOfIterations** | `50` | How many times the algorithm passes over the data. Higher values improve accuracy but take longer. |
| **ApproximationRank** | `64` | The "complexity" of the latent features. A higher rank allows the model to learn more subtle patterns but risks overfitting. |
| **LearningRate** | `0.01` | The "step size" the algorithm takes during optimization. Too high and it might miss the best solution; too low and it will be slow. |
| **Alpha** | `1` | The weight given to "positive" instances (actual sales) vs. the sampled negatives. |
| **C** | `0.00001` | The regularization constant. It prevents the model from becoming too specific to the training data (overfitting). |

---

## 4. Understanding Loss Functions

A **Loss Function** is the mathematical formula the model uses to calculate how "wrong" its prediction is. The goal of training is to minimize this loss.

### SquareLossOneClass (Used Here)
In our dataset, we primarily have "Positive" signals (we know what was bought). We don't have "Negative" signals in the raw data (the data doesn't tell us what a technician *hated*).
*   **SquareLossOneClass** is specifically designed for "One-Class" problems where we only have positive feedback.
*   It assumes that if a pair isn't in our sales history, it's *likely* a zero, but treats these "zeros" with less weight than the confirmed "ones."
*   **Why use it?** It is highly effective for market basket analysis where "missing data" doesn't mean "bad fit," just "not yet observed."

### Other Loss Function Options
1.  **SquareLoss**: Standard regression loss. Best when you have actual ratings (like 1 to 5 stars). Not ideal for binary "sold/not sold" data.
2.  **WeightedSquareLoss**: Similar to SquareLoss but allows you to give more weight to specific observations (e.g., recent sales are more important than old ones).

---

## 5. Performance Tuning

If the model is suggesting too many "random" parts, we typically:
1.  **Increase ApproximationRank**: Let the model learn more distinct "categories" of vehicles and accessories.
2.  **Decrease C (Regularization)**: Allow the model to stick closer to the raw sales patterns.
3.  **Adjust Negative Sampling**: (In the Python preprocessing) Increase the ratio of 0.0 labels to 1.0 labels to force the model to be more selective.
