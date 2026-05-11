# Recommendation Engine: Concepts & Basics

### Question: What is the difference between Matrix Factorization and traditional statistical analysis (e.g., Association Rules)?
Traditional statistical analysis (like **Market Basket Analysis**) and **Matrix Factorization** represent two different eras and approaches to recommendations.

#### 1. Traditional Statistical Analysis (Association Rules / MBA)
This approach is essentially **counting**. It looks at the explicit co-occurrence of items in transactions.
*   **How it works:** "In 10,000 orders, Part A and Part B appeared together 500 times. Therefore, if someone buys A, suggest B."
*   **The Logic:** It relies on the "Support" (frequency) and "Confidence" (probability) of specific combinations.
*   **Limitations:**
    *   **The "Cold Start" & Sparsity:** If a new part hasn't been bought with anything yet, the system knows nothing.
    *   **Rigidity:** It only knows about direct links. It cannot infer that Part B is similar to Part C unless they are explicitly bought together.
    *   **Scalability:** As the number of parts grows, the number of possible combinations explodes, making it computationally expensive to "comb" every possible pair.

#### 2. Matrix Factorization (Latent Factor Modeling)
This approach is about **understanding hidden patterns**. Instead of just counting pairs, it maps users and items into a "latent space."
*   **How it works:** It breaks down the massive, sparse matrix of User-Item interactions into two smaller, dense matrices:
    1.  **User Latent Factors:** A vector representing a user's "tastes" (e.g., prefers premium brands, buys for heavy-duty trucks).
    2.  **Item Latent Factors:** A vector representing an item's "characteristics" (e.g., is high-end, is a consumable, is brand-specific).
*   **The "Magic" Beyond Counting:**
    *   **Generalization:** Matrix Factorization can predict that a user will like a part they've never seen, even if that part has *never* been bought with any of their previous items. It does this by seeing that the "Latent Factors" of the user match the "Latent Factors" of the item.
    *   **Similarity Discovery:** It automatically learns that two parts are "similar" because they are bought by similar *types* of users, even if those two parts never appear in the same basket.
    *   **Dimensionality Reduction:** It compresses millions of transactions into a few hundred "factors," making the recommendation calculation near-instant.

### Summary: How does Matrix Factorization help beyond simple rules?
If you only use purchase history rules, you are looking at **what happened**. With Matrix Factorization, you are modeling **why it happened**. 

| Feature | Association Rules (Counting) | Matrix Factorization (Learning) |
| :--- | :--- | :--- |
| **Discovery** | Only finds things already bought together. | Finds new, unexpected connections. |
| **Personalization** | Global "Top Picks" for anyone buying Part X. | Tailored to the specific user's total history. |
| **Sparsity** | Fails when data is limited (New parts/users). | "Fills in the blanks" using latent similarities. |
| **Complexity** | Becomes unmanageable with millions of parts. | Highly efficient via vector math (Dot Product). |

### Question: Can we assume the "Glass" is the "User" and "Sundries" are the "Items" it prefers?
**Yes, absolutely.** This is a powerful mental model and a standard implementation strategy for Item-to-Item recommendations.

In a typical recommendation engine, you have `User -> Item`. But in our case, we are modeling the **relationship between parts**. By treating the **Glass** as the "User" and the **Sundries** as the "Items," we are building a "Compatibility Model."

#### How the mapping works:
*   **The "User" (Glass):** Every unique Glass part number (e.g., `FW02345`) acts as a "User profile."
*   **The "Item" (Sundries):** The moldings, clips, and adhesives are the "products" that the Glass "prefers" to be paired with.
*   **The "Rating/Interaction":** Instead of a 1-5 star rating, we use the **frequency of co-occurrence**. If `FW02345` is bought with `Urethane-A` 500 times, the "Glass User" has a very strong "preference" for that "Sundry Item."

#### Why this is better than a simple lookup table:
1.  **Implicit Associations:** If `Glass A` and `Glass B` are both for late-model Toyotas, they will likely share similar "Latent Factors." The model will realize that even if `Glass B` is brand new and has no history, it should probably be paired with the same sundries as `Glass A`.
2.  **Handling "Niche" Glass:** For a rare classic car windshield that only sells twice a year, traditional stats might fail to find a pattern. Matrix Factorization will see that this glass shares traits with other classic windshields and "fill in the blanks" for the correct molding.
3.  **Discovery of "Universal" vs. "Specific":** The model will naturally learn that some sundries (like generic urethane) are "liked" by almost all "Glass Users" (Universal), while others (specific clips) are only "liked" by a very narrow group of "Glass Users" (Fitment-specific).

#### The Result:
When a CSR looks up a piece of glass, you aren't just looking up a static list. You are asking the model: *"Based on the latent DNA of this Glass, what Sundries have the highest compatibility score?"*

### Question: If the data is just IDs (numbers), how does the engine "match DNA" without knowing it's an F-150?
This is the most common "aha!" moment in Machine Learning. The engine doesn't need to know the name "Ford" or "F-150" to understand they are related. It discovers the "DNA" through **shared behavior.**

#### 1. The Vector (The "DNA" Strand)
When the training starts, every ID is assigned a random "vector" (a list of numbers, e.g., `[0.12, -0.45, 0.88...]`). Think of this as a blank DNA strand.

#### 2. Discovery by Association
If `Glass 447457` and `Glass 447148` both frequently appear with `Sundry 429672` (Urethane) and `Sundry 460424` (a specific clip), the model's math forces their vectors to become **mathematically similar.**
*   The engine thinks: *"I don't know what 447457 is, but it seems to 'eat' the same things as 447148."*
*   Because they "eat" the same things, they must be the same "species" (e.g., F-150 glass).

#### 3. Defining the Latent Factors
As the model trains on millions of rows, the individual slots in that vector (the "DNA" strands) start to represent real-world concepts automatically, even though we didn't label them:
*   **Dimension 1:** Might become "How much urethane does this need?"
*   **Dimension 2:** Might become "Is this a luxury car part?"
*   **Dimension 3:** Might become "Is this for a domestic or import vehicle?"

The engine discovers these categories because the **sundries themselves** act as the labels. A "luxury clip" will only be bought with "luxury glass." Therefore, any glass bought with that clip is mathematically tagged as "luxury" in its vector.

#### 4. Matching the 2025 F-150
If a new 2025 F-150 Glass ID appears:
*   Even if it only has 1 or 2 sales, those sales will likely be with the same F-150 clips used in 2024.
*   The model immediately pulls the 2025 ID's vector toward the 2024 ID's vector.
*   **The DNA is matched:** The 2025 glass now "inherits" the recommendations of the 2024 glass because they share the same latent space location.

**Summary:** The "DNA" isn't the name of the part; it's the **mathematical signature of its relationships.**

### Question: If we only pass 1/0 labels, can we capture dimensions like "How much urethane is needed"?
This is where we move from **Implicit Signals** to **Explicit Features.**

#### 1. The 1/0 Label (Implicit Signal)
When you use a 1/0 label, you are telling the model: *"This combination is possible."*
*   **What it learns:** It can infer "Is this a luxury part?" because luxury clips only pair with luxury glass. The relationship is categorical.
*   **What it misses:** It doesn't naturally know the difference between needing **1 tube** of urethane versus **3 tubes**, because both scenarios result in a "1" (Yes, they go together).

#### 2. Using "Quantity" as the Label (Weighting)
To capture "how much," we can change the **Label** from a binary 1/0 to a continuous number (e.g., the Quantity bought).
*   **The Math:** If `Glass A` has a label of `3.0` with `Urethane` and `Glass B` has a label of `1.0`, the model will try to predict a higher "interaction score" for Glass A.
*   **The Result:** The prediction becomes a "Compatibility Score" that correlates with volume.

#### 3. Adding "Features" (Factorization Machines)
If we want the engine to know *explicit* facts (like "This glass is 2000 sq inches" or "This is a front windshield"), we move from pure Matrix Factorization to **Factorization Machines (FM)**.
*   **Pure MF:** Only uses `GlassID` and `SundryID`.
*   **FM / Hybrid Models:** Allows us to pass "Side Information":
    *   `GlassID` + `SurfaceArea` + `VehicleBrand`
    *   `SundryID` + `IsConsumable` + `UnitOfMeasure`
*   **How it helps:** The model no longer has to "guess" that a part is luxury; you've told it. It can then use that fact to make better guesses about *other* luxury parts.

#### 4. The "Urethane" Dimension specifically
Even without adding "Quantity" as a feature, Matrix Factorization often captures "Volume" in a weird way. If a glass part is huge, it might appear in the data with 3 different sundries (Urethane, Dam Tape, Primer) more consistently than a small window. The "DNA" vector will end up having a higher magnitude for "Consumables," which effectively flags it as "High Volume."

**Recommendation:** For the first version, 1/0 is great for **"What goes with what."** If you want to recommend **"How many of each,"** we should transition to using `Quantity` as the label or adding `PartAttributes` as side features.

### Question: If we manually add relations (e.g., Glass A + Sundry S1), does it help or create bias?
Mixing "Expert Knowledge" with "Machine Learning" is called **Hybrid Modeling**. It is very effective but requires a careful balance.

#### 1. How it helps (The "Cold Start" Solution)
Manual relations are the perfect cure for the **Cold Start problem.**
*   **The Scenario:** You have a brand new 2026 model windshield with zero sales. The ML engine knows nothing.
*   **The Fix:** If an expert manually links it to a specific clip and molding, you provide the "initial DNA."
*   **The Result:** The engine can now place that new glass in the Latent Space immediately. It doesn't have to wait for the first 50 customers to "teach" it.

#### 2. The Risk of Bias
If you rely too heavily on manual rules, you can create several types of bias:
*   **Expert Blindness:** If your expert only knows about "Brand X" primer, they will manually link it. The model will then "learn" that only Brand X is compatible, even if "Brand Y" is actually a better fit or a better seller.
*   **Artificial Gravity:** If you add a manual link for `Glass A + Sundry S1`, you are creating "Artificial Gravity" in the latent space. It pulls the vectors together. If that link is actually wrong or outdated, the model will continue to recommend it, and customers might continue to buy it *because it was recommended*, creating a **feedback loop** where the error reinforces itself.
*   **Stale Knowledge:** Products change. A clip might be redesigned. A manual rule is static, whereas the ML model is dynamic—it sees when people *stop* buying S1 and *start* buying S2.

#### 3. Best Practices for Manual Relations
Instead of just "adding" them to the dataset, treat them as **"Synthetic Training Data"** with specific rules:
1.  **Lower Weighting:** You might count a "Real Sale" as weight `1.0` and a "Manual Rule" as weight `0.5`. This allows real-world behavior to eventually "outvote" a manual rule if they disagree.
2.  **The "Safety Net" Role:** Use manual rules as a **Global Filter** rather than a training input. Let the ML suggest anything, but use the manual rules to say "Never show a Toyota clip for a Ford glass" (The Guardrail approach).
3.  **The "Seed" Role:** Only use manual rules for the first 3 months of a product's life. Once the product has 20 real sales, archive the manual rule and let the ML take over.

**Summary:** Manual rules are like "Training Wheels." They are essential for getting started (Cold Start), but if you leave them on too long, they prevent the engine from learning the actual, nuanced behavior of your customers.

### Question: If I have 400K transactions, won't a single manual rule just get "drowned out" by the noise?
You are right that a single row in a 400,000-row dataset has almost zero impact on the **Global Model** (e.g., it won't change what the model thinks about "Urethane" in general).

However, bias happens at the **Local Level** (the specific Glass ID).

#### 1. The "Big Fish in a Small Pond" Effect
While you have 400k rows total, how many rows do you have for **Glass ID #447457**?
*   If that specific glass only sells 10 times a year, adding **1 manual rule** is actually a **10% shift** in that part's entire history.
*   The model will notice that 10% shift and pull the vector for that glass significantly toward the manual sundry.

#### 2. The "Loudness" of Synthetic Data
When people want to "force" a manual rule to work in a large dataset, they don't just add it once. They use **Oversampling**:
*   They might inject the `Glass A + Sundry S1` relationship **1,000 times** into the training set to make sure the model "hears" it over the noise of 400k rows.
*   *This* is where intentional bias is created. You are essentially shouting at the model until it ignores the real data and listens to your rule.

#### 3. The Dangerous Feedback Loop (The "Invisible" Bias)
This is why even a "small" bias is dangerous over time:
1.  **Day 1:** You add a manual rule for a specific clip. It's just 1 row in 400k.
2.  **Day 2:** The model's score for that clip goes from 0.1 to 0.5 (just enough to show up on the CSR's screen).
3.  **Month 1:** 500 CSRs see that recommendation and click "Add to Cart" because they trust the system.
4.  **Month 6:** You retrain the model. Now you have **501 rows** (1 manual + 500 real sales) for that clip.
5.  **The Result:** The bias is now "locked in" by real sales data, even if the original expert rule was actually a sub-optimal choice.

**Summary:** Machine Learning models are very sensitive to **relative frequency.** If a part is a low-volume seller, a single manual rule can completely redefine its identity in the system.

### Question: How do we evaluate if our recommendations are actually valid?
Evaluating a recommendation engine is different from evaluating a standard "Yes/No" classifier. We use three layers of evaluation: **Offline Metrics**, **Expert Review**, and **Online KPIs.**

#### 1. Offline Evaluation (The "Backtest")
Before you ever show a recommendation to a CSR, you test the model against history.
*   **The Method:** You hide 20% of your order history (the "Test Set") and ask the model: *"Based on the glass in these hidden orders, what sundries would you suggest?"*
*   **Precision @ 5:** If the model suggests 5 items, how many of those were *actually* in the original hidden order?
*   **Recall:** Out of all the sundries the customer actually bought, how many did the model successfully find?
*   **RMSE (Error Score):** How far off was the model's predicted "compatibility score" from the actual 1/0 label? (Lower is better).

#### 2. Qualitative Evaluation (The "Smell Test")
Machine learning can be "mathematically correct" but "logically stupid."
*   **The Method:** Export a report of 50 random glass parts and their "Top 5 Recommended Sundries."
*   **Expert Audit:** Have a senior parts manager look at the list.
    *   *Good:* "Yes, those clips always go with that glass."
    *   *Bad:* "It's suggesting a generic molding for a luxury windshield that requires an OEM trim."
*   **Why this happens:** The data might be biased toward cheap/generic parts because people buy them more, but they might not be the "correct" recommendation for every job.

#### 3. Online Evaluation (Production KPIs)
The ultimate test is whether the engine changes behavior in the real world.
*   **Attachment Rate:** This is your North Star. If the average glass order used to have **1.2 sundries** and now it has **1.8 sundries**, the engine is working.
*   **Click-Through Rate (CTR):** How often do CSRs actually click "Add to Cart" on a recommended item?
*   **Return Rate:** Are we recommending the *wrong* parts? If returns for "Incorrect Clip" go up, your engine has a brand-matching problem.

#### 4. The "Diversity" vs. "Accuracy" Trade-off
A "perfectly accurate" model might only recommend **Urethane** because everyone buys it. But that's not a helpful recommendation.
*   **Helpfulness:** A valid recommendation is one that reminds the user of something they **might have forgotten** (like a specific clip), not just something obvious (like glue).

**Summary:** You know your engine is valid when your **Offline Recall** is high, your **Experts** don't laugh at the output, and your **Attachment Rate** increases in the warehouse.

### Question: Is running a "Parallel Test" on tomorrow's orders a valid way to validate?
**Yes, this is actually the gold standard of validation.** It is called **Prospective Validation** (or Out-of-Time testing).

#### 1. Why it is better than a random split
When you do a standard 80/20 split on historical data, there is a risk of "Data Leakage." For example, a big promotion on a specific clip might have happened on a Tuesday, and your model might see some of those sales in the training set and "cheat" to guess the others in the test set.
*   By testing on **Tomorrow's orders**, you are ensuring that the model has absolutely no "knowledge of the future." It proves the model can generalize to new, unseen behavior.

#### 2. What this test tells you
This test measures **"Agreement with Current Behavior."**
*   If tomorrow a customer buys `Glass A` and `Sundry S1`, and your model had `S1` in its Top 5 list, you have a **"Hit."**
*   **A high Hit Rate** means your model has successfully captured the "tribal knowledge" of your customers and CSRs. It is doing what the best humans are already doing.

#### 3. What this test *doesn't* tell you
There is one limitation to this parallel test: **It cannot measure "Discovery."**
*   Tomorrow's orders are being placed **without** seeing your recommendations.
*   If your model suggests a "Better Clip" that the customer *didn't* buy (because they didn't know about it), the parallel test will mark that as a **"Miss" (Failure)**.
*   In reality, that "Miss" might have been a brilliant recommendation that would have resulted in a sale if the CSR had seen it.

#### 4. The "Hit Rate" vs. "Potential Rate"
When you run this parallel test tomorrow, look at the results in two ways:
1.  **Direct Hits:** "We predicted it, and they bought it." (Validation of Accuracy).
2.  **Interesting Misses:** "We predicted it, they *didn't* buy it, but our experts say they *should* have." (Validation of Potential Revenue).

**Conclusion:** Running this in parallel tomorrow is a fantastic way to build confidence. If your "Hit Rate" is high (e.g., >30% of bought items were in your Top 10), your engine is ready for prime time.

### Question: Why did we inject "Negative Cases" (Label 0) in a 1:10 ratio?
This is the most critical part of training a Recommendation Engine. Without negative cases, the model would be "too optimistic."

#### 1. Why we need negatives (The "Everything is Awesome" Problem)
If you only show the model successful sales (Label 1), the math will try to make *every* prediction a "1." 
*   **The Result:** The model would learn that `Any Glass + Any Sundry = Match.` 
*   **The Solution:** You must show the model examples of what **not** to do. You need to say: *"Glass A goes with Sundry S1 (1), but it does NOT go with Sundry S99 (0)."* This forces the model to learn **Discrimination**—the ability to tell the difference between a fit and a mismatch.

#### 2. Why a 1:10 Ratio?
In the real world, the "Space of Failures" is much larger than the "Space of Successes."
*   For every 1 sundry that fits a piece of glass, there are thousands that don't.
*   **Why not 1:1?** If you only use 1 negative for every 1 positive, the model isn't "punished" enough for making a mistake. It becomes lazy.
*   **Why 1:10?** By providing 10 negative examples for every 1 positive, you are creating **"Selective Pressure."** You are teaching the model to be very picky. It has to find the one "needle in the haystack" of 11 items that actually belongs.

#### 3. How did we do it? (Random Negative Sampling)
Since our database only records **Sales** (Positives), we have to "hallucinate" the negatives:
1.  **Pick a Positive:** Take a real order: `Glass 447457` + `Sundry 429672`.
2.  **Generate Negatives:** Go to the part catalog and pick 10 random sundry IDs that have **never** been bought with `Glass 447457`.
3.  **Label them 0:** Add these to the training set with a Label of `0.0`.
4.  **Repeat:** Do this for all 400,000 transactions.

#### 4. The Result: Fighting Popularity Bias
Negative sampling is how we stop "Popularity Bias."
*   **The Problem:** Generic Urethane is in almost every order. Without negatives, the model would suggest Urethane for *everything* (even a mirror that doesn't need it) because the "Positive" signal is so strong.
*   **The Fix:** By randomly pairing that Urethane with glasses it *doesn't* belong to and labeling them `0`, we teach the model: *"Urethane is great, but only when the Glass DNA matches. If the DNA doesn't match, the score must be 0."*

**Summary:** Negative sampling turns your model from a "Yes Man" into a "Discerning Expert." The 1:10 ratio ensures the model is 10x more focused on avoiding mistakes than it is on just being "happy."
