# RiskCost — Cost-Sensitive Machine Learning for Financial Risk Decisions

> A machine learning project exploring how financial risk decisions can be improved when different types of classification errors have different business consequences.

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Project Objective](#project-objective)
- [Why This Problem Matters](#why-this-problem-matters)
- [Dataset](#dataset)
- [Approach](#approach)
- [Business Cost Framework](#business-cost-framework)
- [Machine Learning Models](#machine-learning-models)
- [Adaptive Cost-Sensitive Loss](#adaptive-cost-sensitive-loss)
- [Decision Threshold Optimization](#decision-threshold-optimization)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Understanding the Results](#understanding-the-results)
- [Business Impact Example](#business-impact-example)
- [Financial Data Science Relevance](#financial-data-science-relevance)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Experiments](#experiments)
- [Evaluation Strategy](#evaluation-strategy)
- [Key Learnings](#key-learnings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Takeaway](#project-takeaway)
- [Disclaimer](#disclaimer)

---

# Overview

Financial systems make decisions using large volumes of transaction and behavioral data.

A machine learning model can help identify risky or fraudulent transactions, but simply predicting whether something is fraudulent is not the complete problem.

The system also needs to answer:

> **What should happen when the model is wrong?**

For example, missing a fraudulent transaction may create financial loss, while incorrectly flagging a legitimate transaction may create customer inconvenience or additional manual-review work.

These two mistakes are not necessarily equally expensive.

**RiskCost** explores this problem using credit-card fraud detection as a practical case study.

The project combines:

- Imbalanced classification
- Machine learning model comparison
- Cost-sensitive learning
- Custom loss functions
- Decision-threshold optimization
- Validation-based model selection
- Business-cost evaluation

The central idea is:

> **Machine learning decisions should be evaluated not only by statistical performance, but also by the potential business impact of prediction errors.**

---

# Business Problem

Fraud detection is a highly imbalanced classification problem.

Most transactions are legitimate, while only a very small percentage are fraudulent.

This creates two important types of mistakes.

### False Negative

The model predicts:

```text
Legitimate

but the transaction is actually:

Fraud

This means the system failed to detect fraud.

Potential consequences can include:

Financial loss
Chargeback costs
Investigation costs
Customer impact
Increased operational risk
False Positive

The model predicts:

Fraud

but the transaction is actually:

Legitimate

Potential consequences can include:

Customer inconvenience
Additional verification
Manual review
Transaction delays
Unnecessary operational workload
Why the Difference Matters

Both predictions are incorrect, but their consequences can be different.

A traditional ML evaluation may count both as simply:

Incorrect prediction

A financial decision system may instead need to think:

How expensive is this mistake?

That difference is the main motivation behind this project.

Project Objective

The objective of RiskCost is to investigate how machine learning models behave when the cost of different classification errors is explicitly considered.

The project focuses on four questions:

1. Can machine learning effectively detect rare fraudulent transactions?
2. What happens when false negatives are considered more expensive than false positives?
3. Does changing the model's training objective provide additional value compared with changing the decision threshold?
4. How can ML results be translated into business-oriented decision metrics?
  Why This Problem Matters

Consider a financial platform processing thousands or millions of transactions.

A model might produce:

Transaction A → Fraud probability: 0.82
Transaction B → Fraud probability: 0.18
Transaction C → Fraud probability: 0.51

The model provides probabilities.

But the business still needs to decide:

Approve?
Review?
Block?
Request additional verification?

That decision depends on more than the probability alone.

It can depend on:

Cost of missing fraud
Cost of reviewing legitimate transactions
Customer experience
Operational capacity
Risk tolerance
Business policies

Therefore, a practical ML system can be viewed as:

Data
  ↓
Prediction
  ↓
Risk Assessment
  ↓
Business Cost
  ↓
Decision
  ↓
Operational Action

RiskCost focuses on this connection between machine learning predictions and business decisions.

Dataset

The project uses the public Credit Card Fraud Detection dataset.

The target variable is:

Class = 0 → Legitimate transaction
Class = 1 → Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing only a very small portion of the total observations.

This makes it useful for studying:

Rare-event classification
Fraud detection
Class imbalance
False-negative reduction
Precision-recall trade-offs
Cost-sensitive decision making
Why Accuracy Is Not Enough

In an extremely imbalanced dataset, accuracy can give a misleading picture.

For example, imagine:

100,000 transactions

99,800 legitimate
200 fraudulent

A model that predicts every transaction as legitimate would achieve approximately:

99.8% accuracy

However:

Fraud detected = 0

The model would completely fail at its main purpose.

Therefore, this project focuses on metrics that provide more useful information for rare-event detection.

Evaluation Metrics

The project evaluates models using both traditional ML metrics and business-oriented metrics.

Precision

Precision answers:

Of the transactions predicted as fraud, how many were actually fraud?

Precision =
True Positives /
(True Positives + False Positives)

Higher precision means fewer legitimate transactions are incorrectly flagged.

Recall

Recall answers:

Of all actual fraudulent transactions, how many did the model detect?

Recall =
True Positives /
(True Positives + False Negatives)

Recall is particularly important when missing a fraudulent transaction can be expensive.

F1 Score

F1-score balances precision and recall.

F1 =
2 × Precision × Recall /
(Precision + Recall)
ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

## Pr-Auc

Precision-Recall AUC is particularly useful for highly imbalanced classification because it focuses on the model's ability to identify the minority class while considering precision.

False Positives

Number of legitimate transactions incorrectly classified as fraud.

False Negatives

Number of fraudulent transactions incorrectly classified as legitimate.

Business Cost

The project also calculates an explicit decision cost based on false positives and false negatives.

This allows model performance to be viewed from a business perspective rather than only from a statistical perspective.

Business Cost Framework

For the main experiment, the project uses the following cost assumption:

False Negative Cost = 10
False Positive Cost = 1

Therefore:

Total Business Cost
=
10 × False Negatives
+
1 × False Positives

This means that, within this experiment:

A false negative is considered 10 times more costly than a false positive.

Why Use a Cost Ratio?

Suppose two models produce:

Model A
False Negatives = 20
False Positives = 20

Business cost:

(20 × 10) + (20 × 1)
= 220
Model B
False Negatives = 12
False Positives = 60

Business cost:

(12 × 10) + (60 × 1)
= 180

Model B creates more false positives.

However, under the project's assumed cost structure, it produces a lower total decision cost because it misses fewer fraudulent transactions.

This illustrates an important concept:

The model with the best conventional metric is not necessarily the model with the lowest business cost.

The 10:1 ratio is a project-defined assumption for experimentation. It is not intended to represent the actual cost structure of a bank or financial institution.

Machine Learning Models

The project evaluates multiple approaches to understand how different strategies behave under extreme class imbalance.

The main approaches include:

Logistic Regression
Random Forest
Cost-weighted learning
Oversampling approaches
PyTorch MLP
Standard Binary Cross Entropy
Class-weighted Binary Cross Entropy
Adaptive Cost-Sensitive Loss

The purpose of using multiple approaches is not simply to find the most complicated model.

Instead, the project asks:

Does a specialized cost-sensitive approach provide meaningful additional value compared with strong baseline approaches?

Adaptive Cost-Sensitive Loss

The main methodological component of the project is a custom Adaptive Cost-Sensitive Loss implemented using PyTorch.

The approach combines two ideas.

1. Business Cost

Fraud examples receive a higher penalty when they are incorrectly classified.

This reflects the assumption that missing fraud is more costly than incorrectly flagging a legitimate transaction.

2. Sample Difficulty

Not every transaction is equally easy for the model to classify.

For example:

Transaction A

Strongly unusual behavior
Large unusual amount
Rare transaction pattern

→ Easier to identify

while:

Transaction B

Looks very similar to legitimate transactions Only subtle differences

→ More difficult to classify

The adaptive loss tracks the historical difficulty of individual training samples.

The simplified idea is:

Sample Importance
=
Business Cost
×
Sample Difficulty

This allows difficult and costly examples to receive more attention during training.

Sample Difficulty Tracking

The project maintains a difficulty estimate for each training sample.

The estimate is updated using an exponential moving average.

Conceptually:

New Difficulty
=
0.9 × Previous Difficulty
+
0.1 × Current Loss

This means the model does not react only to one difficult training step.

Instead, it retains information about whether a sample has been consistently difficult over time.

The goal is to distinguish between:

Temporary difficulty

and:

Persistent difficulty
Decision Threshold Optimization

One of the most important parts of the project is the distinction between:

Model Training

and:

Decision Making

A classification model does not necessarily output:

Fraud

directly.

It often outputs a probability.

For example:

Fraud Probability = 0.73

A common default threshold is:

Probability >= 0.50
→ Fraud

But there is no general rule that says 0.50 must be the best business threshold.

A financial application may prefer a different threshold depending on the consequences of false positives and false negatives.

Example of Threshold Selection

Suppose a model produces:

Transaction A → 0.08
Transaction B → 0.21
Transaction C → 0.62
Transaction D → 0.91

At a threshold of:

0.50

the system would classify:

A → Legitimate
B → Legitimate
C → Fraud
D → Fraud

At a lower threshold such as:

0.20

the system would classify:

A → Legitimate
B → Fraud
C → Fraud
D → Fraud

The second strategy may detect more fraud but may also flag more legitimate transactions.

Therefore:

Threshold
   ↓
False Positives
   ↓
False Negatives
   ↓
Business Cost

Threshold selection is therefore part of the decision system.

Experimental Design

A separate experiment was created to test an important question:

Does adaptive cost-sensitive training provide additional value beyond standard model training followed by threshold optimization?

The experiment compares:

Random Forest
MLP + standard BCE
MLP + class-weighted BCE
MLP + Adaptive Cost-Sensitive Loss
Data Split

The original dataset is divided into training and test data.

The training data is then further divided into:

Training
Validation

The final test set remains untouched.

The workflow is:

Original Dataset
       ↓
Train / Test Split
       ↓
Training Data
       ↓
Train / Validation Split

Threshold selection is performed using the validation set.

Final performance is evaluated using the untouched test set.

This prevents the test set from being used to optimize the model's decision threshold.

Validation-Based Threshold Selection

For every model:

Training Data
      ↓
Model Training
      ↓
Validation Predictions
      ↓
Search Different Thresholds
      ↓
Select Lowest Validation Business Cost
      ↓
Apply Selected Threshold to Test Data
      ↓
Final Evaluation

This is important because selecting the threshold directly on the test set would make the final evaluation less trustworthy.

Experimental Results

The threshold optimization experiment produced the following final test-set results.

Model	Selected Threshold	Precision	Recall	F1	PR-AUC	False Positives	False Negatives	Business Cost Random Forest	0.12	0.6744	0.8878	0.7665	0.8570	42	11	152 MLP + BCE	0.04	0.6222	0.8571	0.7210	0.7356	51	14	191 MLP + Weighted BCE	0.98	0.6800	0.8673	0.7623	0.7191	40	13	170 MLP + Adaptive Loss	0.55	0.5513	0.8776	0.6772	0.8015	70	12	190 Understanding the Results

The main takeaway is not simply which model has the lowest number.

The experiment demonstrates several important ideas.

1. Threshold selection matters

The Random Forest model selected a threshold of:

0.12

rather than the conventional:

0.50

This produced:

Recall = 88.78%
False Positives = 42
False Negatives = 11
Business Cost = 152

The result demonstrates how much the operating point of a model can change when the business objective is explicitly considered.

2. More complex training does not automatically mean better business performance

The adaptive loss is more specialized than the baseline approaches.

However, in this particular experiment, the adaptive model did not produce the lowest business cost.

Its result was:

False Positives = 70
False Negatives = 12
Business Cost = 190

This is important because the project does not assume that a custom method must outperform simpler alternatives.

Instead, it evaluates the method experimentally.

3. Strong baselines matter

The Random Forest baseline performed strongly after threshold optimization.

This demonstrates why a new machine learning technique should always be compared against strong baseline approaches.

A more complicated model is only useful if its additional complexity produces meaningful improvement.

4. Model training and decision optimization are different problems

The experiment showed that improving the final decision does not necessarily require changing the underlying model.

Sometimes:

Better threshold selection

can produce a substantial improvement in business performance.

This is an important practical lesson for financial ML systems.

Business Impact Example

Consider a financial platform monitoring transactions.

The ML model produces:

Fraud Probability = 0.18

The system now has several choices.

Option 1 — Approve

Potential benefit:

No customer friction

Potential risk:

Fraud could be missed
Option 2 — Review

Potential benefit:

Additional verification

Potential cost:

Operational workload
Customer friction
Option 3 — Block

Potential benefit:

Potential fraud prevented

Potential cost:

Legitimate customers may be incorrectly blocked

A business-aware decision system therefore needs to consider:

Model Probability
       +
Error Cost
       +
Operational Constraints
       ↓
Final Decision

The project demonstrates this concept using a simplified fraud-detection setting.

Financial Data Science Relevance

The methodology explored in this project can apply to several financial data-science problems.

Fraud Detection

Identify suspicious transactions while balancing:

Fraud Loss
vs.
Customer Friction
Credit Risk

Predict customers who may default while balancing:

Credit Loss
vs.
Rejected / Restricted Legitimate Customers
Payment Risk

Identify risky payments while considering:

Fraudulent Payments
vs.
Declined Legitimate Payments
Account Security

Identify suspicious account activity while balancing:

Security Risk
vs.
Additional Authentication
Financial Anomaly Detection

Prioritize unusual events while balancing:

Risk Investigation
vs.
Investigation Workload

The specific cost structure would need to be determined from real business data for each application.

End-to-End Workflow

The complete project can be summarized as:

                Transaction Data
                       │
                       ▼
               Data Preparation
                       │
                       ▼
             Class Imbalance Analysis
                       │
                       ▼
              Baseline ML Models
                       │
                       ▼
          Cost-Sensitive Experiments
                       │
                       ▼
             Model Probability
                       │
                       ▼
          Validation Threshold Search
                       │
                       ▼
             Final Test Evaluation
                       │
                       ▼
       ┌─────────────────────────────┐
       │                             │
       ▼                             ▼
 ML Performance              Business Cost
       │                             │
       └──────────────┬──────────────┘
                      ▼
             Decision Analysis
Project Architecture
                         Credit Card Dataset
                                  │
                                  ▼
                         Data Loading Layer
                                  │
                                  ▼
                         Train/Test Split
                                  │
                                  ▼
Train/Validation Split │
┌─────────────┴─────────────┐ │                           │ ▼                           ▼ Baseline Models            Adaptive Model │                           │ │                   Business Cost │                           + │                   Sample Difficulty │                           │ └─────────────┬─────────────┘ │ ▼ Probability Scores │ ▼ Validation Threshold │ ▼ Final Test Set │ ▼ ML + Business Evaluation │ ▼ Decision Analysis Repository Structure adaptive-cost-sensitive-loss/ │ ├── data/ │   └── raw/ │       └── creditcard.csv │ ├── experiments/ │   └── final_model_summary.csv │ ├── notebooks/ │   ├── 01_data_overview.ipynb │   ├── 02_baseline_models.ipynb │   ├── 03_torch_baseline.ipynb │   ├── 04_results_summary.ipynb │   └── 05_threshold_vs_loss_experiment.ipynb │ ├── src/ │   ├── __init__.py │   │ │   ├── data/ │   │   └── load_data.py │   │ │   ├── losses/ │   │   └── adaptive_cost_sensitive.py │   │ │   └── models/ │       ├── __init__.py │       └── mlp.py │ ├── requirements.txt ├── README.md └── LICENSE Technology Stack Programming Python Data Processing Pandas NumPy Machine Learning Scikit-learn PyTorch Visualization Matplotlib Development Jupyter Notebook VS Code Git GitHub Installation
1. Clone the Repository
git clone [github.com/tausif1320/adaptive-cost-sensitive-loss.git](https://github.com/tausif1320/adaptive-cost-sensitive-loss.git)

Move into the project directory:

cd adaptive-cost-sensitive-loss
2. Create a Virtual Environment
Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv

Activate it:

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

The project requires the main Python data-science and machine-learning libraries used throughout the notebooks.

Dataset Setup

Download the Credit Card Fraud Detection dataset and place it inside:

data/raw/

The expected file name is:

creditcard.csv

The final structure should be:

adaptive-cost-sensitive-loss/
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
└── ...

The dataset should contain the target column:

Class

where:

0 = Legitimate
1 = Fraud
Running the Project

Start Jupyter:

jupyter notebook

Then run the notebooks in order.

Notebook 1 — Data Overview
01_data_overview.ipynb

Purpose:

Load the dataset
Inspect the data
Understand class imbalance
Explore the target distribution
Notebook 2 — Baseline Models
02_baseline_models.ipynb

Purpose:

Train baseline machine learning models
Compare traditional approaches
Establish reference performance
Notebook 3 — PyTorch Baseline
03_torch_baseline.ipynb

Purpose:

Build the neural-network baseline
Train the MLP
Evaluate standard and weighted loss approaches Notebook 4 — Results Summary 04_results_summary.ipynb

Purpose:

Consolidate model results
Compare model performance
Analyze business cost
Notebook 5 — Threshold vs Loss Experiment
05_threshold_vs_loss_experiment.ipynb

Purpose:

Create a validation split
Optimize decision thresholds using validation data Compare Random Forest and MLP approaches Compare standard BCE, weighted BCE, and adaptive loss Evaluate final performance on the untouched test set

This notebook contains the additional experiment used to investigate whether adaptive training provides value beyond threshold optimization.

Experiment Design

The project intentionally keeps the original baseline experiments separate from the additional threshold experiment.

The original notebooks establish the baseline methodology.

Notebook 5 investigates a more specific question:

Does adaptive cost-sensitive training provide additional value beyond threshold optimization?

This creates a cleaner experimental structure:

Original Project
      ↓
Baseline Results
      ↓
Additional Experiment
      ↓
Threshold Optimization
      ↓
Final Comparison
Evaluation Strategy

The evaluation follows three stages.

Stage 1 — Train

Models are trained using the training data.

Stage 2 — Select Threshold

Predictions are generated on the validation set.

Different thresholds are evaluated.

The threshold with the lowest validation business cost is selected.

Stage 3 — Final Evaluation

The selected threshold is applied to the untouched test set.

The final test performance is then measured.

This prevents the test set from being used to tune the decision threshold.

Why the Test Set Remains Untouched

A model-selection experiment should not repeatedly optimize decisions using the final test set.

For example, this would be problematic:

Test Set
  ↓
Try threshold 0.50
  ↓
Try threshold 0.30
  ↓
Try threshold 0.20
  ↓
Try threshold 0.12
  ↓
Choose the cheapest

The test set would effectively become part of the optimization process.

Instead, this project uses:

Training Set
     ↓
Train Model

Validation Set
     ↓
Choose Threshold

Test Set
     ↓
Evaluate Once

This provides a cleaner estimate of performance on unseen data.

Key Learnings

1. Business metrics can be as important as ML metrics

A model should not always be optimized solely for accuracy or F1-score.

For financial applications, the consequences of errors can matter significantly.

2. Imbalanced classification requires careful evaluation

When the positive class is rare, accuracy can hide poor minority-class performance.

Precision, recall, PR-AUC, false negatives, and false positives become more informative.

3. Training and decision-making are separate problems

A model can produce good probability estimates, but the final decision still depends on the threshold used.

4. Threshold optimization can be highly valuable

The experiment showed that changing the decision threshold can materially change:

Recall
Precision
False positives
False negatives
Business cost
5. More complex models are not automatically better

The adaptive loss did not outperform every simpler baseline in the threshold-optimized experiment.

This reinforces an important Data Science principle:

Complexity should be justified by measurable improvement.

6. Strong baselines are essential

A new methodology should be compared with strong conventional approaches.

Otherwise, it is difficult to determine whether the new technique actually provides additional value.

7. Model outputs need business interpretation

A prediction is not necessarily the final product.

In a financial system:

Prediction
   ↓
Risk
   ↓
Decision
   ↓
Action

Understanding this complete chain is important when building practical ML systems.

Business Impact Perspective

The project is designed around a simple financial decision principle:

Not all mistakes have the same cost.

For example:

Missed Fraud
    ↓
Potential Financial Loss

while:

False Fraud Alert
    ↓
Customer Friction
    +
Manual Review Cost

The appropriate trade-off depends on the actual business environment.

Therefore, a production system should not simply ask:

"What gives the highest accuracy?"

It should also ask:

"What happens when the model is wrong?"
Practical Financial Use Case

A similar decision framework could be used in a financial platform.

For example:

Transaction
     ↓
Risk Model
     ↓
Risk Score
     ↓
Business Policy
     ↓
Decision

Possible outcomes:

Low Risk
→ Approve

Medium Risk
→ Additional Verification

High Risk
→ Review / Block

The exact thresholds and actions would depend on real financial data and business requirements.

Limitations

This project is an experimental machine-learning project using a public fraud-detection benchmark.

It is not a production fraud-detection system.

Cost Assumption

The 10:1 false-negative to false-positive cost ratio is a project-defined assumption.

It does not represent an actual financial institution's internal cost structure.

Dataset Limitation

The model is trained and evaluated using a public benchmark dataset.

Production financial data may have:

Different transaction distributions
Different fraud patterns
Different customer behavior
Different feature availability
Different class imbalance
Different costs
Production Considerations

A real financial system would require additional capabilities such as:

Model monitoring
Data-drift detection
Model-drift detection
Probability calibration
Explainability
Security controls
Privacy controls
Human-review workflows
Model governance
Regulatory compliance
Continuous retraining
Future Improvements

Potential future improvements include:

1. Real Business Cost Estimation

Instead of using a fixed 10:1 ratio, estimate costs from historical financial and operational data.

2. Probability Calibration

Calibrate model probabilities so that predicted probabilities better represent observed event frequencies.

3. Explainable Risk Decisions

Add explanations for why a transaction was assigned a particular risk score.

4. Model Monitoring

Track model performance after deployment.

5. Data Drift Detection

Detect changes in transaction behavior that could affect model performance.

6. Production API

Expose the trained model through an API for integration with other systems.

7. Human Review Workflow

Introduce a review stage for uncertain predictions.

For example:

Low Risk
→ Automatic Approval

Medium Risk
→ Human Review

High Risk
→ Investigation / Block
What This Project Demonstrates
Machine Learning
Imbalanced classification
Supervised learning
Random Forest
Logistic Regression
Neural Networks
Cost-sensitive learning
Model evaluation
Data Science
Data preprocessing
Train/validation/test splitting
Experimental design
Metric selection
Threshold optimization
Error analysis
Business-cost analysis
Financial Analytics
Risk-oriented classification
False-positive / false-negative trade-offs
Decision-cost modeling
Financial risk interpretation
Engineering
Python
PyTorch
Scikit-learn
Modular project structure
Jupyter experimentation
Git/GitHub
Reproducible experiments
Project Takeaway

RiskCost started with a simple question:

Can machine learning be made more useful for financial risk decisions by considering the cost of different mistakes?

The project explored this through:

Highly Imbalanced Data
        ↓
Baseline Models
        ↓
Cost-Sensitive Learning
        ↓
Adaptive Loss
        ↓
Threshold Optimization
        ↓
Business Cost Evaluation

The most important lesson was that improving a financial ML system is not necessarily about making the model more complicated.

The final decision depends on the interaction between:

Model
  +
Probability
  +
Threshold
  +
Business Cost
  +
Operational Constraints

The threshold experiment demonstrated that a strong baseline model combined with an appropriate decision threshold can be highly competitive with more specialized approaches.

This led to a broader conclusion:

The goal of financial machine learning is not simply to predict correctly. It is to support better decisions under real-world consequences.

Disclaimer

RiskCost is an independent machine-learning project developed for educational and experimental purposes using a public fraud-detection dataset.

The project does not use confidential data from any financial institution.

The business-cost assumptions used in the experiments are hypothetical and are intended to demonstrate the methodology rather than represent the actual policies, costs, or decision processes of any specific organization.

Author

Shaik Tausif Ali

B.Tech — Computer Science (AI & Data Science)

Hyderabad, India

GitHub: tausif1320

License

This project is licensed under the terms specified in the repository's LICENSE file.
