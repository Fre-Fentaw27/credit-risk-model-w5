# Credit Risk Probability Model for Alternative Data

# Credit Scoring Business Understanding

## 1. Basel II Accord’s Impact on Model Requirements

The **Basel II Capital Accord** mandates that financial institutions maintain **risk-sensitive capital reserves** based on their credit exposure. This directly affects our model development in two key ways:

- **Transparency & Explainability**: Regulators require clear documentation of risk assessment models to ensure compliance with capital adequacy rules.
- **Risk Quantification**: The model must reliably distinguish between high-risk and low-risk borrowers to avoid under/overestimating capital requirements.

The **Basel II Capital Accord**fundamentally reshapes how financial institutions approach credit risk by introducing three key pillars:

1. **Minimum Capital Requirements**:

   - Mandates banks to hold capital proportional to their risk exposure
   - Requires rigorous quantification of credit risk probabilities
   - Our model must directly support capital adequacy calculations

2. **Supervisory Review Process**:

   - Regulators demand full transparency in risk assessment methodologies
   - Model documentation must explain all assumptions and decision points
   - Requires clear audit trails for all risk classifications

3. **Market Discipline**:
   - Stakeholders need to understand risk assessment logic
   - Model outputs must be explainable to both technical and non-technical audiences

**Implementation Implications**:

- We must prioritize model interpretability over pure predictive power
- All feature engineering decisions require business justification
- Documentation must cover the entire model lifecycle from development to deployment

**Why does this matter for our project?**  
Since Bati Bank must comply with Basel II, our credit risk model must:  
✔ Be **interpretable** (decision logic should be auditable).  
✔ Have **well-documented assumptions** (e.g., how alternative data maps to default risk).  
✔ Avoid **black-box approaches** unless rigorously validated.

---

## 2. The Need for a Proxy Variable & Associated Risks

### **Why a Proxy?**

The eCommerce dataset lacks a direct **"default" label** (unlike traditional banking data). Thus, we must engineer a **proxy variable** using:

- **RFM (Recency, Frequency, Monetary) patterns** – e.g., late payments, order cancellations, high refund rates.
- **Behavioral signals** – e.g., frequent cart abandonment, irregular purchase cycles.
  Our eCommerce context lacks these, requiring innovative alternatives:

**Proxy Construction Approach**:

1. **Behavioral Indicators**:

   - Purchase frequency patterns
   - Cart abandonment rates
   - Return/refund behaviors

2. **Financial Signals**:

   - Order value consistency
   - Payment method trends
   - Discount sensitivity

3. **Engagement Metrics**:
   - Account tenure
   - Customer service interactions
   - Review patterns

### **Business Risks of Using a Proxy**

⚠ **Misaligned Risk Prediction** – If the proxy poorly correlates with actual credit risk, the model may:

- Approve risky customers → Higher defaults.
- Reject creditworthy users → Lost revenue.

⚠ **Regulatory Challenges** – If auditors question the proxy’s validity, the bank could face compliance penalties.

**Mitigation Strategy:**

- Validate the proxy against historical bad debt trends (if available).
- Use **domain expert judgment** to refine risk thresholds.

---

## 3. Model Selection: Simplicity vs. Performance

### **Trade-offs in a Regulated Financial Context**

| **Model Type**                               | **Advantages**                                                                              | **Disadvantages**                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Logistic Regression (Weight of Evidence)** | ✅ Easily interpretable<br>✅ Regulator-friendly<br>✅ Handles categorical data well        | ❌ Limited to linear relationships<br>❌ May underfit complex patterns            |
| **Gradient Boosting (XGBoost, LightGBM)**    | ✅ Higher accuracy<br>✅ Captures non-linear trends<br>✅ Handles feature interactions well | ❌ Harder to explain<br>❌ Requires feature engineering<br>❌ Risk of overfitting |

### **Recommended Approach for Bati Bank**

1. **Start simple** – Use **Logistic Regression with WoE binning** for initial compliance.
2. **Experiment cautiously** – Test Gradient Boosting only if:
   - Performance gains justify complexity.
   - SHAP/LIME can provide post-hoc explainability.
3. **Document rigorously** – Maintain clear records of model logic for auditors.

### Implementation Roadmap

1. **Data Preparation Phase**:

   - Proxy variable definition
   - Feature engineering
   - Segmentation analysis

2. **Model Development**:

   - Baseline model creation
   - Advanced model experimentation
   - Ensemble approaches

3. **Validation Framework**:

   - Statistical validation
   - Business validation
   - Regulatory compliance checks

4. **Deployment Strategy**:
   - A/B testing framework
   - Monitoring dashboard
   - Feedback mechanisms

---

### **Next Steps**

- [ ] Define the **exact proxy variable** (e.g., "90-day delinquency equivalent").
- [ ] Conduct **feature analysis** on eCommerce data.
- [ ] Build and validate both model types.
