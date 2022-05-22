# No Meaningful Interpretation when using Regression Type Algorithms 
## --> Solution Document

---
**Problem with Regression type Algorithms**
- When using Regression Algos, the outcome in their predictive ability is bogus. 
- Solution:
  - Show a Histogram of the Distributions of NAVs
    - Obviously the NAVs are diverging considerably from small funds to extremely large funds 
  - Option to navigate this problem:
1. **Log all the fund flows and try running the regression again.** 
2. **Do not use the respective actual values, but rather split the fund flows into percentage categories:**
   - example: smallest ff == -100.000.000 && largest ff is == +1.000.000.000
     - Convert these ffs into 100 categories from smallest == 1 to largest == 100
     - Then, re-run classification type algorithms and not regression type algorithms.

---