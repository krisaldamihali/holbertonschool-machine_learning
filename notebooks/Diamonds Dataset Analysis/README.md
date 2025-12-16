# 💎 Diamonds Dataset Analysis 💎

Analysis of the **diamonds dataset** with step-by-step data exploration, cleaning, transformation, and visualization using Python, Pandas, Seaborn, and Manim.

---

## Overview

- **EDA:** Inspect dataset structure, types, missing values, and basic statistics.  
- **Data Cleaning:**  
  - Convert categorical columns (`cut`, `color`, `clarity`) to ordered categories  
  - Check for outliers (kept as legitimate)  
- **Data Transformation:**  
  - Derived columns: `price_per_carat`, `volume`, `depth_calculated`, `size_category`, `price_category`  
  - Flags for premium features: `is_ideal_cut`, `is_premium_color`, `is_premium_clarity`  
- **Summarization:** Groupby metrics, pivot tables, value counts, and cross-tabulations.  
- **Visualization:** Distribution, box, violin, scatter, regression plots, faceted comparisons, heatmaps, plus animated visualizations with Manim.  

---

## Key Insights

- Higher cut, premium color (D/E/F), and high clarity (IF/VVS) diamonds generally have higher price per carat.  
- Controlling for size is crucial to avoid misleading averages.  
- Medium-sized diamonds (0.5–1.0 carat) dominate the dataset, showing clear patterns across color and clarity.  

---
