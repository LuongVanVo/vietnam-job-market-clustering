# Stage 04 EDA Summary Optimized

## Dataset overview

- Train rows: 546,190
- Test rows: 60,688
- Clean input columns: 31
- EDA columns after display features: 33

## Key categorical insights

- Top primary industry: Bán hàng - Kinh doanh (9.70%)
- Top location city: TP Hồ Chí Minh (52.25%)
- Unknown location city ratio: 2.46%
- Top job type: Toàn thời gian (92.15%)

## Salary insights

- Salary valid filter: VND, no issue, salary_avg_million_vnd from 1 to 200.
- Valid salary rows: 492,340
- Valid salary ratio: 90.14%
- Median salary_avg_million_vnd: 12.00
- Mean salary_avg_million_vnd: 13.53

## Text length insights

- Median job_description word count: 104.00
- Median requirements word count: 61.00

## Optimizations applied

- Standardized display labels for job_position and education_level.
- Added salary zoom distribution for 1-50 million VND/month.
- Excluded 'Khác/Không rõ' from salary-by-location boxplot.
- Added median salary by primary_industry chart.
- Filtered broken or overly generic n-grams.

## Implication for clustering

- Text columns contain important semantic information, so TF-IDF should be used in feature engineering.
- primary_industry, location_city, job_type, experience_level, education_level, and salary features can help interpret clusters.
- Salary should be used with control rules because it contains unavailable and abnormal values.