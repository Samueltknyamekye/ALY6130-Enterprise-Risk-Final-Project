Do same with this: 

# Data Dictionary: streaming_subscribers.csv

Synthetic panel, one row per subscriber month. Generated with a fixed seed in build_analysis.py.

- content_cost_ratio: content cost as a share of revenue
- competitor_moves: competitor price or bundle moves in the quarter
- price_increase_pct: percent price increase on the subscriber's plan
- bundle_member: 1 if in the Apple One bundle, else 0
- tenure_months: months the subscriber has been active
- engagement_hours: monthly streaming hours
- support_tickets: support tickets raised
- region: NA, EU, APAC, or LATAM
- plan_type: Basic, Standard, Premium, or AdTier
- churn: 1 if the subscriber churned, else 0
