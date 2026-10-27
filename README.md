streamlit app link https://brand-trader-sales-analytics-app.streamlit.app/
# Brand Traders · Sales Analytics (Streamlit)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The seven CSVs are bundled in `data/`. You can also upload new copies from the sidebar
(**Data source**); names like `orders_1.csv` are recognised.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: filters, KPI cards, charts, tables, downloads |
| `analytics.py` | Loading, cleaning, joins, KPIs, RFM, automated insights, Excel export (no UI code) |
| `data/` | The 7 source CSVs (unchanged) |

## Tabs
Executive Summary · Sales Overview · Products & Categories · Customers · Operations · Data & Downloads · Data Model & Quality

## Key modelling decisions
- Net revenue = unitPrice × quantity × (1 − discount), using the historical price on each order line.
- Freight is an order-level value; it is summed per distinct order, never on the joined line table (that would triple-count it).
- No cost/COGS column, so **no profit or margin** is shown. No brand column, so brand analysis is by product and category.
- First (Jul 2013) and last (May 2015) months are partial and are flagged in charts.
"# Trade-Analysis-Data" 
