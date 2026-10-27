"""
Brand Traders - Sales Analytics Dashboard
Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A

st.set_page_config(page_title="Brand Traders | Sales Analytics", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

PALETTE = px.colors.qualitative.Set2
PRIMARY = "#1f6feb"
ACCENT = "#f59e0b"
GOOD, BAD = "#2ca02c", "#d62728"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px; padding: 12px 16px;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
.insight {border-left: 4px solid #1f6feb; padding: 6px 14px; margin: 6px 0; background: rgba(31,111,235,0.06);
    border-radius: 0 8px 8px 0;}
.reco {border-left: 4px solid #2ca02c; padding: 6px 14px; margin: 6px 0; background: rgba(44,160,44,0.07);
    border-radius: 0 8px 8px 0;}
</style>
""", unsafe_allow_html=True)


def money(x: float) -> str:
    return f"${x:,.0f}"


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=legend, legend=dict(orientation="h", y=-0.2),
                      title=dict(font=dict(size=15)))
    return fig


def show(fig: go.Figure, key: str):
    st.plotly_chart(fig, width="stretch", key=key)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and joining tables…")
def load_model(file_bytes: dict[str, bytes]):
    raw = A.load_raw(file_bytes)
    return A.build_model(raw)


def get_source_bytes() -> tuple[dict[str, bytes] | None, str]:
    """Uploaded files take priority; otherwise ./data (or the app folder)."""
    with st.sidebar.expander("📁 Data source", expanded=False):
        ups = st.file_uploader("Upload the 7 CSV tables (optional)", type="csv",
                               accept_multiple_files=True,
                               help="categories, customers, employees, order_details, orders, products, shippers")
    if ups:
        mapped = {}
        for f in ups:
            stem = Path(f.name).stem.lower()
            for n in A.FILES:
                if stem == n or stem.startswith(n + "_") or stem.startswith(n + " "):
                    mapped[n] = f.getvalue()
        if len(mapped) == len(A.FILES):
            return mapped, "uploaded files"
        st.sidebar.warning(f"Missing: {', '.join(sorted(set(A.FILES) - set(mapped)))}. Using bundled data.")
    here = Path(__file__).parent
    for folder in (here / "data", here):
        found = A.find_files(folder)
        if len(found) == len(A.FILES):
            return {n: p.read_bytes() for n, p in found.items()}, f"folder: {folder.name}/"
    return None, ""


src_bytes, src_label = get_source_bytes()
if src_bytes is None:
    st.error("The seven CSV files were not found. Put them in a `data/` folder next to app.py "
             "or upload them from the sidebar.")
    st.stop()

lines_all, orders_all, quality_log = load_model(src_bytes)

# --------------------------------------------------------------------------- #
# Sidebar filters  (empty multiselect = no restriction)
# --------------------------------------------------------------------------- #
st.sidebar.title("🔎 Filters")
st.sidebar.caption("Leave a list empty to include everything.")

dmin, dmax = lines_all["orderDate"].min().date(), lines_all["orderDate"].max().date()
date_sel = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
    d_from, d_to = date_sel
else:
    d_from, d_to = dmin, dmax

cats = st.sidebar.multiselect("Category", sorted(lines_all["categoryName"].unique()))
prod_pool = lines_all if not cats else lines_all[lines_all["categoryName"].isin(cats)]
prods = st.sidebar.multiselect("Product", sorted(prod_pool["productName"].unique()))
countries = st.sidebar.multiselect("Customer country", sorted(lines_all["customerCountry"].unique()))
cust_pool = lines_all if not countries else lines_all[lines_all["customerCountry"].isin(countries)]
custs = st.sidebar.multiselect("Customer", sorted(cust_pool["customerName"].unique()))
emps = st.sidebar.multiselect("Sales employee", sorted(lines_all["employeeName"].unique()))
ships = st.sidebar.multiselect("Shipper", sorted(lines_all["shipperName"].unique()))
disc_mode = st.sidebar.radio("Discount", ["All lines", "Discounted only", "Full price only"], horizontal=False)
status = st.sidebar.radio("Product status", ["All", "Active only", "Discontinued only"])
st.sidebar.caption(f"Data: {src_label}")


def apply_filters(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["orderDate"].dt.date >= start) & (df["orderDate"].dt.date <= end)
    if cats:
        m &= df["categoryName"].isin(cats)
    if prods:
        m &= df["productName"].isin(prods)
    if countries:
        m &= df["customerCountry"].isin(countries)
    if custs:
        m &= df["customerName"].isin(custs)
    if emps:
        m &= df["employeeName"].isin(emps)
    if ships:
        m &= df["shipperName"].isin(ships)
    if disc_mode == "Discounted only":
        m &= df["hasDiscount"]
    elif disc_mode == "Full price only":
        m &= ~df["hasDiscount"]
    if status == "Active only":
        m &= df["discontinued"] == 0
    elif status == "Discontinued only":
        m &= df["discontinued"] == 1
    return df[m]


lines = apply_filters(lines_all, d_from, d_to)
ords = A.order_level(lines, orders_all)

# previous period of equal length for KPI deltas
span = (pd.Timestamp(d_to) - pd.Timestamp(d_from)).days + 1
p_to = pd.Timestamp(d_from) - pd.Timedelta(days=1)
p_from = p_to - pd.Timedelta(days=span - 1)
prev_lines = apply_filters(lines_all, p_from.date(), p_to.date()) if p_from.date() >= dmin else lines_all.iloc[0:0]
prev_ords = A.order_level(prev_lines, orders_all) if len(prev_lines) else orders_all.iloc[0:0]
K = A.kpis(lines, ords)
KP = A.kpis(prev_lines, prev_ords) if len(prev_lines) else None


def delta(key: str, pct=True):
    if KP is None or not KP.get(key):
        return None
    if pct:
        return f"{K[key] / KP[key] - 1:+.1%} vs prev. period"
    return f"{(K[key] - KP[key]) * 100:+.1f} pp vs prev. period"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📊 Brand Traders · Sales Analytics")
st.caption(f"{d_from:%d %b %Y} → {d_to:%d %b %Y}  ·  {len(lines):,} order lines  ·  "
           f"{K['orders']:,} orders  ·  all figures in USD, net of discounts")

if lines.empty:
    st.warning("No data matches the current filters. Widen the selection in the sidebar.")
    st.stop()

tab_exec, tab_sales, tab_prod, tab_cust, tab_ops, tab_data, tab_model = st.tabs(
    ["🏁 Executive Summary", "📈 Sales Overview", "📦 Products & Categories",
     "👥 Customers", "🚚 Operations", "🗂️ Data & Downloads", "🧬 Data Model & Quality"])

mon = A.monthly(lines)
prod = A.by_product(lines)
cat = A.by_category(lines)
cust = A.by_customer(lines)
ctry = A.by_country(lines)
emp = A.by_employee(lines)
shp = A.by_shipper(ords)

# =========================================================================== #
# 1. EXECUTIVE SUMMARY
# =========================================================================== #
with tab_exec:
    c = st.columns(4)
    c[0].metric("Net revenue", money(K["revenue"]), delta("revenue"))
    c[1].metric("Orders", f"{K['orders']:,}", delta("orders"))
    c[2].metric("Avg. order value", money(K["aov"]), delta("aov"))
    c[3].metric("Units sold", f"{K['units']:,}", delta("units"))
    c = st.columns(4)
    c[0].metric("Active customers", f"{K['customers']:,}", delta("customers"))
    c[1].metric("Discount given", money(K["discount"]),
                help="Gross sales minus net revenue (unit price × qty × discount %)")
    c[2].metric("Discount rate", f"{K['discount_rate']:.1%}", delta("discount_rate", pct=False), delta_color="inverse")
    c[3].metric("Freight % of revenue", f"{K['freight_pct']:.1%}",
                help="Freight is an order-level charge; summed once per distinct order. "
                     "It is NOT profit or cost of goods.")
    c = st.columns(4)
    c[0].metric("Gross sales", money(K["gross"]))
    c[1].metric("Freight charged", money(K["freight"]))
    c[2].metric("Late shipments", f"{K['late_rate']:.1%}", help="Shipped after required date, among shipped orders",
                delta_color="inverse")
    c[3].metric("Avg. days to ship", f"{K['avg_days_ship']:.1f}")
    st.info("**Profit is not shown.** The dataset has no product cost / COGS column, so margin cannot be "
            "calculated honestly. Revenue, discounts and freight are reported instead.", icon="ℹ️")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        fig.add_bar(x=mon["orderMonth"], y=mon["revenue"], name="Monthly revenue", marker_color=PRIMARY,
                    marker_pattern_shape=["/" if p else "" for p in mon["partial"]])
        fig.add_scatter(x=mon["orderMonth"], y=mon["rolling3"], name="3-month average",
                        line=dict(color=ACCENT, width=3))
        fig.update_layout(title="Monthly net revenue (hatched = partial month)", yaxis_title="USD")
        show(style(fig), "exec_trend")
    with col2:
        fig = px.pie(cat, names="categoryName", values="revenue", hole=0.5, color_discrete_sequence=PALETTE,
                     title="Revenue mix by category")
        fig.update_traces(textinfo="percent+label")
        show(style(fig, legend=False), "exec_cat")

    findings, recos = A.generate_insights(lines, ords)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔍 Key findings")
        for t in findings:
            st.markdown(f"<div class='insight'>{t}</div>", unsafe_allow_html=True)
    with c2:
        st.subheader("✅ Recommendations")
        for t in recos:
            st.markdown(f"<div class='reco'>{t}</div>", unsafe_allow_html=True)
    st.caption("Findings and recommendations are regenerated from the filtered data on every change.")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("🏆 Top 5 products")
        st.dataframe(prod.head(5)[["productName", "categoryName", "revenue", "units", "share"]],
                     hide_index=True, width="stretch",
                     column_config={"productName": "Product", "categoryName": "Category",
                                    "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
                                    "units": "Units",
                                    "share": st.column_config.NumberColumn("Share", format="percent")})
    with t2:
        st.subheader("⚠️ Bottom 5 products")
        st.dataframe(prod.tail(5).sort_values("revenue")[["productName", "categoryName", "revenue", "units", "share"]],
                     hide_index=True, width="stretch",
                     column_config={"productName": "Product", "categoryName": "Category",
                                    "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
                                    "units": "Units",
                                    "share": st.column_config.NumberColumn("Share", format="percent")})

# =========================================================================== #
# 2. SALES OVERVIEW
# =========================================================================== #
with tab_sales:
    st.subheader("Sales performance over time")
    grain = st.radio("Time grain", ["Month", "Quarter", "Week"], horizontal=True, key="grain")
    freq = {"Month": "MS", "Quarter": "QS", "Week": "W-MON"}[grain]
    ts = (lines.set_index("orderDate").resample(freq)
          .agg(revenue=("netRevenue", "sum"), units=("quantity", "sum"), discount=("discountAmount", "sum"))
          .reset_index())
    ts_orders = ords.set_index("orderDate").resample(freq)["orderID"].count().reset_index(name="orders")
    ts = ts.merge(ts_orders, on="orderDate")
    ts["aov"] = ts["revenue"] / ts["orders"].where(ts["orders"] > 0)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.area(ts, x="orderDate", y="revenue", title=f"Net revenue by {grain.lower()}",
                      color_discrete_sequence=[PRIMARY])
        fig.update_traces(line_shape="spline")
        show(style(fig, legend=False), "s_rev")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=ts["orderDate"], y=ts["orders"], name="Orders", marker_color=PALETTE[1])
        fig.add_scatter(x=ts["orderDate"], y=ts["aov"], name="Avg. order value", yaxis="y2",
                        line=dict(color=ACCENT, width=3))
        fig.update_layout(title=f"Orders and average order value by {grain.lower()}",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="AOV (USD)"),
                          yaxis_title="Orders")
        show(style(fig), "s_ord")

    c1, c2 = st.columns(2)
    with c1:
        m2 = mon.dropna(subset=["mom_growth"])
        m2 = m2.assign(direction=m2["mom_growth"].gt(0).map({True: "Growth", False: "Decline"}))
        fig = px.bar(m2, x="orderMonth", y="mom_growth", title="Month-over-month revenue growth",
                     color="direction", color_discrete_map={"Growth": GOOD, "Decline": BAD})
        fig.update_yaxes(tickformat=".0%")
        show(style(fig, legend=False), "s_mom")
    with c2:
        mc = mon.copy()
        mc["cumulative"] = mc["revenue"].cumsum()
        fig = px.line(mc, x="orderMonth", y="cumulative", markers=True, title="Cumulative net revenue",
                      color_discrete_sequence=[PRIMARY])
        show(style(fig, legend=False), "s_cum")

    c1, c2 = st.columns(2)
    with c1:
        cm = A.category_monthly(lines)
        fig = px.area(cm, x="orderMonth", y="netRevenue", color="categoryName",
                      title="Revenue by category over time", color_discrete_sequence=PALETTE)
        show(style(fig), "s_catm")
    with c2:
        wd = (lines.groupby("weekday").agg(revenue=("netRevenue", "sum"), orders=("orderID", "nunique"))
              .reindex(WEEKDAYS).fillna(0).reset_index())
        fig = px.bar(wd, x="weekday", y="revenue", title="Revenue by day of week (order date)",
                     color_discrete_sequence=[PRIMARY], hover_data=["orders"])
        show(style(fig, legend=False), "s_wd")

    st.subheader("Seasonality heatmap")
    hm = lines.assign(Year=lines["orderDate"].dt.year, Month=lines["orderDate"].dt.month_name().str[:3])
    hm = hm.pivot_table(index="Year", columns="Month", values="netRevenue", aggfunc="sum")
    order_m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hm = hm.reindex(columns=[m for m in order_m if m in hm.columns])
    fig = px.imshow(hm, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                    title="Net revenue by year and month (blank = no orders in range)")
    show(style(fig, height=260, legend=False), "s_heat")

    st.subheader("Outliers: unusually large orders")
    q1, q3 = ords["orderRevenue"].quantile([0.25, 0.75])
    thr = q3 + 1.5 * (q3 - q1)
    out = ords[ords["orderRevenue"] > thr].sort_values("orderRevenue", ascending=False)
    st.caption(f"Orders above {money(thr)} (Q3 + 1.5×IQR): **{len(out)}** orders, "
               f"**{out['orderRevenue'].sum() / K['revenue']:.0%}** of revenue.")
    st.dataframe(out[["orderID", "orderDate", "customerName", "customerCountry", "employeeName",
                      "orderRevenue", "orderItems"]].head(15), hide_index=True, width="stretch",
                 column_config={"orderRevenue": st.column_config.NumberColumn("Revenue", format="$%.0f")})

# =========================================================================== #
# 3. PRODUCTS & CATEGORIES
# =========================================================================== #
with tab_prod:
    st.caption("The dataset has no brand/supplier column, so **products** are the brand-level unit of analysis.")
    st.subheader("Category comparison")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.treemap(prod, path=[px.Constant("All"), "categoryName", "productName"], values="revenue",
                         color="categoryName", color_discrete_sequence=PALETTE, title="Revenue treemap: category → product")
        show(style(fig, height=430, legend=False), "p_tree")
    with c2:
        fig = px.bar(cat.sort_values("revenue"), x="revenue", y="categoryName", orientation="h",
                     text=cat.sort_values("revenue")["share"].map("{:.0%}".format),
                     title="Net revenue by category", color_discrete_sequence=[PRIMARY])
        show(style(fig, height=430, legend=False), "p_cat")
    st.dataframe(cat, hide_index=True, width="stretch", column_config={
        "categoryName": "Category", "revenue": st.column_config.NumberColumn("Net revenue", format="$%.0f"),
        "gross": st.column_config.NumberColumn("Gross", format="$%.0f"),
        "discount": st.column_config.NumberColumn("Discount", format="$%.0f"),
        "share": st.column_config.NumberColumn("Share", format="percent"),
        "discount_rate": st.column_config.NumberColumn("Discount rate", format="percent"),
        "revenue_per_product": st.column_config.NumberColumn("Revenue / product", format="$%.0f")})

    st.subheader("Product performance")
    n = st.slider("Number of products to show", 5, min(30, max(5, len(prod))), min(10, len(prod)), key="topn_prod")
    metric = st.radio("Rank by", ["revenue", "units", "orders"], horizontal=True, key="rank_prod",
                      format_func=lambda x: {"revenue": "Net revenue", "units": "Units sold", "orders": "Order count"}[x])
    ranked = prod.sort_values(metric, ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        d = ranked.head(n).sort_values(metric)
        fig = px.bar(d, x=metric, y="productName", orientation="h", color="categoryName",
                     title=f"Top {n} products", color_discrete_sequence=PALETTE)
        show(style(fig, height=max(360, 26 * n + 120)), "p_top")
    with c2:
        d = ranked.tail(n).sort_values(metric, ascending=False)
        fig = px.bar(d, x=metric, y="productName", orientation="h", color="categoryName",
                     title=f"Bottom {n} products", color_discrete_sequence=PALETTE)
        show(style(fig, height=max(360, 26 * n + 120)), "p_bot")

    c1, c2 = st.columns(2)
    with c1:
        par = prod.reset_index(drop=True)
        fig = go.Figure()
        fig.add_bar(x=par["rank"], y=par["revenue"], name="Revenue", marker_color=PRIMARY)
        fig.add_scatter(x=par["rank"], y=par["cum_share"], name="Cumulative share", yaxis="y2",
                        line=dict(color=ACCENT, width=3))
        fig.add_hline(y=0.8, line_dash="dot", line_color="grey", yref="y2")
        fig.update_layout(title="Pareto: how concentrated is revenue?", xaxis_title="Product rank",
                          yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05], showgrid=False))
        show(style(fig), "p_par")
    with c2:
        fig = px.scatter(prod, x="units", y="revenue", size="orders", color="categoryName", hover_name="productName",
                         title="Volume vs revenue (bubble = number of orders)", color_discrete_sequence=PALETTE)
        show(style(fig), "p_sc")

    st.subheader("Discount effectiveness")
    db = A.discount_buckets(lines)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(db, x="discount_band", y="avg_qty", text=db["avg_qty"].round(1),
                     title="Average quantity per order line by discount level", color_discrete_sequence=[PALETTE[2]])
        show(style(fig, legend=False), "p_disc_q")
    with c2:
        fig = px.bar(db[db["discount_band"] != "None"], x="discount_band", y="discount_cost",
                     title="Discount cost by discount level (USD)", color_discrete_sequence=[BAD])
        show(style(fig, legend=False), "p_disc_c")
    heavy = prod.sort_values("discount", ascending=False).head(10)
    st.markdown("**Products receiving the largest discount value**")
    st.dataframe(heavy[["productName", "categoryName", "gross", "discount", "discount_rate"]], hide_index=True,
                 width="stretch", column_config={
                     "gross": st.column_config.NumberColumn("Gross sales", format="$%.0f"),
                     "discount": st.column_config.NumberColumn("Discount", format="$%.0f"),
                     "discount_rate": st.column_config.NumberColumn("Rate", format="percent")})

    st.subheader("Full product table")
    st.dataframe(prod.drop(columns=["productID", "cum_share"]), hide_index=True, width="stretch", column_config={
        "productName": "Product", "categoryName": "Category",
        "discontinued": st.column_config.CheckboxColumn("Discontinued"),
        "revenue": st.column_config.NumberColumn("Net revenue", format="$%.0f"),
        "gross": st.column_config.NumberColumn("Gross", format="$%.0f"),
        "discount": st.column_config.NumberColumn("Discount", format="$%.0f"),
        "share": st.column_config.NumberColumn("Share", format="percent"),
        "discount_rate": st.column_config.NumberColumn("Disc. rate", format="percent"),
        "avg_price": st.column_config.NumberColumn("Avg. price", format="$%.2f")})

# =========================================================================== #
# 4. CUSTOMERS
# =========================================================================== #
with tab_cust:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(cust):,}")
    c2.metric("Revenue / customer", money(K["revenue"] / max(len(cust), 1)))
    c3.metric("Orders / customer", f"{K['orders'] / max(len(cust), 1):.1f}")
    c4.metric("Top-10 share of revenue", f"{cust.head(10)['share'].sum():.0%}")

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Top customers to show", 5, 30, 10, key="topn_cust")
        d = cust.head(n).sort_values("revenue")
        fig = px.bar(d, x="revenue", y="customerName", orientation="h", color="customerCountry",
                     title=f"Top {n} customers by net revenue", color_discrete_sequence=PALETTE)
        show(style(fig, height=max(380, 26 * n + 120)), "c_top")
    with c2:
        cc = cust.reset_index(drop=True)
        cc["pct_customers"] = (cc.index + 1) / len(cc)
        fig = px.line(cc, x="pct_customers", y="cum_share", title="Revenue concentration curve",
                      color_discrete_sequence=[PRIMARY])
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
        fig.update_xaxes(tickformat=".0%", title="Share of customers (largest first)")
        fig.update_yaxes(tickformat=".0%", title="Share of revenue")
        show(style(fig, height=max(380, 26 * n + 120), legend=False), "c_conc")

    st.subheader("Geography")
    c1, c2 = st.columns(2)
    name_map = {"USA": "United States", "UK": "United Kingdom"}
    with c1:
        g = ctry.assign(map_name=ctry["customerCountry"].replace(name_map))
        fig = px.choropleth(g, locations="map_name", locationmode="country names", color="revenue",
                            hover_name="customerCountry", color_continuous_scale="Blues",
                            title="Net revenue by customer country")
        fig.update_geos(showframe=False, projection_type="natural earth")
        show(style(fig, legend=False), "c_map")
    with c2:
        fig = px.bar(ctry.head(12).sort_values("revenue"), x="revenue", y="customerCountry", orientation="h",
                     color="revenue_per_customer", color_continuous_scale="Tealgrn",
                     title="Top countries (colour = revenue per customer)")
        show(style(fig, legend=False), "c_ctry")

    st.subheader("New vs returning customers")
    first_order = orders_all.groupby("customerID")["orderDate"].min().rename("firstOrder")
    act = ords.merge(first_order, on="customerID")
    act["orderMonth"] = act["orderDate"].dt.to_period("M").dt.to_timestamp()
    act["firstMonth"] = act["firstOrder"].dt.to_period("M").dt.to_timestamp()
    act["type"] = (act["orderMonth"] == act["firstMonth"]).map({True: "New", False: "Returning"})
    nr = act.groupby(["orderMonth", "type"])["customerID"].nunique().reset_index(name="customers")
    fig = px.bar(nr, x="orderMonth", y="customers", color="type", title="Active customers per month",
                 color_discrete_map={"New": ACCENT, "Returning": PRIMARY})
    show(style(fig, height=320), "c_nr")
    st.caption("'New' = the customer's first-ever order in the whole dataset fell in that month.")

    st.subheader("RFM segmentation")
    st.caption(f"Recency is measured from the last order date in the filtered data ({lines['orderDate'].max():%d %b %Y}). "
               "Scores 1–4 are quartiles (4 = best).")
    rf = A.rfm(cust)
    if "R" in rf.columns:
        seg = rf.groupby("segment").agg(customers=("customerID", "count"), revenue=("revenue", "sum")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(seg.sort_values("customers"), x="customers", y="segment", orientation="h",
                         title="Customers per segment", color_discrete_sequence=[PALETTE[0]])
            show(style(fig, height=320, legend=False), "c_seg1")
        with c2:
            fig = px.pie(seg, names="segment", values="revenue", hole=0.45, title="Revenue per segment",
                         color_discrete_sequence=PALETTE)
            show(style(fig, height=320), "c_seg2")
        pick = st.multiselect("Show customers in segment", sorted(rf["segment"].unique()),
                              default=["At risk (high value)"] if "At risk (high value)" in set(rf["segment"]) else [])
        view = rf[rf["segment"].isin(pick)] if pick else rf
        st.dataframe(view[["customerName", "customerCountry", "segment", "revenue", "orders", "recency_days",
                           "last_order", "R", "F", "M"]], hide_index=True, width="stretch",
                     column_config={"revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
                                    "recency_days": "Days since last order",
                                    "last_order": st.column_config.DateColumn("Last order")})
    else:
        st.info("Too few customers in the current selection for RFM segmentation.")

# =========================================================================== #
# 5. OPERATIONS
# =========================================================================== #
with tab_ops:
    st.subheader("Sales team")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(emp.sort_values("revenue"), x="revenue", y="employeeName", orientation="h",
                     title="Net revenue by employee", color_discrete_sequence=[PRIMARY])
        show(style(fig, legend=False), "o_emp")
    with c2:
        fig = px.scatter(emp, x="orders", y="aov", size="revenue", text="employeeName",
                         title="Orders handled vs average order value (bubble = revenue)",
                         color_discrete_sequence=[PALETTE[1]])
        fig.update_traces(textposition="top center")
        show(style(fig, legend=False), "o_emp2")

    st.subheader("Shipping & fulfilment")
    shipped = ords[ords["isShipped"]]
    st.caption(f"{int((~ords['isShipped']).sum())} orders have no shipped date and are excluded from delivery statistics. "
               "Freight is counted once per order.")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(shp, x="shipperName", y="orders", title="Orders by shipper", color="shipperName",
                     color_discrete_sequence=PALETTE)
        show(style(fig, legend=False, height=320), "o_s1")
    with c2:
        fig = px.bar(shp, x="shipperName", y="avg_freight", title="Average freight per order (USD)",
                     color="shipperName", color_discrete_sequence=PALETTE)
        show(style(fig, legend=False, height=320), "o_s2")
    with c3:
        fig = px.bar(shp, x="shipperName", y="late_rate", title="Late-shipment rate", color="shipperName",
                     color_discrete_sequence=PALETTE)
        fig.update_yaxes(tickformat=".1%")
        show(style(fig, legend=False, height=320), "o_s3")

    c1, c2 = st.columns(2)
    with c1:
        if len(shipped):
            fig = px.histogram(shipped, x="daysToShip", nbins=30, title="Days from order to shipment",
                               color_discrete_sequence=[PRIMARY])
            show(style(fig, legend=False), "o_hist")
    with c2:
        if len(shipped):
            lm = shipped.assign(month=shipped["orderDate"].dt.to_period("M").dt.to_timestamp())
            lm = lm.groupby("month").agg(late=("isLate", "mean"), orders=("orderID", "count")).reset_index()
            fig = px.line(lm, x="month", y="late", markers=True, title="Late-shipment rate by month",
                          color_discrete_sequence=[BAD])
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, legend=False), "o_late")

    c1, c2 = st.columns(2)
    with c1:
        fc = (ords.groupby("customerCountry").agg(freight=("freight", "sum"), revenue=("orderRevenue", "sum"))
              .assign(pct=lambda d: d["freight"] / d["revenue"]).reset_index().sort_values("freight", ascending=False).head(12))
        fig = px.bar(fc, x="customerCountry", y="freight", color="pct", color_continuous_scale="Reds",
                     title="Freight by country (colour = freight % of revenue)")
        show(style(fig, legend=False), "o_fc")
    with c2:
        sc = px.scatter(ords, x="orderRevenue", y="freight", color="shipperName", opacity=0.6,
                        title="Freight vs order value", color_discrete_sequence=PALETTE, hover_name="customerName")
        show(style(sc), "o_fs")

    late = shipped[shipped["isLate"]].sort_values("orderDate", ascending=False)
    pending = ords[~ords["isShipped"]].sort_values("requiredDate")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Late orders ({len(late)})**")
        st.dataframe(late[["orderID", "customerName", "requiredDate", "shippedDate", "shipperName", "orderRevenue"]]
                     .head(50), hide_index=True, width="stretch")
    with c2:
        st.markdown(f"**Unshipped orders ({len(pending)})**")
        st.dataframe(pending[["orderID", "customerName", "orderDate", "requiredDate", "shipperName", "orderRevenue"]],
                     hide_index=True, width="stretch")

# =========================================================================== #
# 6. DATA & DOWNLOADS
# =========================================================================== #
with tab_data:
    st.subheader("Filtered data")
    st.caption("All tables below respect the sidebar filters.")
    datasets = {
        "Order lines (fully joined)": lines.drop(columns=["orderMonth", "orderYear", "weekday", "priceVsList"]),
        "Orders (one row per order)": ords,
        "Products summary": prod,
        "Categories summary": cat,
        "Customers summary": cust,
        "Countries summary": ctry,
        "Employees summary": emp,
        "Monthly summary": mon,
    }
    choice = st.selectbox("Dataset", list(datasets))
    df = datasets[choice]
    q = st.text_input("Search text in this table", "")
    if q:
        mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
        df = df[mask]
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

    c1, c2, c3 = st.columns(3)
    safe = choice.split(" (")[0].lower().replace(" ", "_")
    c1.download_button("⬇️ Download this table (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"brand_traders_{safe}.csv", mime="text/csv")

    @st.cache_data(show_spinner="Building Excel report…")
    def build_report(key: str, _sheets: dict) -> bytes:
        # `key` identifies the filter state (cache key); `_sheets` is excluded from hashing.
        return A.to_excel_bytes(_sheets)

    kpi_df = pd.DataFrame({"Metric": ["Net revenue", "Gross sales", "Discount given", "Discount rate", "Orders",
                                      "Avg order value", "Units", "Customers", "Freight", "Freight % revenue",
                                      "Late shipment rate", "Avg days to ship"],
                           "Value": [K["revenue"], K["gross"], K["discount"], K["discount_rate"], K["orders"],
                                     K["aov"], K["units"], K["customers"], K["freight"], K["freight_pct"],
                                     K["late_rate"], K["avg_days_ship"]]})
    findings_df = pd.DataFrame({"Type": ["Finding"] * len(findings) + ["Recommendation"] * len(recos),
                                "Text": [f.replace("**", "") for f in findings] + [r.replace("**", "") for r in recos]})
    sheets = {"KPIs": kpi_df, "Insights": findings_df, "Monthly": mon, "Categories": cat, "Products": prod,
              "Customers": cust, "Countries": ctry, "Employees": emp, "Shippers": shp,
              "Orders": ords, "Order lines": datasets["Order lines (fully joined)"]}
    key = f"{d_from}|{d_to}|{cats}|{prods}|{countries}|{custs}|{emps}|{ships}|{disc_mode}|{status}|{len(lines)}"
    c2.download_button("⬇️ Download this table (Excel)", A.to_excel_bytes({choice[:31]: df}),
                       file_name=f"brand_traders_{safe}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c3.download_button("📘 Full Excel report (all sheets)", build_report(key, sheets),
                       file_name="brand_traders_full_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================================================================== #
# 7. DATA MODEL & QUALITY
# =========================================================================== #
with tab_model:
    st.subheader("Table relationships")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4ff", fontname="Helvetica"];
      categories -> products [label="categoryID"];
      products -> order_details [label="productID"];
      orders -> order_details [label="orderID"];
      customers -> orders [label="customerID"];
      employees -> orders [label="employeeID"];
      shippers -> orders [label="shipperID"];
      employees -> employees [label="reportsTo"];
    }""")
    st.markdown("""
| Table | Primary key | Foreign keys | Grain |
|---|---|---|---|
| categories | categoryID | – | one row per category |
| products | productID | categoryID → categories | one row per product |
| customers | customerID | – | one row per customer |
| employees | employeeID | reportsTo → employees | one row per employee |
| shippers | shipperID | – | one row per shipper |
| orders | orderID | customerID, employeeID, shipperID | one row per order |
| order_details | (orderID, productID) | orderID → orders, productID → products | one row per order line |
""")
    st.subheader("How the data was prepared")
    st.markdown("""
- **Originals are never modified**; cleaning happens on copies.
- All joins are **many-to-one from the order-line table** (lines → orders → customers / employees / shippers, lines → products → categories), validated with `validate="m:1"`; the line count is verified unchanged after joining.
- **Net revenue** = `unitPrice × quantity × (1 − discount)` using the price on the order line (historical), not today's list price.
- **Freight** is stored once per order. It is summed over distinct orders only; summing it on the joined line table would overstate it about 3×.
- Missing `shippedDate` is treated as *not shipped yet*, not imputed.
- Text columns are trimmed, dates parsed, numerics coerced, and mixed file encodings (UTF-8 / Windows-1252) handled.
""")
    st.warning("**Limitations:** no cost/COGS → no profit or margin; no brand/supplier column → analysis by product and category; "
               "the first and last months of the data are partial, so month-over-month comparisons at the edges can mislead.")
    st.subheader("Data-quality log")
    st.dataframe(quality_log, hide_index=True, width="stretch", height=420)
    with st.expander("Raw table previews"):
        raw_preview = A.load_raw(src_bytes)
        t = st.selectbox("Table", A.FILES, key="raw_tbl")
        st.dataframe(raw_preview[t].head(100), hide_index=True, width="stretch")
