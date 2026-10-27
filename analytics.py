"""
Brand Traders - data loading, joining and analytics (pure pandas, no UI code).

Schema (primary keys -> foreign keys)
-------------------------------------
categories.categoryID   (PK)
products.productID      (PK)  products.categoryID      -> categories
customers.customerID    (PK)
employees.employeeID    (PK)  employees.reportsTo      -> employees (self reference)
shippers.shipperID      (PK)
orders.orderID          (PK)  orders.customerID        -> customers
                              orders.employeeID        -> employees
                              orders.shipperID         -> shippers
order_details (orderID, productID) composite PK
                              order_details.orderID    -> orders
                              order_details.productID  -> products

Grain warning
-------------
`freight` lives on the ORDER (one value per order), while revenue lives on the
ORDER LINE.  After joining orders to order_details the freight value is repeated
on every line, so summing it there would overstate freight.  This module keeps
freight in the order-level table and only ever sums it over distinct orders.

Not in the dataset: product cost / COGS, brand, supplier.  Therefore no profit
or margin is calculated, and "brand" analysis is done at product level.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ["categories", "customers", "employees", "order_details",
         "orders", "products", "shippers"]
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_csv(src, name: str) -> pd.DataFrame:
    """Read a path or uploaded file-like object, trying several encodings."""
    if isinstance(src, (bytes, bytearray)):
        raw = bytes(src)
    elif hasattr(src, "getvalue"):
        raw = src.getvalue()
    elif hasattr(src, "read"):
        raw = src.read()
    else:
        raw = Path(src).read_bytes()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Could not decode {name}")


def find_files(folder: str | Path) -> dict:
    """Locate the seven CSVs in a folder; tolerates suffixes like orders_1.csv."""
    folder = Path(folder)
    found = {}
    for name in FILES:
        exact = folder / f"{name}.csv"
        if exact.exists():
            found[name] = exact
            continue
        cands = sorted(folder.glob(f"{name}_*.csv")) + sorted(folder.glob(f"{name}*.csv"))
        if cands:
            found[name] = cands[0]
    return found


def load_raw(sources: dict) -> dict[str, pd.DataFrame]:
    """sources: {table_name: path | uploaded file}. Returns untouched raw frames."""
    missing = [n for n in FILES if n not in sources]
    if missing:
        raise FileNotFoundError(f"Missing tables: {', '.join(missing)}")
    return {n: _read_csv(sources[n], n) for n in FILES}


# --------------------------------------------------------------------------- #
# Cleaning / modelling
# --------------------------------------------------------------------------- #
def build_model(raw: dict[str, pd.DataFrame]):
    """
    Returns (lines, orders, quality_log).

    lines  : one row per order line, fully enriched (revenue measures included)
    orders : one row per order (freight, shipping timings, order-level revenue)
    quality_log : DataFrame documenting every check / transformation
    Raw frames are never mutated.
    """
    log = []

    def note(table, check, result, action):
        log.append({"Table": table, "Check": check, "Result": result, "Action": action})

    cat = raw["categories"].copy()
    cust = raw["customers"].copy()
    emp = raw["employees"].copy()
    od = raw["order_details"].copy()
    orders = raw["orders"].copy()
    prod = raw["products"].copy()
    ship = raw["shippers"].copy()

    # ---- generic hygiene --------------------------------------------------
    for name, df in [("categories", cat), ("customers", cust), ("employees", emp),
                     ("order_details", od), ("orders", orders),
                     ("products", prod), ("shippers", ship)]:
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype("string").str.strip()
        d = int(df.duplicated().sum())
        note(name, "Exact duplicate rows", f"{d} found", "Dropped" if d else "None needed")
        if d:
            df.drop_duplicates(inplace=True)

    # ---- key integrity ----------------------------------------------------
    pk_checks = [("categories", cat, "categoryID"), ("customers", cust, "customerID"),
                 ("employees", emp, "employeeID"), ("orders", orders, "orderID"),
                 ("products", prod, "productID"), ("shippers", ship, "shipperID")]
    for name, df, key in pk_checks:
        dup = int(df[key].duplicated().sum())
        note(name, f"Primary key '{key}' unique", "Yes" if dup == 0 else f"{dup} duplicates",
             "OK" if dup == 0 else "Kept first occurrence")
        if dup:
            df.drop_duplicates(subset=key, keep="first", inplace=True)

    dup_od = int(od.duplicated(["orderID", "productID"]).sum())
    note("order_details", "Composite key (orderID, productID) unique",
         "Yes" if dup_od == 0 else f"{dup_od} duplicates",
         "OK" if dup_od == 0 else "Kept first occurrence")
    if dup_od:
        od = od.drop_duplicates(["orderID", "productID"], keep="first")

    fk_checks = [
        ("order_details.orderID -> orders", od["orderID"], orders["orderID"]),
        ("order_details.productID -> products", od["productID"], prod["productID"]),
        ("orders.customerID -> customers", orders["customerID"], cust["customerID"]),
        ("orders.employeeID -> employees", orders["employeeID"], emp["employeeID"]),
        ("orders.shipperID -> shippers", orders["shipperID"], ship["shipperID"]),
        ("products.categoryID -> categories", prod["categoryID"], cat["categoryID"]),
    ]
    for label, child, parent in fk_checks:
        orphan = int((~child.isin(parent)).sum())
        note("relationship", f"Foreign key {label}", f"{orphan} orphan rows",
             "OK" if orphan == 0 else "Orphans kept; shown as 'Unknown' after left join")

    # ---- types / dates ----------------------------------------------------
    for c in ["orderDate", "requiredDate", "shippedDate"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    unshipped = int(orders["shippedDate"].isna().sum())
    note("orders", "Missing shippedDate", f"{unshipped} orders",
         "Kept as NaT: treated as 'not yet shipped', excluded from delivery-time stats")
    bad = int((orders["shippedDate"] < orders["orderDate"]).sum())
    note("orders", "shippedDate earlier than orderDate", f"{bad} rows", "OK" if bad == 0 else "Flagged")
    for c in ["unitPrice", "quantity", "discount"]:
        od[c] = pd.to_numeric(od[c], errors="coerce")
    orders["freight"] = pd.to_numeric(orders["freight"], errors="coerce")
    note("order_details", "Null / non-numeric price, qty, discount",
         f"{int(od[['unitPrice','quantity','discount']].isna().sum().sum())} cells", "OK")
    note("order_details", "discount range", f"{od['discount'].min():.2f} - {od['discount'].max():.2f}",
         "Valid fractions (0-1)")
    note("employees", "Missing reportsTo", f"{int(emp['reportsTo'].isna().sum())} row(s)",
         "Expected: top of hierarchy, left as null")
    emp["reportsTo"] = emp["reportsTo"].astype("Int64")

    # ---- enrichment (many-to-one joins only => no row multiplication) ------
    prod = prod.rename(columns={"unitPrice": "listPrice"})
    prod = prod.merge(cat.rename(columns={"description": "categoryDescription"}),
                      on="categoryID", how="left", validate="m:1")
    prod["discontinued"] = prod["discontinued"].astype(int)

    cust = cust.rename(columns={"companyName": "customerName", "city": "customerCity",
                                "country": "customerCountry"})
    emp = emp.rename(columns={"city": "employeeCity", "country": "employeeCountry"})
    ship = ship.rename(columns={"companyName": "shipperName"})

    orders = (orders
              .merge(cust[["customerID", "customerName", "customerCity", "customerCountry"]],
                     on="customerID", how="left", validate="m:1")
              .merge(emp[["employeeID", "employeeName"]], on="employeeID", how="left", validate="m:1")
              .merge(ship, on="shipperID", how="left", validate="m:1"))
    for c in ["customerName", "customerCity", "customerCountry", "employeeName", "shipperName"]:
        orders[c] = orders[c].fillna("Unknown")

    lines = od.merge(orders, on="orderID", how="left", validate="m:1")
    n_before = len(od)
    lines = lines.merge(prod[["productID", "productName", "quantityPerUnit", "listPrice",
                              "discontinued", "categoryID", "categoryName"]],
                        on="productID", how="left", validate="m:1")
    lines["productName"] = lines["productName"].fillna("Unknown")
    lines["categoryName"] = lines["categoryName"].fillna("Unknown")
    note("model", "Row count after joins", f"{len(lines)} lines vs {n_before} order_details rows",
         "Identical => no duplication from joins" if len(lines) == n_before else "CHECK JOINS")

    # ---- measures ----------------------------------------------------------
    lines["grossSales"] = lines["unitPrice"] * lines["quantity"]
    lines["discountAmount"] = lines["grossSales"] * lines["discount"]
    lines["netRevenue"] = lines["grossSales"] - lines["discountAmount"]
    lines["hasDiscount"] = lines["discount"] > 0
    lines["priceVsList"] = lines["unitPrice"] / lines["listPrice"] - 1
    lines["orderMonth"] = lines["orderDate"].dt.to_period("M").dt.to_timestamp()
    lines["orderYear"] = lines["orderDate"].dt.year
    lines["weekday"] = lines["orderDate"].dt.day_name()
    note("order_details", "Price differs from current products.unitPrice",
         f"{int((lines['unitPrice'].round(2) != lines['listPrice'].round(2)).sum())} lines",
         "Historical price on the order line is used for revenue (correct practice); "
         "products.unitPrice is treated as the current list price")
    note("model", "Profit / margin", "No cost (COGS) column in data",
         "NOT calculated. Revenue, discounts and freight are reported instead")
    note("model", "Brand", "No brand / supplier column in data",
         "Brand analysis performed at product and category level")

    # order-level revenue (freight stays at order grain)
    rev = lines.groupby("orderID").agg(orderRevenue=("netRevenue", "sum"),
                                       orderGross=("grossSales", "sum"),
                                       orderItems=("quantity", "sum"),
                                       orderLines=("productID", "count")).reset_index()
    orders = orders.merge(rev, on="orderID", how="left", validate="1:1")
    orders["daysToShip"] = (orders["shippedDate"] - orders["orderDate"]).dt.days
    orders["isLate"] = orders["shippedDate"] > orders["requiredDate"]
    orders["isShipped"] = orders["shippedDate"].notna()
    orders["freightPctRevenue"] = orders["freight"] / orders["orderRevenue"]

    return lines, orders, pd.DataFrame(log)


# --------------------------------------------------------------------------- #
# Aggregations used by the dashboard
# --------------------------------------------------------------------------- #
def order_level(lines: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Orders restricted to those present in the filtered lines.  Revenue is
    recomputed from the filtered lines so it respects product/category filters;
    freight stays as the full order freight (one value per order)."""
    keep = lines.groupby("orderID").agg(orderRevenue=("netRevenue", "sum"),
                                        orderGross=("grossSales", "sum"),
                                        orderItems=("quantity", "sum")).reset_index()
    base = orders.drop(columns=["orderRevenue", "orderGross", "orderItems"], errors="ignore")
    out = base.merge(keep, on="orderID", how="inner", validate="1:1")
    out["freightPctRevenue"] = out["freight"] / out["orderRevenue"]
    return out


def kpis(lines: pd.DataFrame, ords: pd.DataFrame) -> dict:
    if lines.empty:
        return {k: 0 for k in ["revenue", "gross", "discount", "discount_rate", "orders",
                               "aov", "units", "customers", "products", "freight",
                               "freight_pct", "late_rate", "avg_days_ship", "lines_with_discount"]}
    revenue = lines["netRevenue"].sum()
    gross = lines["grossSales"].sum()
    disc = lines["discountAmount"].sum()
    n_orders = lines["orderID"].nunique()
    shipped = ords[ords["isShipped"]]
    return {
        "revenue": revenue,
        "gross": gross,
        "discount": disc,
        "discount_rate": disc / gross if gross else 0,
        "orders": n_orders,
        "aov": revenue / n_orders if n_orders else 0,
        "units": int(lines["quantity"].sum()),
        "customers": lines["customerID"].nunique(),
        "products": lines["productID"].nunique(),
        "freight": ords["freight"].sum(),
        "freight_pct": ords["freight"].sum() / revenue if revenue else 0,
        "late_rate": shipped["isLate"].mean() if len(shipped) else 0,
        "avg_days_ship": shipped["daysToShip"].mean() if len(shipped) else 0,
        "lines_with_discount": lines["hasDiscount"].mean(),
    }


def monthly(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["orderMonth", "revenue", "orders", "units", "aov",
                                     "mom_growth", "rolling3", "partial"])
    m = (lines.groupby("orderMonth")
         .agg(revenue=("netRevenue", "sum"), orders=("orderID", "nunique"),
              units=("quantity", "sum"), discount=("discountAmount", "sum"))
         .reset_index().sort_values("orderMonth"))
    m["aov"] = m["revenue"] / m["orders"]
    m["mom_growth"] = m["revenue"].pct_change()
    m["rolling3"] = m["revenue"].rolling(3, min_periods=1).mean()
    # a month is "partial" if the filtered range starts/ends inside it
    first, last = lines["orderDate"].min(), lines["orderDate"].max()
    m["partial"] = False
    if len(m):
        if first.day > 3:
            m.loc[m.index[0], "partial"] = True
        if last.day < last.days_in_month - 3:
            m.loc[m.index[-1], "partial"] = True
    return m


def by_product(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    p = (lines.groupby(["productID", "productName", "categoryName", "discontinued"])
         .agg(revenue=("netRevenue", "sum"), gross=("grossSales", "sum"),
              discount=("discountAmount", "sum"), units=("quantity", "sum"),
              orders=("orderID", "nunique"), customers=("customerID", "nunique"),
              avg_price=("unitPrice", "mean"))
         .reset_index().sort_values("revenue", ascending=False))
    total = p["revenue"].sum()
    p["share"] = p["revenue"] / total if total else 0
    p["cum_share"] = p["share"].cumsum()
    p["rank"] = np.arange(1, len(p) + 1)
    p["discount_rate"] = np.where(p["gross"] > 0, p["discount"] / p["gross"], 0)
    return p


def by_category(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    c = (lines.groupby("categoryName")
         .agg(revenue=("netRevenue", "sum"), gross=("grossSales", "sum"),
              discount=("discountAmount", "sum"), units=("quantity", "sum"),
              orders=("orderID", "nunique"), products=("productID", "nunique"))
         .reset_index().sort_values("revenue", ascending=False))
    c["share"] = c["revenue"] / c["revenue"].sum()
    c["revenue_per_product"] = c["revenue"] / c["products"]
    c["discount_rate"] = c["discount"] / c["gross"]
    return c


def category_monthly(lines: pd.DataFrame) -> pd.DataFrame:
    return (lines.groupby(["orderMonth", "categoryName"])["netRevenue"].sum().reset_index())


def by_customer(lines: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    ref_date = ref_date or lines["orderDate"].max()
    c = (lines.groupby(["customerID", "customerName", "customerCountry", "customerCity"])
         .agg(revenue=("netRevenue", "sum"), orders=("orderID", "nunique"),
              units=("quantity", "sum"), discount=("discountAmount", "sum"),
              first_order=("orderDate", "min"), last_order=("orderDate", "max"))
         .reset_index().sort_values("revenue", ascending=False))
    c["aov"] = c["revenue"] / c["orders"]
    c["recency_days"] = (ref_date - c["last_order"]).dt.days
    c["share"] = c["revenue"] / c["revenue"].sum()
    c["cum_share"] = c["share"].cumsum()
    return c


def rfm(cust: pd.DataFrame) -> pd.DataFrame:
    """Quartile RFM scoring (4 = best) and rule-based segments."""
    if len(cust) < 8:
        return cust.assign(segment="n/a (too few customers)")
    c = cust.copy()
    c["R"] = pd.qcut(c["recency_days"].rank(method="first", ascending=False), 4, labels=[1, 2, 3, 4]).astype(int)
    c["F"] = pd.qcut(c["orders"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    c["M"] = pd.qcut(c["revenue"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    def seg(r):
        if r.R >= 3 and r.F >= 3 and r.M >= 3:
            return "Champions"
        if r.R >= 3 and r.F >= 2:
            return "Loyal / Active"
        if r.R >= 3:
            return "New / Promising"
        if r.M >= 3 or r.F >= 3:
            return "At risk (high value)"
        return "Lost / Low value"
    c["segment"] = c.apply(seg, axis=1)
    return c


def by_country(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    g = (lines.groupby("customerCountry")
         .agg(revenue=("netRevenue", "sum"), orders=("orderID", "nunique"),
              customers=("customerID", "nunique"), units=("quantity", "sum"))
         .reset_index().sort_values("revenue", ascending=False))
    g["share"] = g["revenue"] / g["revenue"].sum()
    g["revenue_per_customer"] = g["revenue"] / g["customers"]
    return g


def by_employee(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    e = (lines.groupby("employeeName")
         .agg(revenue=("netRevenue", "sum"), orders=("orderID", "nunique"),
              customers=("customerID", "nunique"), discount=("discountAmount", "sum"))
         .reset_index().sort_values("revenue", ascending=False))
    e["aov"] = e["revenue"] / e["orders"]
    return e


def by_shipper(ords: pd.DataFrame) -> pd.DataFrame:
    if ords.empty:
        return pd.DataFrame()
    s = ords.groupby("shipperName").agg(
        orders=("orderID", "count"), freight=("freight", "sum"),
        avg_freight=("freight", "mean"), avg_days=("daysToShip", "mean"),
        late_rate=("isLate", "mean")).reset_index()
    return s


def discount_buckets(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    d = lines.copy()
    d["discount_band"] = pd.cut(d["discount"], [-0.001, 0, 0.05, 0.10, 0.15, 0.20, 1.0],
                                labels=["None", "1-5%", "6-10%", "11-15%", "16-20%", ">20%"])
    return (d.groupby("discount_band", observed=True)
            .agg(lines=("orderID", "count"), avg_qty=("quantity", "mean"),
                 revenue=("netRevenue", "sum"), discount_cost=("discountAmount", "sum"))
            .reset_index())


# --------------------------------------------------------------------------- #
# Automated insights (all numbers computed from the filtered data)
# --------------------------------------------------------------------------- #
def _money(x):
    return f"${x:,.0f}"


def generate_insights(lines: pd.DataFrame, ords: pd.DataFrame):
    """Returns (findings, recommendations): lists of markdown strings."""
    if lines.empty:
        return ["No data for the current filter selection."], []
    f, r = [], []
    k = kpis(lines, ords)
    prod = by_product(lines)
    cat = by_category(lines)
    cust = by_customer(lines)
    ctry = by_country(lines)
    mon = monthly(lines)
    emp = by_employee(lines)
    ship = by_shipper(ords)

    # headline
    f.append(f"**Headline:** net revenue of **{_money(k['revenue'])}** from **{k['orders']:,}** orders "
             f"(average order value **{_money(k['aov'])}**). Discounts gave away **{_money(k['discount'])}** "
             f"({k['discount_rate']:.1%} of gross sales).")

    # trend
    full = mon[~mon["partial"]]
    if len(full) >= 12:
        recent, prior = full.tail(6)["revenue"].sum(), full.iloc[-12:-6]["revenue"].sum()
        chg = recent / prior - 1
        f.append(f"**Trend:** the last 6 complete months generated {_money(recent)}, "
                 f"{'up' if chg >= 0 else 'down'} **{abs(chg):.0%}** versus the 6 months before.")
        if chg > 0.1:
            r.append("Momentum is strong: protect it by checking stock cover on the top-10 products before peak months.")
    elif len(full) >= 2:
        chg = full["revenue"].iloc[-1] / full["revenue"].iloc[0] - 1
        f.append(f"**Trend:** revenue moved {chg:+.0%} between the first and last complete month in range.")
    if len(mon):
        best = mon.loc[mon["revenue"].idxmax()]
        f.append(f"**Peak month:** {best['orderMonth']:%B %Y} with {_money(best['revenue'])}.")
    if not mon.empty and mon["partial"].any():
        f.append("*Note: the first/last month in the selected range is partial and can look artificially low.*")

    # category
    if len(cat) > 1:
        top = cat.iloc[0]
        low = cat.iloc[-1]
        f.append(f"**Categories:** **{top['categoryName']}** leads with {top['share']:.0%} of revenue; "
                 f"**{low['categoryName']}** is smallest at {low['share']:.0%}.")
        lowest_eff = cat.sort_values("revenue_per_product").iloc[0]
        r.append(f"Review the range depth of **{lowest_eff['categoryName']}**: only "
                 f"{_money(lowest_eff['revenue_per_product'])} revenue per product, the lowest of any category.")

    # product concentration
    if len(prod) >= 5:
        top5 = prod.head(5)["share"].sum()
        n80 = int((prod["cum_share"] <= 0.8).sum()) + 1
        f.append(f"**Products:** the top 5 products ({', '.join(prod.head(5)['productName'])}) make up "
                 f"**{top5:.0%}** of revenue; **{min(n80, len(prod))} of {len(prod)}** products deliver 80%.")
        bottom = prod.tail(5)
        f.append(f"**Underperformers:** {', '.join(bottom['productName'])} together generate only "
                 f"{_money(bottom['revenue'].sum())} ({bottom['share'].sum():.1%}).")
        r.append(f"Run a keep / bundle / retire review on the bottom products ({', '.join(bottom['productName'].head(3))}…); "
                 "free the shelf space and working capital for the best sellers.")
        disc_prod = prod[prod["discontinued"] == 1]
        if not disc_prod.empty:
            f.append(f"**Discontinued items** still account for {_money(disc_prod['revenue'].sum())} "
                     f"({disc_prod['share'].sum():.1%}) of revenue in this selection.")
            r.append("Plan substitutes for discontinued products that still sell, so those customers are not lost.")

    # discounts
    db = discount_buckets(lines)
    if len(db) > 1:
        none = db[db["discount_band"] == "None"]
        disc = db[db["discount_band"] != "None"]
        if not none.empty and not disc.empty and disc["lines"].sum() > 0:
            q_none = none["avg_qty"].iloc[0]
            q_disc = (disc["avg_qty"] * disc["lines"]).sum() / disc["lines"].sum()
            f.append(f"**Discounting:** {k['lines_with_discount']:.0%} of order lines carry a discount. Discounted lines "
                     f"average **{q_disc:.1f}** units vs **{q_none:.1f}** for full-price lines.")
            if q_disc < q_none * 1.1:
                r.append("Discounts are not lifting basket size noticeably. Tighten discount rules or "
                         "tie them to minimum quantities.")
            else:
                r.append("Discounted lines do carry larger quantities; keep volume-linked discounts but cap the deepest tiers "
                         "and monitor the cost.")

    # customers
    if len(cust) >= 10:
        t10 = cust.head(10)["share"].sum()
        f.append(f"**Customers:** top 10 of {len(cust)} customers contribute **{t10:.0%}** of revenue. "
                 f"The largest is **{cust.iloc[0]['customerName']}** ({_money(cust.iloc[0]['revenue'])}).")
        if t10 > 0.4:
            r.append("Revenue is concentrated in a few accounts. Assign key-account owners to the top 10 "
                     "and build a retention plan.")
        rf = rfm(cust)
        if "R" in rf.columns:
            risk = rf[rf["segment"] == "At risk (high value)"]
            if len(risk):
                f.append(f"**Retention risk:** {len(risk)} high-value customers have not ordered recently "
                         f"(they account for {_money(risk['revenue'].sum())} of historic revenue).")
                r.append("Launch a win-back campaign aimed at the 'At risk (high value)' RFM segment "
                         "(see Customers tab for the list).")
        one = (cust["orders"] == 1).sum()
        if one:
            f.append(f"**Repeat business:** {one} customer(s) ({one/len(cust):.0%}) placed just one order; "
                     f"the average customer placed {cust['orders'].mean():.1f} orders.")
    if len(ctry) >= 3:
        f.append(f"**Geography:** {ctry.iloc[0]['customerCountry']} is the top market ({ctry.iloc[0]['share']:.0%}), "
                 f"followed by {ctry.iloc[1]['customerCountry']} ({ctry.iloc[1]['share']:.0%}) and "
                 f"{ctry.iloc[2]['customerCountry']} ({ctry.iloc[2]['share']:.0%}).")
        opp = ctry[(ctry["customers"] <= ctry["customers"].median())].sort_values("revenue_per_customer", ascending=False)
        if len(opp):
            r.append(f"Expansion opportunity: **{opp.iloc[0]['customerCountry']}** has only {int(opp.iloc[0]['customers'])} customer(s) but high revenue "
                     f"per customer ({_money(opp.iloc[0]['revenue_per_customer'])}). Small sample, so validate before investing, "
                     "but recruiting more accounts there looks promising.")

    # staff
    if len(emp) >= 3:
        f.append(f"**Sales team:** {emp.iloc[0]['employeeName']} leads with {_money(emp.iloc[0]['revenue'])}; "
                 f"{emp.iloc[-1]['employeeName']} is lowest at {_money(emp.iloc[-1]['revenue'])}.")

    # logistics
    if k["late_rate"] > 0 and not ship.empty:
        worst = ship.sort_values("late_rate", ascending=False).iloc[0]
        f.append(f"**Fulfilment:** {k['late_rate']:.1%} of shipped orders left after the required date; average "
                 f"{k['avg_days_ship']:.1f} days to ship. Freight is {k['freight_pct']:.1%} of revenue. "
                 f"**{worst['shipperName']}** has the highest late rate ({worst['late_rate']:.1%}).")
        if k["late_rate"] > 0.03:
            r.append(f"Address late shipments: review service levels with **{worst['shipperName']}** "
                     "and prioritise orders nearing their required date.")
    pending = int((~ords["isShipped"]).sum())
    if pending:
        f.append(f"**Open orders:** {pending} orders have no shipped date yet.")
        r.append("Chase the unshipped orders; check whether any are already past the required date.")

    return f, r


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        for name, df in sheets.items():
            df2 = df.copy()
            for c in df2.select_dtypes(include=["datetimetz"]).columns:
                df2[c] = df2[c].dt.tz_localize(None)
            df2.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, col in enumerate(df2.columns):
                if len(df2):
                    lens = df2[col].astype("object").map(lambda v: len(str(v)) if pd.notna(v) else 0)
                    body = int(np.nan_to_num(lens.quantile(0.9)))
                else:
                    body = 0
                ws.set_column(i, i, min(max(len(str(col)), body) + 2, 40))
            ws.freeze_panes(1, 0)
    return buf.getvalue()
