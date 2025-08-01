import streamlit as st
import pandas as pd
import altair as alt
import re, math
from typing import List, Tuple

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ───────── ヘルパ ─────────

def get_store_name(fname: str) -> str:
    m = re.match(r"\d+[._](.+?)\s", fname)
    if not m:
        raise ValueError("店舗名が判別できません")
    return m.group(1).strip()


def infer_year_month(df: pd.DataFrame) -> Tuple[int, int]:
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    day = df["日付"].dropna()
    if day.empty:
        raise ValueError("日付列がありません")
    return int(day.dt.year.mode()[0]), int(day.dt.month.mode()[0])

# ───── 患者分析シート ─────
def parse_patient_analysis(f):
    """患者分析シートを抽出。無い場合は 0 データで返す"""
    try:
        xls = pd.ExcelFile(f, engine="openpyxl")
        if "患者分析" not in xls.sheet_names:
            raise ValueError("シートなし")
        sheet = xls.parse("患者分析", header=None)
    except Exception:
        import streamlit as st
        st.warning(f"{f.name}: 患者分析シートが見つかりません - 0 件として処理します")
        # シートが無ければすべて 0
        zero = lambda cats: pd.DataFrame({"カテゴリ": cats, "件数": [0]*len(cats)})
        return (
            zero(["男性", "女性"]),
            zero(["チラシ", "紹介", "看板", "ネット", "その他"]),
            zero(["10代未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90歳以上"]),
        )

    def grab(keyword: str, rng: slice | None, cats: list[str]):
        mask = sheet.apply(lambda r: r.astype(str).str.contains(keyword).any(), axis=1)
        if not mask.any():
            return pd.DataFrame({"カテゴリ": cats, "件数": [0]*len(cats)})
        r = mask.idxmax()
        header = sheet.iloc[r + 1]
        vals   = sheet.iloc[r + 2]
        if rng is not None:
            header = header.iloc[rng]
            vals   = vals.iloc[rng]
        header = header.dropna()
        if header.empty:
            return pd.DataFrame({"カテゴリ": cats, "件数": [0]*len(cats)})
        data = pd.to_numeric(vals[header.index], errors="coerce").fillna(0)
        return pd.DataFrame({"カテゴリ": header.values, "件数": data.values})

    gender = grab("男女比率",  slice(0, 2),  ["男性", "女性"])
    reason = grab("来院動機", slice(5, 10), ["チラシ", "紹介", "看板", "ネット", "その他"])
    age    = grab("年齢比率", None,        ["10代未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90歳以上"])
    return gender, reason, age

# ───── LTV ─────

def parse_ltv(f):
    df = pd.read_excel(f, sheet_name="店舗分析", header=None, engine="openpyxl")
    mask = df.apply(lambda r: r.astype(str).str.contains("LTV").any(), axis=1)
    if not mask.any():
        return None
    num = pd.to_numeric(df[mask].iloc[0], errors="coerce").dropna()
    return float(num.iloc[0]) if not num.empty else None

# ───── Excel 読込 ─────

@st.cache_data(show_spinner=False)
def load(files):
    sales, reasons, genders, ages, ltvs = [], [], [], [], []
    for f in files:
        try:
            store = get_store_name(f.name)
        except ValueError as e:
            st.warning(str(e)); continue

        try:
            df_sales = pd.read_excel(f, sheet_name="売上管理", header=4, engine="openpyxl")
        except Exception as e:
            st.warning(f"{f.name}: 売上管理読み込み失敗 ({e})"); continue

        for col in ("総売上", "総来院数"):
            if col in df_sales.columns:
                df_sales[col] = pd.to_numeric(df_sales[col], errors="coerce").fillna(0)

        try:
            y, m = infer_year_month(df_sales)
        except ValueError as e:
            st.warning(f"{f.name}: {e}"); continue

        df_sales["店舗名"], df_sales["年"], df_sales["月"] = store, y, m
        sales.append(df_sales)

        g, r, a = parse_patient_analysis(f)
        for df_, lst in [(g, genders), (r, reasons), (a, ages)]:
            if not df_.empty:
                df_["店舗名"], df_["年"], df_["月"] = store, y, m
                lst.append(df_)

        val = parse_ltv(f)
        if val is not None:
            ltvs.append({"店舗名": store, "年": y, "月": m, "LTV": val})

    out = lambda lst: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
    return out(sales), out(reasons), out(genders), out(ages), pd.DataFrame(ltvs)

# ───── ファイル選択 ─────

files = st.file_uploader("📂 Excel ファイルを選択（複数可）", type="xlsx", accept_multiple_files=True)
if not files:
    st.stop()

sales_df, reason_df, gender_df, age_df, ltv_df = load(files)
if sales_df.empty:
    st.error("売上管理シートが読み込めませんでした"); st.stop()

# ───── 全店舗サマリー ─────

monthly = sales_df.groupby(["店舗名", "年", "月"], as_index=False)[["総売上", "総来院数"]].sum()
latest, prev = monthly["年"].max(), monthly["年"].max() - 1
cur, old = monthly[monthly["年"] == latest], monthly[monthly["年"] == prev]
comp = pd.merge(cur, old, on=["店舗名", "月"], how="left", suffixes=("_今年", "_前年"))
for k in ("総売上", "総来院数"):
    comp[f"{k}増減率%"] = ((comp[f"{k}_今年"] - comp[f"{k}_前年"]) / comp[f"{k}_前年"].replace(0, pd.NA) * 100).round(1)

if not ltv_df.empty:
    ltv_cur = ltv_df[ltv_df["年"] == latest]
    ltv_old = ltv_df[ltv_df["年"] == prev]
    ltv_c   = pd.merge(ltv_cur, ltv_old, on=["店舗名", "月"], how="left", suffixes=("_今年", "_前年"))
    ltv_c["LTV増減率%"] = ((ltv_c["LTV_今年"] - ltv_c["LTV_前年"]) / ltv_c["LTV_前年"].replace(0, pd.NA) * 100).round(1)
    comp = pd.merge(comp, ltv_c[["店舗名", "月", "LTV_前年", "LTV_今年", "LTV増減率%"]], on=["店舗名", "月"], how="left")

num_cols = [c for c in comp.columns if any(k in c for k in ("総売上_", "総来院数_", "LTV_"))]
month_total_rows = []
for m in sorted(comp["月"].unique()):
    sub = comp[comp["月"] == m]
    d = {"店舗名": "月合計", "月": m}
    for c in num_cols:
        d[c] = sub[c].sum()
    for k, rate in [("総売上", "総売上増減率%"), ("総来院数", "総来院数増減率%"), ("LTV", "LTV増減率%")]:
        if d.get(f"{k}_前年", 0):
            d[rate] = ((d[f"{k}_今年"] - d[f"{k}_前年"]) / d[f"{k}_前年"] * 100).round(1)
    month_total_rows.append(d)

comp_all = pd.concat([comp, pd.DataFrame(month_total_rows)], ignore_index=True)

if month_total_rows:
    grand = {"店舗名": "月合計", "月": "総計"}
    mt_df = pd.DataFrame(month_total_rows)
    for c in num_cols:
        grand[c] = mt_df[c].sum()
    for k, rate in [("総売上", "総売上増減率%"), ("総来院数", "総来院数増減率%"), ("LTV", "LTV増減率%")]:
        if grand.get(f"{k}_前年", 0):
            grand[rate] = ((grand[f"{k}_今年"] - grand[f"{k}_前年"]) / grand[f"{k}_前年"] * 100).round(1)
    comp_all = pd.concat([comp_all, pd.DataFrame([grand])], ignore_index=True)

st.subheader(f"📊 全店舗サマリー（{latest}年 vs {prev}年）")
show = ["店舗名", "月", "総売上_前年", "総売上_今年", "総売上増減率%", "総来院数_前年", "総来院数_今年", "総来院数増減率%", "LTV_前年", "LTV_今年", "LTV増減率%"]
st.dataframe(comp_all[show], use_container_width=True)

# ───── 店舗別 ─────

stores = sorted(comp["店舗名"].unique())
store = st.selectbox("店舗選択", stores)
ss = comp[comp["店舗名"] == store].sort_values("月")

sum_row = ss[["総売上_前年", "総売上_今年", "総来院数_前年", "総来院数_今年"]].sum()
col1, col2 = st.columns(2)
col1.metric("売上 前年比", f"{((sum_row['総売上_今年']-sum_row['総売上_前年'])/sum_row['総売上_前年']*100).round(1)} %")
col2.metric("来院数 前年比", f"{((sum_row['総来院数_今年']-sum_row['総来院数_前年'])/sum_row['総来院数_前年']*100).round(1)} %")

col3, col4 = st.columns(2)
col3.metric("売上 (今年)", f"{int(sum_row['総売上_今年']):,} 円")
col4.metric("来院数 (今年)", f"{int(sum_row['総来院数_今年']):,} 人")

ltv_now = ltv_df.query("店舗名 == @store & 年 == @latest")['LTV'].mean()
ltv_before = ltv_df.query("店舗名 == @store & 年 == @prev")['LTV'].mean()
if not math.isnan(ltv_now):
    c5, c6 = st.columns(2)
    c5.metric("LTV (今年)", f"{ltv_now:,.0f} 円")
    if ltv_before and not math.isnan(ltv_before):
        c6.metric("LTV 前年比", f"{((ltv_now-ltv_before)/ltv_before*100).round(1):+.1f} %")

full_m = pd.DataFrame({"月": range(1, 13)})
ss_full = full_m.merge(ss, on="月", how="left").fillna(0)
ss_full = ss_full[(ss_full["総売上_前年"] != 0) | (ss_full["総売上_今年"] != 0)]

for label, cols, ttl, ycap in [
    ("売上", ["総売上_前年", "総売上_今年"], "月別総売上", "金額(万円)"),
    ("来院数", ["総来院数_前年", "総来院数_今年"], "月別来院数", "人数")]:
    plot = ss_full.melt(id_vars="月", value_vars=cols, var_name="年度", value_name=label).replace({cols[0]: prev, cols[1]: latest})
    if label == "売上":
        plot[label] /= 10000
    plot[["月", "年度"]] = plot[["月", "年度"]].astype(str)
    st.altair_chart(
        alt.Chart(plot).mark_bar().encode(
            x=alt.X("月:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{label}:Q", title=ycap),
            xOffset="年度:N", color="年度:N",
            tooltip=["年度", "月", label],
        ).properties(width=400, height=300, title=f"{store} {ttl} (前年 vs 今年)"),
        use_container_width=True,
    )

# 患者分析プロット

def plot_pivot(df_src, title):
    df = df_src.query("店舗名 == @store & 年 == @latest").groupby("カテゴリ", as_index=False)["件数"].sum()
    if df.empty:
        return
    df["割合%"] = (df["件数"] / df["件数"].sum() * 100).round(1)
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            y=alt.Y("カテゴリ:N", sort="-x"),
            x="件数:Q", tooltip=["カテゴリ", "件数", "割合%"],
        ).properties(width=350, height=250, title=title), use_container_width=True,
    )
    with st.expander(f"📄 {title} 明細"):
        st.dataframe(df, use_container_width=True)

plot_pivot(reason_df, "来店動機")
plot_pivot(gender_df, "男女比率")
plot_pivot(age_df,    "年齢比率")

with st.expander("📄 月別比較データ（店舗）"):
    st.dataframe(ss_full, use_container_width=True)
