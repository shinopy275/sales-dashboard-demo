# app.py – Streamlit ダッシュボード（Altair 版）
import streamlit as st
import pandas as pd
import altair as alt
import re

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ---------- 1. ヘルパ ----------
def get_store_name(fname: str) -> str:
    m = re.match(r"\d+[._](.+?)\s", fname)
    if not m:
        raise ValueError(f"店舗名を取得できません: {fname}")
    return m.group(1).strip()

def infer_year_month(df: pd.DataFrame) -> tuple[int, int]:
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    valid = df["日付"].dropna()
    if valid.empty:
        raise ValueError("日付列が解析できません")
    return int(valid.dt.year.mode()[0]), int(valid.dt.month.mode()[0])

@st.cache_data(show_spinner=False)
def load_files(files):
    NUMERIC = ["総売上", "総来院数"]
    rows = []
    for f in files:
        try:
            store = get_store_name(f.name)
        except ValueError as e:
            st.warning(str(e)); continue

        df = pd.read_excel(f, sheet_name="売上管理",
                           engine="openpyxl", header=4)
        for col in NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        try:
            y, m = infer_year_month(df)
        except ValueError as e:
            st.warning(f"{f.name}: {e}"); continue

        df["店舗名"], df["年"], df["月"] = store, y, m
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ---------- 2. ファイルアップロード ----------
files = st.file_uploader(
    "📂 複数店舗の Excel ファイルを選択（複数選択可）",
    type="xlsx", accept_multiple_files=True,
)
if not files:
    st.info("ファイルをアップロードするとダッシュボードが表示されます。")
    st.stop()

df_all = load_files(files)
if df_all.empty:
    st.error("有効なファイルが読み込めませんでした。")
    st.stop()

# ---------- 3. 月次集計 ----------
AGG = {"総売上": "sum", "総来院数": "sum"}
monthly = (df_all.groupby(["店舗名", "年", "月"], as_index=False)
                 .agg(AGG).sort_values(["店舗名", "年", "月"]))

latest_year = monthly["年"].max()
prev_year   = latest_year - 1

this_year   = monthly[monthly["年"] == latest_year]
prev_year_df= monthly[monthly["年"] == prev_year]

comp = pd.merge(this_year, prev_year_df,
                on=["店舗名", "月"], how="left",
                suffixes=("_今年", "_前年"))

for c in ["総売上", "総来院数"]:
    comp[f"{c}増減率%"] = ((comp[f"{c}_今年"] - comp[f"{c}_前年"]) /
                           comp[f"{c}_前年"].replace(0, pd.NA) * 100).round(1)

# ---------- 4. 全店舗サマリー ----------
st.subheader(f"📊 全店舗サマリー（{latest_year}年 vs {prev_year}年）")
st.dataframe(comp[[
        "店舗名", "月",
        "総売上_前年", "総売上_今年", "総売上増減率%",
        "総来院数_前年", "総来院数_今年", "総来院数増減率%",
    ]], use_container_width=True)

# ---------- 5. 店舗別ダッシュボード ----------
store = st.selectbox("🔍 店舗を選択", sorted(comp["店舗名"].unique()))
st.markdown("---")
st.header(f"🏪 {store} の詳細")

ss = comp[comp["店舗名"] == store].sort_values("月")

# 5-1 KPI
tot = ss[["総売上_前年","総売上_今年","総来院数_前年","総来院数_今年"]].sum()
sales_rate = ((tot["総売上_今年"] - tot["総売上_前年"])/tot["総売上_前年"]*100).round(1)
visit_rate = ((tot["総来院数_今年"] - tot["総来院数_前年"])/tot["総来院数_前年"]*100).round(1)
c1, c2 = st.columns(2)
c1.metric("売上 前年比(累計)",  f"{sales_rate} %")
c2.metric("来院数 前年比(累計)", f"{visit_rate} %")
c3, c4 = st.columns(2)
c3.metric("売上 (今年)",  f"{int(tot['総売上_今年']):,} 円")
c4.metric("来院数 (今年)", f"{int(tot['総来院数_今年']):,} 人")

# 5-2 月フルリスト補完
full_months = pd.DataFrame({"月": range(1, 13)})
ss_full = full_months.merge(ss, on="月", how="left").fillna(0)
mask = (ss_full["総売上_前年"] != 0) | (ss_full["総売上_今年"] != 0)
ss_full = ss_full[mask]

# ---------- 5-3 Altair 売上グラフ ----------
sales_plot = (ss_full.melt(id_vars="月",
                           value_vars=["総売上_前年","総売上_今年"],
                           var_name="年度", value_name="売上")
                     .replace({"総売上_前年": prev_year,
                               "総売上_今年": latest_year}))
sales_plot["売上"] /= 10_000   # 円→万円
sales_plot[["月","年度"]] = sales_plot[["月","年度"]].astype(str)

sales_chart = (
    alt.Chart(sales_plot)
        .mark_bar()
        .encode(
            x=alt.X("月:N", title="月", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("売上:Q", title="金額 (万円)"),
            xOffset="年度:N",                      # ← 横並び
            color=alt.Color("年度:N", title="年度"),
            tooltip=["年度", "月", "売上"]
        )
        .properties(width=400, height=300,
                    title=f"{store} 月別総売上（前年 vs 今年）")
)

st.altair_chart(sales_chart, use_container_width=True)

# ---------- 5-4 Altair 来院数グラフ ----------
visit_plot = (ss_full.melt(id_vars="月",
                           value_vars=["総来院数_前年","総来院数_今年"],
                           var_name="年度", value_name="来院数")
                     .replace({"総来院数_前年": prev_year,
                               "総来院数_今年": latest_year}))
visit_plot[["月","年度"]] = visit_plot[["月","年度"]].astype(str)

visit_chart = (
    alt.Chart(visit_plot)
        .mark_bar()
        .encode(
            x=alt.X("月:N", title="月", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("来院数:Q", title="人数"),
            xOffset="年度:N",
            color=alt.Color("年度:N", title="年度"),
            tooltip=["年度", "月", "来院数"]
        )
        .properties(width=400, height=300,
                    title=f"{store} 月別来院数（前年 vs 今年）")
)

st.altair_chart(visit_chart, use_container_width=True)

# ---------- 5-5 元データ表示 ----------
with st.expander("📄 月別比較データ（店舗）"):
    st.dataframe(ss_full, use_container_width=True)
