# app.py ─ Streamlit ダッシュボード（前年同月比較・月次累計）
import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ──────────────────────────────
# 1. ヘルパ関数
# ──────────────────────────────
def get_store_name(fname: str) -> str:
    """
    01.盛岡店 12月 売上管理表.xlsx → '盛岡店'
    03_仙台店 12月 売上管理表.xlsx → '仙台店'
    """
    m = re.match(r"\d+[._](.+?)\s", fname)
    if not m:
        raise ValueError(f"店舗名を取得できません: {fname}")
    return m.group(1).strip()


def infer_year_month(df: pd.DataFrame) -> tuple[int, int]:
    """DataFrame の '日付' 列から (year, month) を推定"""
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    valid = df["日付"].dropna()
    if valid.empty:
        raise ValueError("日付列が解析できません")
    year = int(valid.dt.year.mode()[0])
    month = int(valid.dt.month.mode()[0])
    return year, month


@st.cache_data(show_spinner=False)
def load_files(files):
    """複数ファイルを読み込み、店舗名・年・月を付与して結合"""
    NUMERIC_COLS = ["総売上", "総来院数"]  # ← 集計する数値列を列名に合わせて
    rows = []

    for f in files:
        # 店舗名抽出
        try:
            store = get_store_name(f.name)
        except ValueError as e:
            st.warning(str(e)); continue

        # Excel 読み込み（header 行は 4 → 0-index で 4 == 5 行目）
        df = pd.read_excel(f, sheet_name="売上管理", engine="openpyxl", header=4)

        # 数値列を float に統一
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 年・月推定
        try:
            year, month = infer_year_month(df)
        except ValueError as e:
            st.warning(f"{f.name}: {e}"); continue

        df["店舗名"], df["年"], df["月"] = store, year, month
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ──────────────────────────────
# 2. ファイルアップロード
# ──────────────────────────────
files = st.file_uploader(
    "📂 複数店舗の Excel ファイルを選択（複数選択可）",
    type="xlsx",
    accept_multiple_files=True,
)

if not files:
    st.info("ファイルをアップロードするとダッシュボードが表示されます。")
    st.stop()

df_all = load_files(files)
if df_all.empty:
    st.error("有効なファイルが読み込めませんでした。")
    st.stop()

# ──────────────────────────────
# 3. 月次集計 & 前年同月比較
# ──────────────────────────────
AGG_COLS = {"総売上": "sum", "総来院数": "sum"}

monthly = (
    df_all.groupby(["店舗名", "年", "月"], as_index=False)
          .agg(AGG_COLS)
          .sort_values(["店舗名", "年", "月"])
)

latest_year = monthly["年"].max()
prev_year   = latest_year - 1

this_year   = monthly[monthly["年"] == latest_year]
prev_year_df= monthly[monthly["年"] == prev_year]

comp = pd.merge(
    this_year, prev_year_df,
    on=["店舗名", "月"], how="left",
    suffixes=("_今年", "_前年")
)

for c in ["総売上", "総来院数"]:
    comp[f"{c}増減率%"] = (
        (comp[f"{c}_今年"] - comp[f"{c}_前年"])
        / comp[f"{c}_前年"].replace(0, pd.NA) * 100
    ).round(1)

# ──────────────────────────────
# 4. 全店舗サマリー
# ──────────────────────────────
st.subheader(f"📊 全店舗サマリー（{latest_year}年 vs {prev_year}年）")
st.dataframe(
    comp[[
        "店舗名", "月",
        "総売上_前年", "総売上_今年", "総売上増減率%",
        "総来院数_前年", "総来院数_今年", "総来院数増減率%",
    ]],
    use_container_width=True,
)

# ──────────────────────────────
# 5. 店舗別ダッシュボード
# ──────────────────────────────
store = st.selectbox("🔍 店舗を選択", sorted(comp["店舗名"].unique()))
st.markdown("---")
st.header(f"🏪 {store} の詳細")

ss = comp[comp["店舗名"] == store].sort_values("月")

# ---------- 5.1 KPI：年累計 ----------
kpi_tot = ss[[
    "総売上_前年", "総売上_今年",
    "総来院数_前年", "総来院数_今年"
]].sum()

sales_rate = ((kpi_tot["総売上_今年"] - kpi_tot["総売上_前年"])
              / kpi_tot["総売上_前年"] * 100).round(1)
visit_rate = ((kpi_tot["総来院数_今年"] - kpi_tot["総来院数_前年"])
              / kpi_tot["総来院数_前年"] * 100).round(1)

# KPI 算出後に追加
total_sales = int(kpi_tot["総売上_今年"])
total_visits = int(kpi_tot["総来院数_今年"])

c1, c2 = st.columns(2)
c1.metric("売上 前年比(累計)",  f"{sales_rate} %")
c2.metric("来院数 前年比(累計)", f"{visit_rate} %")

c3,c4 = st.columns(2)
c3.metric("売上 (今年)",f"{total_sales:,} 円")      ### ←追加
c4.metric("来院数 (今年)",f"{total_visits:,} 人")   ### ←追加

# ---------- 5.2 月フルリスト補完 ----------
full_months = pd.DataFrame({"月": range(1, 13)})
ss_full = full_months.merge(ss, on="月", how="left").fillna(0)

# ★ データが両方 0 の月を除外
mask = (ss_full["総売上_前年"] != 0) | (ss_full["総売上_今年"] != 0)
ss_full = ss_full[mask]

# ---------- 売上グラフ ----------
# ---------- 売上グラフ（完全版・これだけ残す） ----------
sales_plot = (
    ss_full.melt(
        id_vars="月",
        value_vars=["総売上_前年","総売上_今年"],
        var_name="年度", value_name="売上"
    )
)
sales_plot["年度"] = sales_plot["年度"].map({
    "総売上_前年": prev_year, "総売上_今年": latest_year
}).astype(str)
sales_plot["売上"] = pd.to_numeric(sales_plot["売上"])/10_000      # ← ここ以外で除算しない
sales_plot["月"]   = sales_plot["月"].astype(str)

fig = px.bar(
    sales_plot, x="月", y="売上",
    color="年度", barmode="group",
    title=f"{store} 月別総売上（前年 vs 今年）",
    labels={"売上":"金額 (万円)"}
)
# ① ここで軸タイプを明示                          ▼これを追加
fig.update_yaxes(type="linear", rangemode="tozero", tickformat=",.0f")

fig.update_layout(bargap=0.15, bargroupgap=0.05)  # 棒幅だけ
fig.update_yaxes(rangemode="tozero", tickformat=",.0f")  # y軸は自動レンジ

ymax = sales_plot["売上"].max()
fig.update_yaxes(range=[0, ymax * 1.2])  # ← ここを仮で入れる

st.plotly_chart(fig, use_container_width=True, key="sales")   # ★ key を付ける
st.write("axis type:", fig.layout.yaxis.type)   # ← 'category' なら原因確定
# ---------- 5.4 来院数グラフ ----------
visit_plot = (
    ss_full.melt(id_vars="月",
                 value_vars=["総来院数_前年", "総来院数_今年"],
                 var_name="年度", value_name="来院数")
      .replace({"総来院数_前年": prev_year, "総来院数_今年": latest_year})
)
visit_plot[["月","年度"]] = visit_plot[["月","年度"]].astype(str)

fig2 = px.bar(
    visit_plot, x="月", y="来院数",
    color="年度",
    title=f"{store} 月別来院数（前年 vs 今年）",
    labels={"来院数":"人数", "月":"月", "年度":"年"}
)
fig2.update_xaxes(type="category",
                  categoryorder="array",
                  categoryarray=[str(i) for i in range(1, 13)])
fig2.update_traces(width=0.35)
fig2.update_yaxes(tickformat=",")
st.plotly_chart(fig2, use_container_width=True, key="visit")

# ---------- 5.5 デバッグ用表示（任意） ----------
with st.expander("📄 月別比較データ（店舗）"):
    st.dataframe(ss_full, use_container_width=True)

