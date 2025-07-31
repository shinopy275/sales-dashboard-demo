# app.py  ─ Streamlit ダッシュボード（店舗・前年同月比較）
import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ──────────────────────────────────────
# 1. ヘルパ関数
# ──────────────────────────────────────
def get_store_name(fname: str) -> str:
    """
    例:
      01.盛岡店 12月 売上管理表.xlsx  → '盛岡店'
      03_仙台店 12月 売上管理表.xlsx       → '仙台店'
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
    all_rows = []
    for f in files:
        try:
            store = get_store_name(f.name)
        except ValueError as e:
            st.warning(str(e))
            continue

        df = pd.read_excel(f, sheet_name="売上管理", engine="openpyxl")

        try:
            year, month = infer_year_month(df)
        except ValueError as e:
            st.warning(f"{f.name}: {e}")
            continue

        df["店舗名"] = store
        df["年"] = year
        df["月"] = month
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ──────────────────────────────────────
# 2. ファイルアップロード
# ──────────────────────────────────────
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

# ──────────────────────────────────────
# 3. 月次集計 & 前年同月比較
#    ※ 列名はご利用の Excel に合わせて変更してください
# ──────────────────────────────────────
AGG_COLS = {"総売上": "sum", "総来院数": "sum"}

monthly = (
    df_all.groupby(["店舗名", "年", "月"], as_index=False)
    .agg(AGG_COLS)
    .sort_values(["店舗名", "年", "月"])
)

latest_year = monthly["年"].max()
prev_year = latest_year - 1

this_year = monthly[monthly["年"] == latest_year]
prev_year_df = monthly[monthly["年"] == prev_year]

comp = pd.merge(
    this_year,
    prev_year_df,
    on=["店舗名", "月"],
    how="left",
    suffixes=("_今年", "_前年"),
)

for col in ["総売上", "総来院数"]:
    comp[f"{col}増減率%"] = (
        (comp[f"{col}_今年"] - comp[f"{col}_前年"])
        / comp[f"{col}_前年"].replace(0, pd.NA)
        * 100
    ).round(1)

# ──────────────────────────────────────
# 4. 全店舗サマリー表示
# ──────────────────────────────────────
st.subheader(f"📊 全店舗サマリー（{latest_year}年 vs {prev_year}年）")
st.dataframe(
    comp[
        [
            "店舗名",
            "月",
            "総売上_前年",
            "総売上_今年",
            "総売上増減率%",
            "総来院数_前年",
            "総来院数_今年",
            "総来院数増減率%",
        ]
    ],
    use_container_width=True,
)

# ──────────────────────────────────────
# 5. 店舗別ダッシュボード
# ──────────────────────────────────────
store = st.selectbox("🔍 店舗を選択", sorted(comp["店舗名"].unique()))
st.markdown("---")
st.header(f"🏪 {store} の詳細")

ss = comp[comp["店舗名"] == store].sort_values("月")
current = ss.iloc[-1]

k1, k2 = st.columns(2)
k1.metric("売上 前年比", f"{current['総売上増減率%']} %")
k2.metric("来院数 前年比", f"{current['総来院数増減率%']} %")

# 売上グラフ
fig = px.bar(
    ss,
    x="月",
    y=["総売上_前年", "総売上_今年"],
    title=f"{store} 月別総売上（前年 vs 今年）",
    labels={"value": "金額", "variable": "年度"},
    barmode="group",
)
st.plotly_chart(fig, use_container_width=True)

# 来院数グラフ
fig2 = px.bar(
    ss,
    x="月",
    y=["総来院数_前年", "総来院数_今年"],
    title=f"{store} 月別来院数（前年 vs 今年）",
    labels={"value": "人数", "variable": "年度"},
    barmode="group",
)
st.plotly_chart(fig2, use_container_width=True)

# 元データ確認（オプション）
with st.expander("📄 元データを見る"):
    st.dataframe(ss, use_container_width=True)
