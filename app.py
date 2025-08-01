import streamlit as st
import pandas as pd
import altair as alt
import re
import math
from typing import List, Tuple

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ---------- 1. ヘルパ ----------

def get_store_name(fname: str) -> str:
    """ファイル名 (例: 01.札幌店 2025年01月 売上管理表.xlsx) から店舗名を抽出"""
    m = re.match(r"\d+[._](.+?)\s", fname)
    if not m:
        raise ValueError(f"店舗名を取得できません: {fname}")
    return m.group(1).strip()


def infer_year_month(df: pd.DataFrame) -> Tuple[int, int]:
    """日付列からファイルの年・月を推定"""
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    valid = df["日付"].dropna()
    if valid.empty:
        raise ValueError("日付列が解析できません")
    return int(valid.dt.year.mode()[0]), int(valid.dt.month.mode()[0])


# ---------- 2. 追加パース関数 ----------

def parse_visit_reason(f) -> pd.DataFrame:
    """患者分析シート → 来店動機カテゴリごとの件数を長テーブルで返す"""
    df_raw = pd.read_excel(
        f, sheet_name="患者分析", header=None, engine="openpyxl"
    )

    # 『来院動機』を含む行を検索
    idx = (
        df_raw.apply(lambda r: r.astype(str).str.contains("来院動機").any(), axis=1)
        .idxmax()
    )

    # 行が見つからなければ空 DataFrame
    if pd.isna(idx):
        return pd.DataFrame()

    cat_row = idx + 1  # カテゴリ名が横並び
    val_row = idx + 2  # 件数が横並び (レイアウトにより調整ください)

    cats = df_raw.loc[cat_row].dropna()
    vals = df_raw.loc[val_row, cats.index]

    # 数値化
    vals = pd.to_numeric(vals, errors="coerce").fillna(0)

    out = pd.DataFrame({"カテゴリ": cats.values, "件数": vals.values})
    return out


def parse_ltv(f) -> float:
    """店舗分析シート → LTV (1値) 取得"""
    df_raw = pd.read_excel(
        f, sheet_name="店舗分析", header=None, engine="openpyxl"
    )

    mask = df_raw.apply(lambda r: r.astype(str).str.contains("LTV").any(), axis=1)
    rows = df_raw[mask]
    if rows.empty:
        return None

    # 最初に見つかった数値を LTV とみなす
    numeric = pd.to_numeric(rows.iloc[0], errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


# ---------- 3. ファイル読込 ----------

@st.cache_data(show_spinner=False)
def load_files(files) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Excel 一式を読み込み、売上・来店動機・LTV の3テーブルを返す"""

    NUMERIC_COLS = ["総売上", "総来院数"]

    sales_rows: List[pd.DataFrame] = []
    reasons_rows: List[pd.DataFrame] = []
    ltv_rows: List[dict] = []

    for f in files:
        try:
            store = get_store_name(f.name)
        except ValueError as e:
            st.warning(str(e))
            continue

        # ---------- 売上管理 ----------
        try:
            df_sales = pd.read_excel(
                f, sheet_name="売上管理", header=4, engine="openpyxl"
            )
        except Exception as e:
            st.warning(f"{f.name}: 売上管理シートが読み込めません ({e})")
            continue

        # 数値列型変換
        for col in NUMERIC_COLS:
            if col in df_sales.columns:
                df_sales[col] = pd.to_numeric(df_sales[col], errors="coerce").fillna(0)

        try:
            y, m = infer_year_month(df_sales)
        except ValueError as e:
            st.warning(f"{f.name}: {e}")
            continue

        df_sales["店舗名"], df_sales["年"], df_sales["月"] = store, y, m
        sales_rows.append(df_sales)

        # ---------- 来店動機 ----------
        try:
            df_reason = parse_visit_reason(f)
            if not df_reason.empty:
                df_reason["店舗名"], df_reason["年"], df_reason["月"] = store, y, m
                reasons_rows.append(df_reason)
        except Exception as e:
            st.warning(f"{f.name} 来店動機: {e}")

        # ---------- LTV ----------
        try:
            ltv_val = parse_ltv(f)
            if ltv_val is not None:
                ltv_rows.append({"店舗名": store, "年": y, "月": m, "LTV": ltv_val})
        except Exception as e:
            st.warning(f"{f.name} LTV: {e}")

    # concat
    df_sales_all = pd.concat(sales_rows, ignore_index=True) if sales_rows else pd.DataFrame()
    df_reason_all = pd.concat(reasons_rows, ignore_index=True) if reasons_rows else pd.DataFrame()
    df_ltv_all = pd.DataFrame(ltv_rows)

    return df_sales_all, df_reason_all, df_ltv_all


# ---------- 4. ファイルアップロード ----------

files = st.file_uploader(
    "📂 複数店舗の Excel ファイルを選択（複数選択可）",
    type="xlsx",
    accept_multiple_files=True,
)

if not files:
    st.info("ファイルをアップロードするとダッシュボードが表示されます。")
    st.stop()

# ここで3テーブル取得
sales_df, reason_df, ltv_df = load_files(files)

if sales_df.empty:
    st.error("有効なファイルが読み込めませんでした。")
    st.stop()

# ---------- 5. 月次集計 (売上・来院数) ----------

AGG = {"総売上": "sum", "総来院数": "sum"}
monthly = (
    sales_df.groupby(["店舗名", "年", "月"], as_index=False)
    .agg(AGG)
    .sort_values(["店舗名", "年", "月"])
)

latest_year = monthly["年"].max()
prev_year = latest_year - 1

this_year = monthly[monthly["年"] == latest_year]
prev_year_df = monthly[monthly["年"] == prev_year]

comp = pd.merge(
    this_year, prev_year_df, on=["店舗名", "月"], how="left", suffixes=("_今年", "_前年")
)

for c in ["総売上", "総来院数"]:
    comp[f"{c}増減率%"] = (
        (comp[f"{c}_今年"] - comp[f"{c}_前年"]) / comp[f"{c}_前年"].replace(0, pd.NA) * 100
    ).round(1)

# ---------- 6. 全店舗サマリー ----------

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

# ---------- 7. 店舗別ダッシュボード ----------

store = st.selectbox("🔍 店舗を選択", sorted(comp["店舗名"].unique()))
st.markdown("---")
st.header(f"🏪 {store} の詳細")

ss = comp[comp["店舗名"] == store].sort_values("月")

# 7-1 KPI

tot = ss[["総売上_前年", "総売上_今年", "総来院数_前年", "総来院数_今年"]].sum()
sales_rate = (
    (tot["総売上_今年"] - tot["総売上_前年"]) / tot["総売上_前年"] * 100
).round(1)
visit_rate = (
    (tot["総来院数_今年"] - tot["総来院数_前年"]) / tot["総来院数_前年"] * 100
).round(1)

c1, c2 = st.columns(2)
c1.metric("売上 前年比(累計)", f"{sales_rate} %")
c2.metric("来院数 前年比(累計)", f"{visit_rate} %")

c3, c4 = st.columns(2)
c3.metric("売上 (今年)", f"{int(tot['総売上_今年']):,} 円")
c4.metric("来院数 (今年)", f"{int(tot['総来院数_今年']):,} 人")

# ----- LTV KPI -----
ltv_val = (
    ltv_df.query("店舗名 == @store & 年 == @latest_year")["LTV"].mean()
    if not ltv_df.empty
    else float("nan")
)
ltv_prev = (
    ltv_df.query("店舗名 == @store & 年 == @prev_year")["LTV"].mean()
    if not ltv_df.empty
    else float("nan")
)

if not math.isnan(ltv_val):
    ltv_delta = (
        (ltv_val - ltv_prev) / ltv_prev * 100
        if ltv_prev and not math.isnan(ltv_prev) and ltv_prev != 0
        else None
    )
    c5, c6 = st.columns(2)
    c5.metric("LTV (今年)", f"{ltv_val:,.0f} 円")
    if ltv_delta is not None:
        c6.metric("LTV 前年比", f"{ltv_delta:+.1f} %")

# 7-2 月フルリスト補完

full_months = pd.DataFrame({"月": range(1, 13)})
ss_full = full_months.merge(ss, on="月", how="left").fillna(0)
mask = (ss_full["総売上_前年"] != 0) | (ss_full["総売上_今年"] != 0)
ss_full = ss_full[mask]

# ---------- 7-3 Altair 売上グラフ ----------

sales_plot = (
    ss_full.melt(id_vars="月", value_vars=["総売上_前年", "総売上_今年"], var_name="年度", value_name="売上")
    .replace({"総売上_前年": prev_year, "総売上_今年": latest_year})
)
sales_plot["売上"] /= 10_000  # 円→万円
sales_plot[["月", "年度"]] = sales_plot[["月", "年度"]].astype(str)

sales_chart = (
    alt.Chart(sales_plot)
    .mark_bar()
    .encode(
        x=alt.X("月:N", title="月", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("売上:Q", title="金額 (万円)"),
        xOffset="年度:N",
        color=alt.Color("年度:N", title="年度"),
        tooltip=["年度", "月", "売上"],
    )
    .properties(width=400, height=300, title=f"{store} 月別総売上（前年 vs 今年）")
)

st.altair_chart(sales_chart, use_container_width=True)

# ---------- 7-4 Altair 来院数グラフ ----------

visit_plot = (
    ss_full.melt(id_vars="月", value_vars=["総来院数_前年", "総来院数_今年"], var_name="年度", value_name="来院数")
    .replace({"総来院数_前年": prev_year, "総来院数_今年": latest_year})
)
visit_plot[["月", "年度"]] = visit_plot[["月", "年度"]].astype(str)

visit_chart = (
    alt.Chart(visit_plot)
    .mark_bar()
    .encode(
        x=alt.X("月:N", title="月", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("来院数:Q", title="人数"),
        xOffset="年度:N",
        color=alt.Color("年度:N", title="年度"),
        tooltip=["年度", "月", "来院数"],
    )
    .properties(width=400, height=300, title=f"{store} 月別来院数（前年 vs 今年）")
)

st.altair_chart(visit_chart, use_container_width=True)

# ---------- 7-5 Altair 来店動機グラフ ----------

if not reason_df.empty:
    rs = (
        reason_df[(reason_df["店舗名"] == store) & (reason_df["年"] == latest_year)]
        .groupby("カテゴリ", as_index=False)["件数"].sum()
    )

    if not rs.empty:
        rs["割合%"] = (rs["件数"] / rs["件数"].sum() * 100).round(1)

        reason_chart = (
            alt.Chart(rs)
            .mark_bar()
            .encode(
                y=alt.Y("カテゴリ:N", sort="-x", title="来店動機"),
                x=alt.X("件数:Q", title="件数"),
                tooltip=["カテゴリ", "件数", "割合%"],
            )
            .properties(width=350, height=250, title=f"{store} 来店動機内訳（{latest_year}年）")
        )

        st.altair_chart(reason_chart, use_container_width=True)

        with st.expander("📄 来店動機 明細"):
            st.dataframe(rs, use_container_width=True)

# ---------- 7-6 Altair LTV 折れ線 ----------

if not ltv_df.empty:
    ltv_monthly = (
        ltv_df.query("店舗名 == @store")
        .pivot(index="月", columns="年", values="LTV")
        .reset_index()
        .melt(id_vars="月", var_name="年度", value_name="LTV")
        .dropna()
    )

    if not ltv_monthly.empty:
        ltv_monthly[["月", "年度"]] = ltv_monthly[["月", "年度"]].astype(str)

        ltv_chart = (
            alt.Chart(ltv_monthly)
            .mark_line(point=True)
            .encode(
                x=alt.X("月:O", title="月"),
                y=alt.Y("LTV:Q", title="金額 (円)"),
                color=alt.Color("年度:N"),
                tooltip=["年度", "月", "LTV"],
            )
            .properties(width=400, height=250, title=f"{store} 月別LTV推移")
        )

        st.altair_chart(ltv_chart, use_container_width=True)

# ---------- 7-7 元データ表示 ----------

with st.expander("📄 月別比較データ（店舗）"):
    st.dataframe(ss_full, use_container_width=True)
