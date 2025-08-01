import streamlit as st
import pandas as pd
import altair as alt
import re, math
from typing import List, Tuple

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 前年同月ダッシュボード")

# ──────────────────────────────
# 1. Helper
# ──────────────────────────────

def get_store_name(fname: str) -> str:
    m = re.match(r"\d+[._](.+?)\s", fname)
    if not m:
        raise ValueError("店舗名が解析できません")
    return m.group(1).strip()


def infer_year_month(df: pd.DataFrame) -> Tuple[int, int]:
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    valid = df["日付"].dropna()
    if valid.empty:
        raise ValueError("日付列がありません")
    return int(valid.dt.year.mode()[0]), int(valid.dt.month.mode()[0])

# ──────────────────────────────
# 2. 患者分析シートパース
# ──────────────────────────────

def parse_patient_analysis(f):
    sheet = pd.read_excel(f, sheet_name="患者分析", header=None, engine="openpyxl")

    def _grab(keyword):
        idx_mask = sheet.apply(lambda r: r.astype(str).str.contains(keyword).any(), axis=1)
        if not idx_mask.any():
            return pd.DataFrame()
        r = idx_mask.idxmax()
        header = sheet.loc[r + 1].dropna()
        if header.empty:
            return pd.DataFrame()
        vals = sheet.loc[r + 2, header.index]
        return pd.DataFrame({"カテゴリ": header.values, "件数": pd.to_numeric(vals, errors="coerce").fillna(0)})

    gender = _grab("男女比率")
    reason = _grab("来院動機")
    age    = _grab("年齢比率")
    return gender, reason, age

# ──────────────────────────────
# 3. LTV パース
# ──────────────────────────────

def parse_ltv(f):
    df = pd.read_excel(f, sheet_name="店舗分析", header=None, engine="openpyxl")
    mask = df.apply(lambda r: r.astype(str).str.contains("LTV").any(), axis=1)
    if not mask.any():
        return None
    nums = pd.to_numeric(df[mask].iloc[0], errors="coerce").dropna()
    return float(nums.iloc[0]) if not nums.empty else None

# ──────────────────────────────
# 4. ファイルロード
# ──────────────────────────────

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
            st.warning(f"{f.name}: 売上シート読込失敗 {e}"); continue

        for col in ["総売上", "総来院数"]:
            if col in df_sales.columns:
                df_sales[col] = pd.to_numeric(df_sales[col], errors="coerce").fillna(0)

        try:
            y, m = infer_year_month(df_sales)
        except ValueError as e:
            st.warning(f"{f.name}: {e}"); continue

        df_sales["店舗名"], df_sales["年"], df_sales["月"] = store, y, m
        sales.append(df_sales)

        try:
            g, r, a = parse_patient_analysis(f)
            for _df in (g, r, a):
                if not _df.empty:
                    _df["店舗名"], _df["年"], _df["月"] = store, y, m
            if not r.empty: reasons.append(r)
            if not g.empty: genders.append(g)
            if not a.empty: ages.append(a)
        except Exception as e:
            st.warning(f"{f.name}: 患者分析パース失敗 {e}")

        val = parse_ltv(f)
        if val is not None:
            ltvs.append({"店舗名": store, "年": y, "月": m, "LTV": val})

    return (
        pd.concat(sales,   ignore_index=True) if sales   else pd.DataFrame(),
        pd.concat(reasons, ignore_index=True) if reasons else pd.DataFrame(),
        pd.concat(genders, ignore_index=True) if genders else pd.DataFrame(),
        pd.concat(ages,    ignore_index=True) if ages    else pd.DataFrame(),
        pd.DataFrame(ltvs),
    )

# ──────────────────────────────
# 5. UI: ファイルアップロード
# ──────────────────────────────

files = st.file_uploader("📂 複数店舗の Excel ファイルを選択（複数選択可）", type="xlsx", accept_multiple_files=True)
if not files:
    st.info("ファイルをアップロードするとダッシュボードが表示されます。")
    st.stop()

sales_df, reason_df, gender_df, age_df, ltv_df = load(files)
if sales_df.empty:
    st.error("有効なファイルがありません"); st.stop()

# ──────────────────────────────
# 6. 全店舗サマリー
# ──────────────────────────────

monthly = (sales_df.groupby(["店舗名", "年", "月"], as_index=False)[["総売上", "総来院数"]].sum())
latest, prev = monthly["年"].max(), monthly["年"].max()-1
this_y, prev_y = monthly[monthly["年"]==latest], monthly[monthly["年"]==prev]
comp = pd.merge(this_y, prev_y, on=["店舗名", "月"], how="left", suffixes=("_今年", "_前年"))
for k in ["総売上", "総来院数"]:
    comp[f"{k}増減率%"] = ((comp[f"{k}_今年"]-comp[f"{k}_前年"])/comp[f"{k}_前年"].replace(0,pd.NA)*100).round(1)

# LTV merge
if not ltv_df.empty:
    ltv_this = ltv_df[ltv_df["年"]==latest]
    ltv_prev = ltv_df[ltv_df["年"]==prev]
    ltv_cmp  = pd.merge(ltv_this, ltv_prev, on=["店舗名", "月"], how="left", suffixes=("_今年","_前年"))
    ltv_cmp["LTV増減率%"] = ((ltv_cmp["LTV_今年"]-ltv_cmp["LTV_前年"])/ltv_cmp["LTV_前年"].replace(0,pd.NA)*100).round(1)
    comp = pd.merge(comp, ltv_cmp[["店舗名","月","LTV_前年","LTV_今年","LTV増減率%"]], on=["店舗名","月"], how="left")

# 合計行
num_cols = [c for c in comp.columns if any(k in c for k in ["総売上_","総来院数_","LTV_"])]
rows=[]
for m in sorted(comp["月"].unique()):
    sub=comp[comp["月"]==m]
    if sub.empty: continue
    d={"店舗名":"合計","月":m}
    for c in num_cols: d[c]=sub[c].sum()
    if d.get("総売上_前年",0): d["総売上増減率%"]=((d["総売上_今年"]-d["総売上_前年"])/d["総売上_前年"]*100).round(1)
    if d.get("総来院数_前年",0): d["総来院数増減率%"]=((d["総来院数_今年"]-d["総来院数_前年"])/d["総来院数_前年"]*100).round(1)
    if d.get("LTV_前年",0): d["LTV増減率%"]=((d["LTV_今年"]-d["LTV_前年"])/d["LTV_前年"]*100).round(1)
    rows.append(d)
comp_disp = pd.concat([comp, pd.DataFrame(rows)], ignore_index=True)

st.subheader(f"📊 全店舗サマリー（{latest}年 vs {prev}年）")
show_cols=["店舗名","月","総売上_前年","総売上_今年","総売上増減率%","総来院数_前年","総来院数_今年","総来院数増減率%","LTV_前年","LTV_今年","LTV増減率%"]
st.dataframe(comp_disp[show_cols], use_container_width=True)

# ──────────────────────────────
# 7. 店舗別ビュー
# ──────────────────────────────

store = st.selectbox("🔍 店舗を選択", sorted(comp["店舗名"].unique()))
st.markdown("---"); st.header(f"🏪 {store} の詳細")
ss = comp[comp["店舗名"]==store].sort_values("月")

# KPI
sum_row = ss[["総売上_前年","総売上_今年","総来院数_前年","総来院数_今年"]].sum()
c1,c2=st.columns(2)
c1.metric("売上 前年比(累計)", f"{((sum_row['総売上_今年']-sum_row['総売上_前年'])/sum_row['総売上_前年']*100).round(1)} %")
c2.metric("来院数 前年比(累計)", f"{((sum_row['総来院数_今年']-sum_row['総来院数_前年'])/sum_row['総来院数_前年']*100).round(1)} %")

c3,c4=st.columns(2)
c3.metric("売上 (今年)", f"{int(sum_row['総売上_今年']):,} 円")
c4.metric("来院数 (今年)", f"{int(sum_row['総来院数_今年']):,} 人")

ltv_cur = ltv_df.query("店舗名 == @store & 年 == @latest")['LTV'].mean()
ltv_prev_mean = ltv_df.query("店舗名 == @store & 年 == @prev")['LTV'].mean()
if not math.isnan(ltv_cur):
    c5,c6=st.columns(2)
    c5.metric("LTV (今年)", f"{ltv_cur:,.0f} 円")
    if ltv_prev_mean and not math.isnan(ltv_prev_mean):
        c6.metric("LTV 前年比", f"{((ltv_cur-ltv_prev_mean)/ltv_prev_mean*100).round(1):+.1f} %")

# 月別補完
full_m=pd.DataFrame({"月":range(1,13)})
ss_full=full_m.merge(ss,on="月",how="left").fillna(0)
mask=(ss_full['総売上_前年']!=0)|(ss_full['総売上_今年']!=0)
ss_full=ss_full[mask]

# 売上 & 来院数グラフ
for label, cols, title, ytitle in [
    ("売上", ["総売上_前年","総売上_今年"], "月別総売上", "金額 (万円)"),
    ("来院数", ["総来院数_前年","総来院数_今年"], "月別来院数", "人数")]:
    df_plot=ss_full.melt(id_vars="月", value_vars=cols, var_name="年度", value_name=label)
    df_plot=df_plot.replace({cols[0]:prev, cols[1]:latest})
    if label=="売上": df_plot[label]/=10000
    df_plot[["月","年度"]]=df_plot[["月","年度"]].astype(str)
    chart=(alt.Chart(df_plot).mark_bar().encode(
        x=alt.X("月:N",axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{label}:Q", title=ytitle),
        xOffset="年度:N", color="年度:N", tooltip=["年度","月",label]
    ).properties(width=400,height=300,title=f"{store} {title}（前年 vs 今年）"))
    st.altair_chart(chart,use_container_width=True)

# 来店動機
if not reason_df.empty:
    rs=reason_df.query("店舗名 == @store & 年 == @latest").groupby("カテゴリ",as_index=False)["件数"].sum()
    if not rs.empty:
        rs["割合%"]=(rs["件数"] / rs["件数"].sum()*100).round(1)
        chart=alt.Chart(rs).mark_bar().encode(y=alt.Y("カテゴリ:N",sort="-x"),x="件数:Q",tooltip=["カテゴリ","件数","割合%"]).properties(width=350,height=250,title="来店動機")
        st.altair_chart(chart, use_container_width=True)
        with st.expander("📄 来店動機 明細"):
            st.dataframe(rs, use_container_width=True)

# 男女比率
if not gender_df.empty:
    gd=gender_df.query("店舗名 == @store & 年 == @latest").groupby("カテゴリ",as_index=False)["件数"].sum()
    if not gd.empty:
        gd["割合%"]=(gd["件数"] / gd["件数"].sum()*100).round(1)
        chart=alt.Chart(gd).mark_bar().encode(x=alt.X("カテゴリ:N"),y="件数:Q",tooltip=["カテゴリ","件数","割合%"],color="カテゴリ:N").properties(width=350,height=250,title="男女比率")
        st.altair_chart(chart, use_container_width=True)
        with st.expander("📄 男女比率 明細"):
            st.dataframe(gd, use_container_width=True)

# 年齢比率
if not age_df.empty:
    ad=age_df.query("店舗名 == @store & 年 == @latest").groupby("カテゴリ",as_index=False)["件数"].sum()
    if not ad.empty:
        ad["割合%"]=(ad["件数"] / ad["件数"].sum()*100).round(1)
        chart=alt.Chart(ad).mark_bar().encode(y=alt.Y("カテゴリ:N",sort="-x"),x="件数:Q",tooltip=["カテゴリ","件数","割合%"]).properties(width=350,height=300,title="年齢比率")
        st.altair_chart(chart, use_container_width=True)
        with st.expander("📄 年齢比率 明細"):
            st.dataframe(ad, use_container_width=True)

# 元データ
with st.expander("📄 月別比較データ（店舗）"):
    st.dataframe(ss_full, use_container_width=True)
