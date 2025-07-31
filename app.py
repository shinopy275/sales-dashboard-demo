import streamlit as st, pandas as pd, tempfile, re, plotly.express as px

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("Excel→前年同月ダッシュボード")
files = st.file_uploader("複数ファイルを選択", type="xlsx", accept_multiple_files=True)

def parse_meta(name):
    m = re.match(r"(\d+)店.*?(\d{4})(\d{2})", name)
    return m.group(1)+"店", int(m.group(2)), int(m.group(3))

@st.cache_data  # セッション中にキャッシュ
def load_files(files):
    dfs=[]
    for f in files:
        shop, year, month = parse_meta(f.name)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        tmp.write(f.read()); tmp.close()
        df = pd.read_excel(tmp.name, sheet_name="売上管理", engine="openpyxl")
        df["店舗名"], df["年"], df["月"] = shop, year, month
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

if files:
    df_all = load_files(files)

    # ---- 月次集計 ----
    agg_cols = {"総売上":"sum", "総来院数":"sum"}
    monthly = df_all.groupby(["店舗名","年","月"]).agg(agg_cols).reset_index()

    # ---- 前年同月比較 ----
    this = monthly[monthly["年"]==monthly["年"].max()]
    prev = monthly[monthly["年"]==monthly["年"].max()-1]
    comp = pd.merge(this, prev, on=["店舗名","月"], suffixes=("_今年","_前年"))

    for k in ["総売上","総来院数"]:
        comp[f"{k}増減率%"] = ((comp[f"{k}_今年"]-comp[f"{k}_前年"])
                               /comp[f"{k}_前年"].replace(0,pd.NA)*100).round(1)

    st.subheader("全店舗サマリー")
    st.dataframe(comp)

    # ---- 店舗別ダッシュボード ----
    store = st.selectbox("店舗を選択", sorted(comp["店舗名"].unique()))
    ss = comp[comp["店舗名"]==store]
    col1,col2 = st.columns(2)
    col1.metric("売上前年比", f"{ss['総売上増減率%'].iat[0]} %")
    col2.metric("来院数前年比", f"{ss['総来院数増減率%'].iat[0]} %")

    st.plotly_chart(px.bar(ss, x="月", y=["総売上_前年","総売上_今年"],
                           barmode="group", title=f"{store} 売上比較"))

