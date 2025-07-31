import streamlit as st, pandas as pd, plotly.express as px

st.set_page_config(page_title="売上ダッシュボード", layout="wide")
st.title("📝 Excelアップロード → 売上ダッシュボード")

up = st.file_uploader("Excel ファイル (.xlsx)", type="xlsx")
if up:
    df = pd.read_excel(up, engine="openpyxl")
    # 列名はお手元フォーマットに合わせて調整してください
    df["日付"] = pd.to_datetime(df["日付"])
    df["月"]   = df["日付"].dt.to_period("M").astype(str)

    summary = df.groupby("月").agg(
        総売上=("総売上","sum"),
        来院数=("来院数","sum")
    ).reset_index()

    # KPI 表示
    col1, col2 = st.columns(2)
    col1.metric("今月 総売上", f"{summary['総売上'].iloc[-1]:,.0f} 円")
    col2.metric("今月 来院数",  f"{summary['来院数'].iloc[-1]:,} 人")

    # グラフ
    st.plotly_chart(
        px.bar(summary, x="月", y="総売上", title="月別総売上"),
        use_container_width=True
    )

    # 元データ確認
    st.dataframe(df)
else:
    st.info("Excel をアップロードするとダッシュボードが表示されます。")
