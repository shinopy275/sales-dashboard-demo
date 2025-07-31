import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="棒グラフ検証", layout="centered")
st.title("📊 Plotly 棒グラフ – 固定データ検証")

# ──────────────────────────────
# 1. テストデータ（万円）
# ──────────────────────────────
data = {
    "月":   ["1", "2", "1", "2"],
    "年度": ["2024", "2024", "2025", "2025"],
    "売上": [252, 326, 315, 274]          # ← 万円単位の値
}
df = pd.DataFrame(data)

st.write("### 📄 DataFrame（テスト用）")
st.dataframe(df)

# ──────────────────────────────
# 2. 棒グラフ
# ──────────────────────────────
fig = px.bar(
    df,
    x="月",
    y="売上",
    color="年度",
    barmode="group",
    labels={"売上": "金額 (万円)"},
    category_orders={"月": ["1", "2"]},
    title="月別総売上（固定データ 2024 vs 2025）"
)
fig.update_yaxes(type="linear", rangemode="tozero", tickformat=",.0f")
fig.update_layout(bargap=0.15, bargroupgap=0.05)

# キーを付け、一度だけ描画
st.plotly_chart(fig, use_container_width=True, key="test-chart")
