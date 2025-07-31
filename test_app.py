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

st.set_page_config(layout="centered")
st.title("💡 軸フォーマットを完全に外した検証")

df = pd.DataFrame({
    "月": ["1","2","1","2"],
    "年度": ["2024","2024","2025","2025"],
    "売上": [252, 326, 315, 274]
})

fig = px.bar(
    df, x="月", y="売上",
    color="年度", barmode="group",
    # ★ 軸設定をいっさい指定しない ★
)

st.plotly_chart(fig, key="plain")
