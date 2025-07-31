import altair as alt
import pandas as pd
import streamlit as st

# テストデータ（万円）
df = pd.DataFrame({
    "月": ["1","2","1","2"],
    "年度": ["2024","2024","2025","2025"],
    "売上": [252, 326, 315, 274]
})

chart = (
    alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("月:N", title="月", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("売上:Q", title="金額 (万円)"),
            xOffset="年度:N",               # ← これが “横並び” になる鍵
            color=alt.Color("年度:N", title="年度"),
            tooltip=["年度", "月", "売上"]
        )
        .properties(width=350, height=300)
)

st.altair_chart(chart, use_container_width=True)
