import pandas as pd
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("Housing.csv")

pr = df.profile_report()

st.title("Pandas Profiling in Streamlit")
st.write(df)
st_profile_report(pr)
