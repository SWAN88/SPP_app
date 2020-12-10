import streamlit as st

st.title("Hello Streamlit!!")

st.subheader("This is calculator.")

a = st.slider("a: ", 0, 10, 5, 1)
b = st.slider("b: ", 0, 10, 5, 1)

st.write(f"{a} x {b} = {a*b}")
