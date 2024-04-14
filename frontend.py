import requests
import streamlit as st
import json

with open("values.json", "r") as f:
    values = json.load(f)

Shop = st.selectbox("Shop", values["Shop"])
BrandName = st.selectbox("BrandName", values["BrandName"])
ModelGroup = st.selectbox("ModelGroup", values["ModelGroup"])
ProductGroup = st.selectbox("ProductGroup", values["ProductGroup"])
OriginalSaleAmountInclVAT = st.number_input("OriginalSaleAmountInclVAT")
Day = st.selectbox("Day", values["day"])
Month = st.selectbox("Month", values["month"])

if st.button("Predict!"):
    inputs = {
        "Shop": int(Shop),
        "BrandName": int(BrandName),
        "ModelGroup": int(ModelGroup),
        "ProductGroup": int(ProductGroup),
        "day": str(Day),
        "month": str(Month),
        "OriginalSaleAmountInclVAT": float(OriginalSaleAmountInclVAT),
    }
    
    try:
        r = requests.post(url="http://127.0.0.1:8000/pred", json=inputs)
        if r.status_code == 200:
            response_data = r.json()
            prediction = response_data["pred"]
            st.subheader(f"Return?: {prediction}")
        else:
            st.error(f"Status code: {r.status_code} Error: {r.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")