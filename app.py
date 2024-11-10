import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล Random Forest ที่บันทึกไว้
model = joblib.load('random_forest_model.pkl')

# สร้าง dictionary เพื่อแปลงจากผลลัพธ์ตัวเลขเป็นชื่อสายพันธุ์
species_dict = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

# ฟังก์ชันทำนายประเภทของเพนกวิน
def predict_penguin(species_features):
    prediction = model.predict([species_features])
    return species_dict[prediction[0]]  # แปลงผลลัพธ์เป็นชื่อสายพันธุ์

# UI บน Streamlit
st.title("Penguin Species Prediction")
st.write("กรุณาใส่ข้อมูลของเพนกวินเพื่อทำนาย species")

# รับข้อมูล input จากผู้ใช้
bill_length = st.number_input("culmen_length_mm")
bill_depth = st.number_input("culmen_depth_mm")
flipper_length = st.number_input("flipper_length_mm")
body_mass = st.number_input("body_mass_g")

# แปลงข้อมูล input ให้เป็น array
input_features = np.array([bill_length, bill_depth, flipper_length, body_mass])

# ปุ่มทำนาย
if st.button("Predict"):
    species = predict_penguin(input_features)
    st.write(f"The predicted species is: {species}")
