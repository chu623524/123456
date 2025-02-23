import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO

# 设置模型和标准化器的路径
MODEL_URL = 'https://raw.githubusercontent.com/chu623524/123456/main/best_svm_model.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/chu623524/123456/main/scaler.pkl'

# 下载模型和标准化器
@st.cache_resource
def load_model():
    # 下载模型
    model_response = requests.get(MODEL_URL)
    model_data = BytesIO(model_response.content)
    model = joblib.load(model_data)

    # 下载标准化器
    scaler_response = requests.get(SCALER_URL)
    scaler_data = BytesIO(scaler_response.content)
    scaler = joblib.load(scaler_data)

    return model, scaler

model, scaler = load_model()

# 定义类别变量选项
吸烟状态_options = {0: "不吸烟", 1: "吸烟"}
性别_options = {0: "男", 1: "女"}
家庭月收入_options = {0: "0-2000", 1: "2001-5000", 2: "5001-8000", 3: "8000+"}
心理负担_options = {0: "没有", 1: "稍有", 2: "中度", 3: "较重", 4: "严重"}
文化程度_options = {0: "小学及以下", 1: "初中", 2: "高中/中专", 3: "大专及以上"}

# 设置Web界面
st.title("PTSD 预测系统")
st.write("基于支持向量机 (SVM) 进行 PTSD 预测")

# 获取用户输入的特征
ASDS = st.number_input("ASDS (分)", value=50.0, help="单位: 分，最高95分")
白蛋白 = st.number_input("白蛋白 (g/L)", value=50.0, help="单位: g/L")
吸烟状态 = st.selectbox("吸烟状态", options=list(吸烟状态_options.keys()), format_func=lambda x: 吸烟状态_options[x])
疼痛评分 = st.number_input("疼痛评分 (分)", value=5.0, help="单位: 分")
心理负担 = st.selectbox("心理负担", options=list(心理负担_options.keys()), format_func=lambda x: 心理负担_options[x])  # 分类变量
谷胺酰基移换酶 = st.number_input("谷胺酰基移换酶 (U/L)", value=50.0, help="单位: U/L")
A_G = st.number_input("A/G", value=0.5, help="范围 0 - 3.0")
住院天数 = st.number_input("住院天数 (天)", value=10, help="单位: 天")
文化程度 = st.selectbox("文化程度", options=list(文化程度_options.keys()), format_func=lambda x: 文化程度_options[x])  # 分类变量
氯 = st.number_input("氯 (mmol/L)", value=50.0, help="单位: mmol/L，范围 0-150")
性别 = st.selectbox("性别", options=list(性别_options.keys()), format_func=lambda x: 性别_options[x])  # 0男 1女
舒张压 = st.number_input("舒张压 (mmHg)", value=80.0, help="单位: mmHg")
中性粒细胞绝对值 = st.number_input("中性粒细胞绝对值 (10^9/L)", value=50.0, help="单位: 10^9/L，范围 0-100")
碱性磷酸酶 = st.number_input("碱性磷酸酶 (U/L)", value=150.0, help="单位: U/L")
体温 = st.number_input("体温 (°C)", value=37.0, help="单位: 摄氏度，范围 30-42")
凝血酶原时间比值 = st.number_input("凝血酶原时间比值", value=1.0, help="单位: 无单位")
家庭月收入 = st.selectbox("家庭月收入", options=list(家庭月收入_options.keys()), format_func=lambda x: 家庭月收入_options[x])  # 收入区间

# 创建一个字典来存储所有输入的特征
input_data = {
    'ASDS': ASDS,
    '白蛋白': 白蛋白,
    '吸烟状态': 吸烟状态,
    '疼痛评分': 疼痛评分,
    '心理负担': 心理负担,
    '谷胺酰基移换酶': 谷胺酰基移换酶,
    'A/G': A_G,
    '住院天数': 住院天数,
    '文化程度': 文化程度,
    '氯': 氯,
    '性别': 性别,
    '舒张压': 舒张压,
    '中性粒细胞绝对值': 中性粒细胞绝对值,
    '碱性磷酸酶': 碱性磷酸酶,
    '体温': 体温,
    '凝血酶原时间比值': 凝血酶原时间比值,
    '家庭月收入': 家庭月收入
}

# 预测按钮
if st.button("预测"):
    # 将输入数据转换为 NumPy 数组
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # 标准化
    input_scaled = scaler.transform(input_array)

    # 进行预测
    prediction_prob = model.predict_proba(input_scaled)[0, 1]  # PTSD 的概率
    prediction = "PTSD 高风险" if prediction_prob > 0.5 else "PTSD 低风险"
    
    # 输出结果
    st.write(f"**预测结果:** {prediction}")
    st.write(f"**PTSD 概率:** {prediction_prob:.4f}")
