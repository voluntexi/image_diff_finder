import streamlit as st
import cv2
import numpy as np
import image_detect
def process_image(input_image):
    # 这里可以加入你的图像处理代码
    # 作为示例，这里只是将图像转换为灰度图
    img = image_detect.start(input_image)

    return img

def main():
    st.set_page_config(
        page_title="大家来找茬~",
        page_icon=":smiley:",
        layout="wide",  # 可选值："centered"、"wide"、"wide"、"auto"
        initial_sidebar_state="auto",  # 可选值："expanded"、"collapsed"
    )

    st.title("益禾堂-疯狂星期二")
    st.write("点击下方上传图片！🔥🔥")
    uploaded_file = st.file_uploader("点击上传文件", type=["jpg", "jpeg", "png"],help="上传图片",label_visibility="hidden",key="file_uploader")
    hide_label = """
    <style>
        .css-9ycgxx {
            display: none;
        }
        .css-1aehpvj
        {
            display:none
        }
    </style>
    """

    st.markdown(hide_label, unsafe_allow_html=True)
    if uploaded_file is not None:
        # 读取上传的图像文件
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="上传的图片",channels="BGR", use_column_width=True)

        # 处理图像
        processed_image = process_image(image)
        # 显示处理后的图像
        st.image(processed_image, caption="处理后的图片",channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
