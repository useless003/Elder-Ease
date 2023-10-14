import streamlit as st
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Helper",
                   page_icon="👨‍🏭")


def lottie_url(url:str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

st.header(" Aid to  Elders and Disabled People with A.I")

lottie_animation = "https://lottie.host/1a128bcd-8b39-4798-a776-7d92da3a732b/SLotOXOGpI.json"
lottie_json = lottie_url(lottie_animation)
st_lottie(lottie_json,key="welcome")


col1 , col2 = st.columns(2)
with col1:
    lottie_animation2 = "https://lottie.host/6f5c3c8b-b964-4676-bf5b-a9e76f35f564/a9Ch45mRQH.json"
    lottie_json2=lottie_url(lottie_animation2)
    st_lottie(lottie_json2,height=300,width=300,key="elders")
with col2:
    st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
             "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
             "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
             "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

col4 , col5 = st.columns(2)
with col4:
    st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt"
                " ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
                "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
                "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat"
                " non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

with col5:
    lottie_animation3 = "https://lottie.host/3847dba8-0fc5-4fd8-9099-3a1751a34fc6/4ycL0qTfIG.json"
    lottie_json3 = lottie_url(lottie_animation3)
    st_lottie(lottie_json3,height=300,width=300,key="disabled")

st.header("Problems")
lottie_animation4 = "https://lottie.host/cb1625d9-1b08-4ada-b23f-b586c9176d95/bryFtsy9Jj.json"
lottie_json4 = lottie_url(lottie_animation4)
st_lottie(lottie_json4,height=600,width=600,key="problems")

st.write("These are the common problems faced by the elders and disabled people:")
st.markdown("- Item 1")
st.markdown("- Item 2")
st.markdown("- Item 3")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)
