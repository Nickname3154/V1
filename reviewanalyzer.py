import streamlit as st
import time
from collections import Counter

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration


@st.cache_resource
def load_sentiment_pipeline():
    model_name = "beomi/KcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@st.cache_resource
def load_summarizer():
    model_name = "digit82/kobart-summarization"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def get_coupang_reviews(product_url, max_reviews=30):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(product_url)
    time.sleep(3)

    try:
        review_tab = driver.find_element(By.XPATH, '//a[contains(@href, "review")]')
        review_tab.click()
        time.sleep(3)

        reviews = []
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_tries = 0

        while len(reviews) < max_reviews and scroll_tries < 10:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            review_elements = driver.find_elements(By.CLASS_NAME, "sdp-review__article__list__review__content")

            for elem in review_elements:
                text = elem.text.strip()
                if text and text not in reviews:
                    reviews.append(text)
                if len(reviews) >= max_reviews:
                    break

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_tries += 1

        return reviews[:max_reviews]

    finally:
        driver.quit()


def summarize_reviews(reviews, tokenizer, model):
    text = " ".join(reviews)
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=128, min_length=30, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def analyze_sentiment(reviews, sentiment_pipeline):
    results = sentiment_pipeline(reviews)
    labels = [r['label'] for r in results]
    return Counter(labels)


# Streamlit UI
st.set_page_config(page_title="쿠팡 리뷰 분석기", layout="wide")
st.title("🛍️ 쿠팡 상품 리뷰 요약 및 감정 분석")

product_url = st.text_input("쿠팡 상품 URL", placeholder="https://www.coupang.com/vp/products/XXXX")
max_reviews = st.slider("수집할 최대 리뷰 수", 10, 100, 30)

if st.button("분석 시작") and product_url:
    with st.spinner("🔄 리뷰 수집 중..."):
        reviews = get_coupang_reviews(product_url, max_reviews=max_reviews)
        st.success(f"✅ {len(reviews)}개 리뷰 수집 완료")

    with st.spinner("🔍 감정 분석 중..."):
        sentiment_pipeline = load_sentiment_pipeline()
        sentiment_result = analyze_sentiment(reviews, sentiment_pipeline)
        st.subheader("📊 감정 분석 결과")
        for label, count in sentiment_result.items():
            st.write(f"{label}: {count}개")

    with st.spinner("📝 리뷰 요약 중..."):
        tokenizer, model = load_summarizer()
        summary = summarize_reviews(reviews, tokenizer, model)
        st.subheader("🧾 리뷰 요약")
        st.write(summary)

else:
    st.info("쿠팡 상품 URL을 입력한 뒤 '분석 시작' 버튼을 눌러주세요.")
