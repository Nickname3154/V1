# app.py

import time
from collections import Counter

import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import chromedriver_autoinstaller

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration


def get_coupang_reviews(product_url, max_reviews=30):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0")

    # 자동으로 chromedriver 설치 확인
    chromedriver_autoinstaller.install()

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


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "beomi/KcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@st.cache_resource(show_spinner=False)
def load_summarizer():
    model_name = "digit82/kobart-summarization"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def summarize_reviews_kobart(reviews, tokenizer, model):
    text = " ".join(reviews)
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=128, min_length=30, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def analyze_sentiment(reviews, sentiment_pipeline):
    results = sentiment_pipeline(reviews)
    labels = [r['label'] for r in results]
    return Counter(labels)


def main():
    st.title("쿠팡 리뷰 요약 및 감정 분석")
    st.write("쿠팡 상품 URL을 입력하면 리뷰를 크롤링하여 요약 및 감정 분석을 수행합니다.")

    url = st.text_input("쿠팡 상품 URL 입력", "")

    if url:
        with st.spinner("리뷰를 수집하는 중..."):
            try:
                reviews = get_coupang_reviews(url)
            except Exception as e:
                st.error(f"리뷰 수집 중 오류 발생: {e}")
                return

        st.success(f"총 {len(reviews)}개의 리뷰를 수집했습니다.")
        if not reviews:
            st.warning("리뷰가 없습니다.")
            return

        st.subheader("리뷰 예시")
        for i, r in enumerate(reviews[:5], 1):
            st.write(f"{i}. {r}")

        with st.spinner("감정 분석 중..."):
            sentiment_pipeline = load_sentiment_model()
            sentiment_result = analyze_sentiment(reviews, sentiment_pipeline)

        st.subheader("감정 분석 결과")
        for label, count in sentiment_result.items():
            st.write(f"{label}: {count}개")

        with st.spinner("리뷰 요약 중..."):
            tokenizer, summary_model = load_summarizer()
            summary = summarize_reviews_kobart(reviews, tokenizer, summary_model)

        st.subheader("리뷰 요약 결과")
        st.write(summary)


if __name__ == "__main__":
    main()