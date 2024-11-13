import os
import time
import html
import torch
import re  # 新增此行以使用正則表達式進行名稱前處理
import jieba
import numpy as np
import pandas as pd
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from numpy.linalg import norm
from nltk.corpus import wordnet
from src import config, inference_api
from argparse import ArgumentParser
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# %%


def get_semantic_model():
    # 設定語意模型
    model = ORTModelForFeatureExtraction.from_pretrained(
        'clw8998/semantic_model-for-EE5327701')
    tokenizer = AutoTokenizer.from_pretrained(
        'clw8998/semantic_model-for-EE5327701')
    return model, tokenizer

# %%


def scrapy_MOMO(keyword='鞋櫃', page=10, max_item_num=10000):
    file_path = f'./{momo_searched_results_folder}/momo_{keyword}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f'load {file_path}...')
        return df[:max_item_num]

    print(f'Searching {keyword}...')

    # 定義 MOMO 爬蟲函數
    # special case
    if keyword == '[馬玉山] 紫山藥黑米仁 (30gx12入x袋)':
        keyword = '[馬玉山] 紫山藥黑米仁 (30g*12入/袋)'

    headers = {
        "Referer": "https://m.momoshop.com.tw/",
        'User-Agent': 'Mozilla/5.0'
    }
    df = pd.DataFrame(columns=['Title', 'Price'])
    item_number = 0
    for index in range(1, page + 1):
        url = f'https: // m.momoshop.com.tw/search.momo?curPage = {
            index} & searchKeyword = {keyword}'
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            for item in soup.select('li.goodsItemLi > a'):
                title = str(item["title"]).title()
                price_element = item.find('b', {'class': 'price'})
                price = price_element.text if price_element else '0'
                df = pd.concat([df, pd.DataFrame(
                    {'Title': title, 'Price': price}, index=[0])], ignore_index=True)
        df = df.drop_duplicates(ignore_index=True)
        if df.shape[0] >= max_item_num:
            break

    # special case
    if keyword == '[馬玉山] 紫山藥黑米仁 (30g*12入/袋)':
        keyword = '[馬玉山] 紫山藥黑米仁 (30gx12入x袋)'

    momo_search_output_folder = momo_searched_results_folder
    os.makedirs(momo_search_output_folder, exist_ok=True)
    output_file_path = os.path.join(
        momo_search_output_folder, f'momo_{keyword}.csv')
    df.to_csv(output_file_path, index=False, encoding='utf-8')

    return df[:max_item_num]

# %%


def preprocess_name(name):
    # 針對名稱進行前處理
    name = re.sub(r'[^\w\s]', '', name)  # 去除非字母或數字的符號
    name = re.sub(r'\s+', ' ', name).strip()  # 移除多餘空白
    return name


def load_search_terms(filename):
    # 讀取搜尋詞和商品名稱
    with open(filename, 'r', encoding='utf-8') as f:
        search_terms = f.read().splitlines()
    return search_terms


def load_product_data(filename):

    if os.path.exists(filename) == False:
        return None

    df = pd.read_csv(filename)
    # 使用前處理
    # return [preprocess_name(name) for name in df['product_name'].head(30).tolist()]
    return df['product_name'].head(30).tolist()


def inference(tokenizer, model, texts):
    # 嵌入向量計算
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


def cosine_similarity(vec1, vec2):
    # 計算餘弦相似度
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def get_weighted_similarity(momo_embedding, product_embedding, momo_attrs, product_attrs, attribute_weights):
    # 計算文本的語意相似度
    text_similarity = cosine_similarity(momo_embedding, product_embedding)

    # 計算屬性相似度（加權計算）
    weighted_attr_similarity = 0
    for attr in momo_attrs:
        if attr in product_attrs:
            for (momo_attr_value, momo_confidence), (product_attr_value, product_confidence) in zip(momo_attrs[attr], product_attrs[attr]):
                if momo_attr_value == product_attr_value:
                    weighted_attr_similarity += attribute_weights.get(
                        attr, 0) * min(momo_confidence, product_confidence)

    # 綜合相似度（加權結合文本和屬性相似度）
    total_similarity = (text_similarity + weighted_attr_similarity) / 2
    return total_similarity


# 將屬性結果結合進去

def get_ner_attributes(item, ner_result):
    # 提取 NER 屬性資料
    if item in ner_result:
        return ner_result[item]
    return {}


def jieba_tokenizer(text):
    # 初始化分詞器
    tokens = jieba.lcut(text, cut_all=False)
    stop_words = ['【', '】', '/', '~', '＊', '、', '（',
                  '）', '+', '‧', ' ', '', "的", "是", "在", "了", "和"]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def process_match(momo_title, product_title, momo_embedding, product_embedding, momo_ner, product_ner, threshold, attribute_weights):
    # 提取 MOMO 和產品的 NER 屬性資料
    momo_attrs = get_ner_attributes(momo_title, momo_ner)
    product_attrs = get_ner_attributes(product_title, product_ner)

    # 計算加權相似度
    similarity = get_weighted_similarity(
        momo_embedding, product_embedding, momo_attrs, product_attrs, attribute_weights)

    # 如果相似度小於閾值，返回 0 代替 None
    if similarity >= threshold:
        return (momo_title, product_title, similarity)
    else:
        print(f'similarity: {similarity}')
        return (momo_title, None, 0)  # 返回 0 而非 None


def refine_search_term(product_names, ner_model, mer_tokenizer, min_count_threshold=2, min_f1_threshold=0.6):
    # 指定需要的屬性
    target_attributes = ['品牌', '名稱', '產品', '顏色']

    # 用來記錄指定屬性中每個值的出現次數和 F1-score
    attribute_counts = {attr: {} for attr in target_attributes}

    # 處理每個產品名稱
    for name in product_names:
        # 取得 NER 標記
        attrs = inference_api.get_ner_tags(
            ner_model, mer_tokenizer, [name], target_attributes)

        for attr, values in attrs.get(name, {}).items():
            # 確保屬性在 attribute_counts 中存在
            if attr not in attribute_counts:
                attribute_counts[attr] = {}

            # 累積每個值的出現次數
            for value, confidence in values:
                # 假設 F1-score 存在 confidence 中，若無需額外處理可以直接使用 confidence 作為 F1-score
                f1_score = confidence  # 假設 confidence 即是 F1-score

                if value not in attribute_counts[attr]:
                    attribute_counts[attr][value] = {'count': 0, 'f1_score': 0}

                # 累積值的出現次數
                attribute_counts[attr][value]['count'] += 1
                # 記錄最高的 F1-score
                attribute_counts[attr][value]['f1_score'] = max(
                    attribute_counts[attr][value]['f1_score'], f1_score)

    # 根據指定屬性的出現頻率和 F1-score 選出值，若超過指定次數和 F1-score 閾值則添加到 refined_terms
    refined_terms = []
    for attr, counts in attribute_counts.items():
        for value, stats in counts.items():
            count = stats['count']
            f1_score = stats['f1_score']
            if count >= min_count_threshold and f1_score >= min_f1_threshold:  # 如果出現次數和 F1-score 都符合條件
                refined_terms.append(value)

    # 如果 refined_terms 為空字串，回傳 None
    if not refined_terms:
        return None

    return " ".join(refined_terms)


def load_models_and_matrices(path):
    # Function to load the models
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']


def jieba_tokenizer(text):
    # 使用 jieba 的精確模式進行分詞
    tokens = jieba.lcut(text, cut_all=False)
    # 定義不需要的符號列表，包括 preprocess_name 中去除的符號
    stop_words = ['【', '】', '/', '~', '＊', '、', '（', '）', '+', '‧', ' ', '',
                  "的", "是", "在", "了", "和", '.', ',', '!', '?', '-', '_', ':', ';', '’', '‘', '“', '”']
    # 過濾掉不需要的符號
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def save_models_and_matrices(tfidf, items_tfidf_matrix, path):
    # Function to save the models
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
        }, file)


def load_models_and_matrices(path):
    # Function to load the models
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']


def filter_important_words(texts, vectorizer):
    # 設定 TF-IDF 過濾參數
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 保留重要詞彙
    filtered_texts = []
    for row in tfidf_matrix.toarray():
        filtered_text = " ".join(
            [feature_names[i] for i, score in enumerate(row) if score > 0]
        )

        if len(filtered_text) != 0:
            filtered_texts.append(filtered_text)

    # print(filtered_texts)
    return filtered_texts


def flatten_tags(tag_dict, att):
    return set(ent for ents in tag_dict.get(att, []) for ent in ents)


def calculate_weighted_relevancy(query_pool, target_pool, check_att, margin, tag_weights):
    # Use tag weights if provided, otherwise default to equal weighting
    tag_weights = tag_weights or {att: 1 for att in check_att}

    relevancy_score = 0
    # Total weight for normalizing
    total_weight = sum(tag_weights[att] for att in check_att)

    # Loop over the required attribute categories
    for att in check_att:

        query_tag_set = flatten_tags(query_pool, att)  # Flattened query tags
        target_tag_set = flatten_tags(
            target_pool, att)   # Flattened target tags

        # Calculate intersection based on set size
        intersection_size = len(query_tag_set & target_tag_set)

        # Calculate relevancy for current attribute based on margin and weights
        if intersection_size >= len(query_tag_set) * margin:
            relevancy_score += tag_weights[att]  # Fully match
        elif intersection_size > 0:
            relevancy_score += tag_weights[att] * 0.5  # Partial match

    return relevancy_score / total_weight  # Normalize score


def ner_relevancy(df, index, all_results, check_att, margin, target_weights):
    print(df.loc[index, '搜尋詞'])

    if df.loc[index, '搜尋詞'].lower() not in all_results:
        print(f"Warning(query): '{
              df.loc[index, '搜尋詞']}' not found in NER results.")
        return df
    if df.loc[index, 'coupang'].lower() not in all_results:
        print(
            f"Warning(coupang): '{df.loc[index, 'coupang']}' not found in NER results.")
        return df
    if df.loc[index, 'momo'].lower() not in all_results:
        print(f"Warning(momo): '{
              df.loc[index, 'momo']}' not found in NER results.")
        return df

    query_tags_dict = all_results[df.loc[index, '搜尋詞'].lower()]
    coupang_tags_dict = all_results[df.loc[index, 'coupang'].lower()]
    MOMO_tags_dict = all_results[df.loc[index, 'momo'].lower()]

    query_tags_pool = set(
        ent[0] for ents in query_tags_dict.values() for ent in ents if ent)
    coupang_tags_pool = set(
        ent[0] for ents in coupang_tags_dict.values() for ent in ents if ent)
    MOMO_tags_pool = set(
        ent[0] for ents in MOMO_tags_dict.values() for ent in ents if ent)

    # 依權重計算相似度
    if len(coupang_tags_pool & MOMO_tags_pool) >= len(check_att) * margin * 0.5:
        tfidf_relevancy_score = calculate_weighted_relevancy(
            coupang_tags_pool, MOMO_tags_pool, check_att, margin, target_weights)
        df.loc[index, 'ner_relevancy'] = tfidf_relevancy_score

    return df


def product_matching():
    for search_term in search_terms:
        print(f'current term: {search_term}')

        # special case
        if search_term == '[馬玉山] 紫山藥黑米仁 (30g*12入/袋)':
            search_term = '[馬玉山] 紫山藥黑米仁 (30gx12入x袋)'

        # skip
        t_matched_file_name = f'./{matched_results_folder}/{search_term}.csv'
        if os.path.exists(t_matched_file_name):
            print(f'file already exists: {t_matched_file_name}')
            continue

        # 從 CSV 讀取商品名稱
        # Coupang 的搜尋結果
        original_product_names = load_product_data(
            f'./{coupang_searched_results_folder}/{student_id}_{search_term}.csv')
        if original_product_names is None:
            continue
        product_names = [preprocess_name(name)
                         for name in original_product_names]

        # MOMO 爬取商品資料
        momo_search_term = search_term
        momo_products = scrapy_MOMO(momo_search_term)
        original_momo_titles = momo_products['Title'].tolist()
        momo_titles = [preprocess_name(title)
                       for title in original_momo_titles]
        # momo_search_term = refine_search_term(
        #     product_names, NER_model, NER_tokenizer)
        # if momo_search_term is None:
        #     momo_search_term = search_term
        # print(momo_search_term)
        # momo_products = scrapy_MOMO(momo_search_term)
        # momo_products_2 = scrapy_MOMO(search_term)
        # original_momo_titles = momo_products['Title'].tolist()
        # original_momo_titles.extend(momo_products_2['Title'].tolist())
        # momo_titles = [preprocess_name(title)
        #                for title in momo_products['Title'].tolist()]

        # 嵌入 MOMO 商品名稱
        if not momo_titles:
            print(f'search term: {momo_search_term} is empty')
            continue

        # 使用 TF-IDF 過濾不重要詞彙
        filtered_momo_titles = filter_important_words(
            momo_titles, tfidf_vectorizer)
        filtered_product_names = filter_important_words(
            product_names, tfidf_vectorizer)

        # ner tags
        momo_ner = inference_api.get_ner_tags(
            NER_model, NER_tokenizer, filtered_momo_titles, all_attribute)
        product_ner = inference_api.get_ner_tags(
            NER_model, NER_tokenizer, filtered_product_names, all_attribute)

        momo_embeddings = inference(tokenizer, model, filtered_momo_titles)
        product_embeddings = inference(
            tokenizer, model, filtered_product_names)

        # 匹配商品並生成 DataFrame
        all_products = []

        # for momo_title, momo_embedding, original_momo_title in tqdm(zip(momo_titles, momo_embeddings, original_momo_titles), desc="匹配中"):
        #     max_similarity = 0
        #     best_match = None
        #     for product_name, product_embedding, original_product_name in zip(product_names, product_embeddings, original_product_names):
        #         similarity = cosine_similarity(momo_embedding, product_embedding)
        #         if similarity > max_similarity:
        #             max_similarity = similarity
        #             best_match = original_product_name
        #     if max_similarity >= threshold:
        #         all_products.append(
        #             (original_momo_title, best_match, max_similarity))
        #     else:
        #         all_products.append((original_momo_title, None, None))

        # 將屬性結果與語意匹配結合
        for momo_title, momo_embedding in tqdm(zip(momo_titles, momo_embeddings), desc="匹配中"):
            max_similarity = 0
            best_match = None

            for product_name, product_embedding in zip(product_names, product_embeddings):
                # 提取 NER 結果
                similarity_result = process_match(momo_title, product_name, momo_embedding,
                                                  product_embedding, momo_ner, product_ner, threshold, attribute_weights)
                if similarity_result[2] > max_similarity:
                    max_similarity = similarity_result[2]
                    best_match = similarity_result[1]

            if max_similarity >= threshold:
                # 儲存原始商品名稱而不是處理過的名稱
                all_products.append((original_momo_titles[momo_titles.index(momo_title)],
                                    original_product_names[product_names.index(best_match)], max_similarity))
            else:
                all_products.append(
                    (original_momo_titles[momo_titles.index(momo_title)], None, None))

        # 統一 DataFrame 結構
        matched_products = [
            item for item in all_products if item[1] is not None]
        unmatched_products = [item for item in all_products if item[1] is None]
        final_df = pd.DataFrame(matched_products + unmatched_products,
                                columns=['MOMO Title', 'Product Title', 'Similarity'])

        output_folder = matched_results_folder
        os.makedirs(output_folder, exist_ok=True)

        output_file_path = os.path.join(output_folder, f'{search_term}.csv')
        final_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f'檔案已儲存至 {output_file_path}')


def product_matching_v2():
    for search_term in search_terms:
        print(f'current term: {search_term}')

        output_folder = matched_results_folder
        os.makedirs(output_folder, exist_ok=True)

        # Special case
        if search_term == '[馬玉山] 紫山藥黑米仁 (30g*12入/袋)':
            search_term = '[馬玉山] 紫山藥黑米仁 (30gx12入x袋)'

        # Skip if file already exists
        t_matched_file_name = f'./{matched_results_folder}/{search_term}.csv'
        if os.path.exists(t_matched_file_name):
            print(f'file already exists: {t_matched_file_name}')
            continue

        # Load product names from Coupang
        original_product_names = load_product_data(
            f'./{coupang_searched_results_folder}/{student_id}_{search_term}.csv')
        if original_product_names is None:
            continue
        product_names = [preprocess_name(name)
                         for name in original_product_names]

        # Fetch products from MOMO
        momo_search_term = search_term
        momo_products = scrapy_MOMO(momo_search_term)
        original_momo_titles = momo_products['Title'].tolist()
        momo_titles = [preprocess_name(title)
                       for title in original_momo_titles]

        if not momo_titles:
            print(f'search term: {momo_search_term} is empty')
            continue

        # Filter words with TF-IDF
        filtered_momo_titles = filter_important_words(
            momo_titles, tfidf_vectorizer)
        filtered_product_names = filter_important_words(
            product_names, tfidf_vectorizer)

        # Get NER tags for both MOMO and Coupang products
        momo_ner = inference_api.get_ner_tags(
            NER_model, NER_tokenizer, filtered_momo_titles, all_attribute)
        product_ner = inference_api.get_ner_tags(
            NER_model, NER_tokenizer, filtered_product_names, all_attribute)

        # Compute embeddings
        momo_embeddings = inference(tokenizer, model, filtered_momo_titles)
        product_embeddings = inference(
            tokenizer, model, filtered_product_names)

        all_products = []

        # Matching each MOMO product with Coupang products
        for momo_title, momo_embedding, momo_tags in zip(momo_titles, momo_embeddings, momo_ner):
            best_match = None
            max_score = 0

            for product_name, product_embedding, product_tags in zip(product_names, product_embeddings, product_ner):
                attribute_score = 0
                matched_attributes = 0

                # For each attribute, calculate semantic similarity
                for attribute, momo_attr_values in momo_tags.items():
                    if attribute in product_tags:
                        product_attr_values = product_tags[attribute]

                        # Compare attribute values for similarity
                        for (momo_value, momo_conf), (product_value, product_conf) in zip(momo_attr_values, product_attr_values):
                            if momo_value * product_value > 0.5:
                                attribute_score += 1  # Fully matched attribute
                                break
                        matched_attributes += 1  # Count this attribute as compared

                # Calculate similarity as the percentage of matched attributes
                if matched_attributes > 0:
                    total_similarity = attribute_score / matched_attributes
                else:
                    total_similarity = 0

                # Keep track of the highest similarity match
                if total_similarity > max_score:
                    max_score = total_similarity
                    best_match = product_name

            # Save best match if similarity exceeds threshold
            if max_score >= threshold:
                all_products.append((original_momo_titles[momo_titles.index(momo_title)],
                                     original_product_names[product_names.index(best_match)], max_score))
            else:
                all_products.append(
                    (original_momo_titles[momo_titles.index(momo_title)], None, None))

        # Create DataFrame with results
        matched_products = [
            item for item in all_products if item[1] is not None]
        unmatched_products = [item for item in all_products if item[1] is None]
        final_df = pd.DataFrame(matched_products + unmatched_products,
                                columns=['MOMO Title', 'Product Title', 'Similarity'])

        output_file_path = os.path.join(output_folder, f'{search_term}.csv')
        final_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f'檔案已儲存至 {output_file_path}')


# %% 主程式邏輯
student_id = 'M11207424'
threshold = 0.4
# 儲存 coupang 搜尋結果的資料夾
coupang_searched_results_folder = 'coupang_search_results_M11207424'
# momo 搜尋結果
momo_searched_results_folder = 'momo_search_results_100'
# 匹配結果
matched_results_folder = 'matching_results'

# 使用 jieba_tokenizer 進行分詞
tokenizer = jieba_tokenizer

# 建立 TfidfVectorizer 物件
tfidf_vectorizer = TfidfVectorizer(
    token_pattern=None, tokenizer=tokenizer, ngram_range=(1, 3))


# 加載 semantic 模型
model, tokenizer = get_semantic_model()

# load NER model
NER_model, NER_tokenizer = inference_api.load_model(
    "clw8998/Product-Name-NER-model", device=config.device)

all_attribute = ['品牌', '名稱', '產品', '產品序號', '顏色', '材質', '對象與族群', '適用物體、事件與場所',
                 '特殊主題', '形狀', '圖案', '尺寸', '重量', '容量', '包裝組合', '功能與規格']

check_att = ['品牌', '名稱', '產品', '顏色']

# 品牌匹配權重較高
attribute_weights = {
    "品牌": 5,
    "名稱": 5,
    "產品": 5,
    "顏色": 3,
}

# 讀取搜尋詞和產品名稱
search_terms = load_search_terms(f'./queries/{student_id}_queries.txt')

# %% 進行匹配流程
product_matching()
