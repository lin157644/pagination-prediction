```python
token_feature = {
        'text-exact': replace_digits(text.strip()[:100].strip()),
        'query': query_param_names_ngrams: list(),
        'parent-tag': parent: str,
        'class':_as_list(ngrams_wb(css_classes, 4, 5),
                          AUTOPAGER_LIMITS.max_css_features),
        'text': _as_list(ngrams_wb(replace_digits(text), 2, 5),
                         AUTOPAGER_LIMITS.max_text_features),
        'text-full': ''
}
```

class 與 text 是使用 ngrams_wb 製作的 Embedding，text 實際上沒有被用到

`token_features = get_text_around_selector_list() # text-full = sibling's text-exact around node's text-exact.`
`link_to_features(): Text contecnt of the link otherwise alt or img`
**text-full** pattern: `f'{before},{Plink_to_features},{after}'`

----

word_to_vector 有兩種方式Fasttext與Laser，其中Fasttext未實際使用。

Sentence(text content) embedding
`laser.getSentenceVector(word_list)`
-> `laser.embed_sentences(sents, lang=self.lang)[0]`

laser_full_tokens_emb

leaser做'text-full'的embedding

----

Parent tag feature(ptags)
ptags_vector = get_ptags_vector(token_features, data_map_for_ptag)

Parent tag 由大到小排序
假如為 `[('li', 100), ('div', 50), ('span', 30)]`
一 node 的 parent tag為 div 則 `ptags_vector = nd.array([0, 1, 0])`

----

Class feature (class="XXX")
Query feature (?q=XXX)

token_feature 裡的 class 有加上 parent class
self_and_children_classes = ' '.join(link.xpath(".//@class").extract())
css_classes = normalize(parent_classes + ' ' + self_and_children_classes)

用訓練資料製作一個Tokenizer
由該class/query name出現的次數排序，
輸入一個tag，對應到一個數字(在排序後的index)

然後做padding
pages_class, pages_query = prepare_input_ids(token_features, max_len)

---

tag_features: autopager的特徵工程製成之向量
```python
tag_feature: dict[bool] = {
        'isdigit', 'isalpha' ,'has-href' ,'path-has-page' ,'path-has-pageXX' ,'path-has-number'
        'href-has-year' ,'class-has-disabled'
}
```

---

```pyton
train_composite_with_token = [
    laser_full_tokens_emb,
    ptags_vector
    pages_class,
    pages_query,
    tag_features
]
```

拿data_all.csv html_all訓練

----

BiLSTM-CRF 可讓我們結果是 PREV PAGE ... PAGE NEXT 的形式

----

Macro 每一類之間的權重相同

Micro 每一預測結果之間的權重相同

F1 = 2 x (precision x recall) / (precision + recall)

準確率（Accuracy）= (tp+tn)/(tp+fp+fn+tn)
精確率（Precision）= tp/(tp+fp)，即陽性的樣本中有幾個是預測正確的。
召回率（Recall）= tp/(tp+fn)，即事實為真的樣本中有幾個是預測正確的。

When true positive + false positive == 0, precision is undefined.
When true positive + false negative == 0, recall is undefined.

----

Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.

`libdevice.10.bc`

網域不同 挑掉?

----

Embedding output_dim 是 32
AveragePooling2D pool_size=(256, 1)

這啥??

tf.nn.embedding_lookup 相當於 keras.layers.Embedding ??

參考URLNet
default_emb_dim = 32