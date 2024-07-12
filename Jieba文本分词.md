````r
library(jiebaR)
library(text2vec)
library(readtext)
````

# 导入中文Word文档
````r
file_path <- file.choose()
text_data <- readtext(file_path, encoding = "UTF-8")
````

# 提取和显示文本内容
````r
chinese_text <- as.character(text_data$text)
cat(chinese_text)
````

# 创建一个分词器对象
````r
cutter <- worker()
````

# 对文本进行分词
````
segmented_text <- segment(chinese_text, cutter)
print(segmented_text)
````

# 使用 text2vec 包将文本转换为向量
````r
# 创建一个词袋（Bag of Words）模型
corpus <- list(segmented_text)
it <- itoken(corpus, progressbar = FALSE)
vectorizer <- create_vocabulary(it) %>%
  vocab_vectorizer()
dtm <- create_dtm(it, vectorizer)
print(dtm)
````
