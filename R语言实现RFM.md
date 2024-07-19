# 包含两种方法计算权重：赋值，随机森林
```r
> 加载必要的R包
library(dplyr)
library(tidyr)
library(ggplot2)
library(knitr)

> 读取数据
data <- read.csv(file.choose())

> 检查数据结构
str(data)

# 计算RFM值
# Recency: 使用 membership_years 作为近度指标（假设数据是近年数据）
# Frequency: 使用 purchase_frequency 作为频度指标
# Monetary: 使用 last_purchase_amount 作为货币值指标

rfm_data <- data %>%
  mutate(Recency = 2024 - membership_years,  # 假设数据年份为2024年
         Frequency = purchase_frequency,
         Monetary = last_purchase_amount) %>%
  select(id, Recency, Frequency, Monetary)

str(rfm_data)

> 对RFM值进行评分（1-5分）
rfm_data <- rfm_data %>%
  mutate(R_score = ntile(Recency, 5),
         F_score = ntile(Frequency, 5),
         M_score = ntile(Monetary, 5))

> 计算RFM总分
rfm_data <- rfm_data %>%
  mutate(RFM_score = R_score * 100 + F_score * 10 + M_score)

> 显示前几行数据
kable(head(rfm_data))

> RFM得分分布
rfm_distribution <- rfm_data %>%
  group_by(RFM_score) %>%
  summarise(Count = n())

ggplot(rfm_distribution, aes(x = RFM_score, y = Count)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "RFM Score Distribution",
       x = "RFM Score",
       y = "Number of Customers")

> RFM矩阵
rfm_matrix <- rfm_data %>%
  group_by(R_score, F_score, M_score) %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = M_score, values_from = Count, values_fill = list(Count = 0))
# 识别用户分布：RFM矩阵可以帮助我们识别用户在不同RFM评分组合下的分布情况，了解哪些类型的用户数量最多或最少。

kable(rfm_matrix, caption = "RFM Matrix")

> 客户细分
rfm_segments <- rfm_data %>%
  mutate(Segment = case_when(
    R_score >= 4 & F_score >= 4 & M_score >= 4 ~ "Best Customers",
    R_score >= 3 & F_score >= 3 & M_score >= 3 ~ "Loyal Customers",
    R_score == 5 ~ "Recent Customers",
    F_score == 5 ~ "Frequent Customers",
    M_score == 5 ~ "High Spending Customers",
    TRUE ~ "Others"
  ))

> 细分客户数量
segment_distribution <- rfm_segments %>%
  group_by(Segment) %>%
  summarise(Count = n())

ggplot(segment_distribution, aes(x = Segment, y = Count, fill = Segment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Customer Segmentation",
       x = "Segment",
       y = "Number of Customers") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

> 显示细分后的前几行数据
kable(head(rfm_segments))
```

# 用随机森林来决定RFM的权重
````R
> 加载必要的包
library(randomForest)
library(rgl)
library(dplyr)

> 读取数据
data <- read.csv(file.choose())

> 计算RFM值
# R值：以年份计算（假设数据截止到2024年）
data$Recency <- 2024 - data$membership_years

> F值：购买频率
data$Frequency <- data$purchase_frequency

> M值：最后一次购买金额
data$Monetary <- data$last_purchase_amount

> 创建RFM数据框
rfm_data <- data %>% select(Recency, Frequency, Monetary)

> 使用随机森林模型来确定RFM的权重
set.seed(123)
rfm_model <- randomForest(Monetary ~ Recency + Frequency, data = rfm_data)

> 获取变量的重要性
importance <- importance(rfm_model)

# No. of variables tried at each split: 1
# 在每个节点分裂时，随机选择一个变量进行分裂，这是随机森林的一部分，旨在增加模型的多样性和鲁棒性。
# Mean of squared residuals: 94812.26
# 这是模型的平均平方残差 (Mean Squared Residuals, MSR)，即预测值与实际值之间差异的平方和的平均值。这个值越小，模型的预测精度越高。
# % Var explained: -8.51
# 这是模型解释的方差百分比 (Percentage of Variance Explained)，即模型解释了目标变量总方差的百分比。一般来说，这个值应为正数，表示模型在一定程度上能够解释目标变量的变化。
# 解释：
# 
# 平均平方残差 (94812.26) 较大，表明模型的预测误差较高。
# 负的解释方差百分比 (-8.51%) 表明模型的表现非常差，甚至不如简单的均值模型（即总是预测目标变量的平均值）。这个值为负数表示模型的预测效果不佳，可能是由于过拟合、欠拟合或数据本身的问题。
# 优化建议：
# 
# 数据预处理：检查数据的质量，处理缺失值和异常值，确保数据的完整性和一致性。
# 特征工程：增加新的特征，或转换现有特征，以更好地捕捉数据中的信息。
# 超参数调优：尝试不同的参数组合，例如增加树的数量，调整每个节点分裂时的变量数，或者修改其他超参数。
# 交叉验证：使用交叉验证方法来评估模型的稳定性和性能。
# 尝试其他模型：除了随机森林，还可以尝试其他回归模型，如线性回归、支持向量机、梯度提升树等。

> 打印变量的重要性
print(importance)

> RFM分析报告
rfm_report <- data.frame(
  Variable = rownames(importance),
  Importance = importance[, 1]
)

> 添加Monetary默认权重
rfm_report <- rbind(rfm_report, data.frame(Variable = "Monetary", Importance = mean(rfm_report$Importance)))

print(rfm_report)

> 将重要性值标准化为权重
rfm_weights <- rfm_report$Importance / sum(rfm_report$Importance)

str(data)

> 计算RFM分数
data$RFM_Score <- with(data, 
                       rfm_weights[1] * data$Recency + 
                         rfm_weights[2] * data$Frequency + 
                         rfm_weights[3] * data$Monetary
)


> 打印前几行数据以验证结果
print(head(data))

> 3D柱形图展示RFM模型结果
plot3d(
  rfm_data$Recency,
  rfm_data$Frequency,
  rfm_data$Monetary,
  col = "blue",
  xlab = "Recency",
  ylab = "Frequency",
  zlab = "Monetary",
  type = "h"
)
# 使用不同的算法：除了随机森林，还可以尝试其他机器学习算法，如线性回归、支持向量机、梯度提升树等，来比较各个算法的表现，并选择最优的算法。
# 
# 超参数调优：对于随机森林模型，可以通过网格搜索或随机搜索对超参数（如树的数量、最大深度、最小样本分裂数等）进行调优，以提高模型的性能。
# 
# 交叉验证：使用交叉验证方法来评估模型的稳定性和性能，以确保模型在不同的数据集上都有良好的表现。
# 
# 特征选择与工程：对输入特征进行选择和工程，可能会提高模型的表现。例如，可以创建新的特征、转换现有特征、或者删除不相关或冗余的特征。
# 
# 数据预处理：确保数据质量，如处理缺失值、异常值、标准化和归一化数据等，这些都可以对模型性能产生影响。
````
