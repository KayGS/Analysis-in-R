## 该文档包含：XGBOOST, SVR,GBT,Bayesian, ElasticNet

````R
library(dplyr)
library(caret)
library(car)
library(xgboost)
library(glmnet)
library(e1071)
library(gbm)
library(stats)
library(base)
library(carData)
library(ggplot2)
library(lattice)
library(Matrix)
````
````R
# 读取数据
data <- read.csv(file.choose())

# 查看数据结构
str(data)

# 将必要的变量转化为因子（分类数据）
data$Gender <- as.factor(data$Gender)
data$CampaignChannel <- as.factor(data$CampaignChannel)
data$CampaignType <- as.factor(data$CampaignType)
data$AdvertisingPlatform <- as.factor(data$AdvertisingPlatform)
data$AdvertisingTool <- as.factor(data$AdvertisingTool)

# 检查因子变量是否有单一水平
factor_vars <- c("Gender", "CampaignChannel", "CampaignType", "AdvertisingPlatform", "AdvertisingTool")
for (var in factor_vars) {
  print(levels(data[[var]]))
}
````
> [1] "Female" "Male"  
> [1] "Email"        "PPC"          "Referral"     "SEO"          "Social Media"
> [1] "Awareness"     "Consideration" "Conversion"    "Retention"    
> [1] "IsConfid" 这个是单一
> [1] "ToolConfid" 这个是单一

````R
# 标准化数值型数据
num_vars <- c("Age", "Income", "AdSpend", "ClickThroughRate", "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks", "PreviousPurchases", "LoyaltyPoints")
data[num_vars] <- scale(data[num_vars])

# 删除单一水平的因子变量
df <- within(data, rm(`AdvertisingPlatform`,`AdvertisingTool`))

# 将因子变量转化为哑变量
data_dummy <- model.matrix(ConversionRate ~ ., data = df)[,-1]
data_dummy <- as.data.frame(data_dummy)

# 拆分数据集为训练集和测试集
set.seed(123)
trainIndex <- createDataPartition(df$ConversionRate, p = .8, list = FALSE)
trainData <- data_dummy[trainIndex, ]
testData <- data_dummy[-trainIndex, ]

# 添加目标变量
trainData$ConversionRate <- df$ConversionRate[trainIndex]
testData$ConversionRate <- df$ConversionRate[-trainIndex]

# 使用VIF判断共线性问题
vif_values <- vif(lm(ConversionRate ~ ., data = trainData))
print(vif_values)

> <10没有共线性问题,10~100存在一定共线性问题,>100共线性问题很严重

# 使用特征值判断共线性问题
eigen_values <- eigen(cor(trainData[, num_vars]))$values
print(eigen_values)
> 多个自变量接近于0，说明存在严重共线性问题
````
# 构建XGBoost模型
````R
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -which(names(trainData) == "ConversionRate")]), label = trainData$ConversionRate)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -which(names(testData) == "ConversionRate")]), label = testData$ConversionRate)

# 目的：XGBoost模型需要将输入数据转化为DMatrix格式，这是一种优化的矩阵格式，有助于提高计算效率。
# 步骤：
# 使用as.matrix函数将训练集和测试集的自变量部分（即除去目标变量ConversionRate的部分）转换为矩阵格式。
# 使用xgb.DMatrix函数将上述矩阵和相应的目标变量（标签）转化为DMatrix格式。此步骤分别对训练集和测试集进行。

> 设置参数
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# 目的：为XGBoost模型设置训练参数，以便控制模型的行为和性能。
# 主要参数：
# objective：指定目标函数，这里使用“reg
# ”，表示回归任务且使用均方误差作为损失函数。
# eta：学习率，决定每次提升的步长，通常取值较小（如0.1）以防止过拟合。
# max_depth：树的最大深度，控制树的复杂度，防止过拟合。
# subsample：每次提升时使用的数据子样本比例，取值在(0,1]之间，值越小防止过拟合的效果越好。
# colsample_bytree：每次构建树时使用的特征子样本比例，类似于subsample，但作用于特征而非样本。

> 训练模型
model_xgb <- xgb.train(params = params, data = train_matrix, nrounds = 100)

# 目的：使用训练数据和设定的参数训练XGBoost模型。
# 步骤：
# params：前面设置的模型参数。
# data：训练数据的DMatrix格式。
# nrounds：提升（迭代）的次数，即树的数量。这里设定为100，意味着模型将训练100棵树。

> 预测
pred_xgb <- predict(model_xgb, newdata = test_matrix)

> 评估模型效果
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

rmse_xgb <- rmse(testData$ConversionRate, pred_xgb)
print(paste("XGBoost RMSE:", rmse_xgb))
````
# 构建Ridge回归模型
````R
model_bayesian <- lm.ridge(ConversionRate ~ ., data = trainData, lambda = seq(0, 10, 0.1))
# 目的：使用 Ridge 回归模型对训练数据进行拟合。
# 步骤：
# lm.ridge 函数用于进行 Ridge 回归。
# ConversionRate ~ . 表示使用所有其他变量来预测 ConversionRate。
# data = trainData 指定使用训练数据。
# lambda = seq(0, 10, 0.1) 设置 Ridge 回归中的正则化参数 λ 的范围，从 0 到 10，每次增加 0.1。

> 手动计算预测值
best_lambda <- model_bayesian$lambda[which.min(model_bayesian$GCV)]
# 目的：选择能够使广义交叉验证 (GCV) 值最小的 λ，这通常对应于最优的正则化强度。
# 步骤：
# which.min(model_bayesian$GCV) 找到使 GCV 值最小的索引。
# model_bayesian$lambda[...] 获取对应索引的 λ 值。
# coefficients <- coef(model_bayesian)[which.min(model_bayesian$GCV), ]
# 目的：提取使用最佳 λ 训练的 Ridge 回归模型的系数。
# 步骤：
# coef(model_bayesian) 提取 Ridge 回归模型的系数矩阵。
# [which.min(model_bayesian$GCV), ] 获取对应于最佳 λ 的系数。

> 添加截距项
testData_intercept <- cbind(1, as.matrix(testData[, -which(names(testData) == "ConversionRate")]))
# 目的：在测试数据中添加截距项，以便计算预测值。
# 步骤：
# as.matrix(testData[, -which(names(testData) == "ConversionRate")]) 将测试数据中的自变量部分转换为矩阵格式。
# cbind(1, ...) 在矩阵的第一列添加一列全为 1 的数据，表示截距项。

> 计算预测值
pred_bayesian <- testData_intercept %*% coefficients
# 解释：
# 
# 目的：使用 Ridge 回归模型的系数和测试数据计算预测值。
# 步骤：
# 矩阵乘法 testData_intercept %*% coefficients 计算预测值。

> 评估模型效果
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
# 目的：定义均方根误差 (RMSE) 函数来评估模型预测的准确性。
# 步骤：
# actual 是实际值，predicted 是预测值。
# mean((actual - predicted) ^ 2) 计算均方误差 (MSE)。
# sqrt(...) 计算均方根误差 (RMSE)。
# 步骤 3.2：计算 RMSE 并输出结果

rmse_bayesian <- rmse(testData$ConversionRate, pred_bayesian)
print(paste("Bayesian Ridge RMSE:", rmse_bayesian))
# 目的：计算 Ridge 回归模型在测试数据上的 RMSE，并输出结果。
# 步骤：
# rmse(testData$ConversionRate, pred_bayesian) 计算测试数据的实际值和预测值之间的 RMSE。
# print(...) 输出 RMSE 结果。
````

# ElasticNet
````R
x <- as.matrix(trainData[, -which(names(trainData) == "ConversionRate")])
y <- trainData$ConversionRate
model_elasticnet <- cv.glmnet(x, y, alpha = 0.5)
# 目的：使用交叉验证构建 ElasticNet 回归模型，并选择最佳的正则化参数 λ。
# 函数：cv.glmnet 是 glmnet 包中的函数，用于对 Generalized Linear Model (GLM) 执行交叉验证以找到最佳的正则化参数。
# 参数：
# x：自变量矩阵，训练数据中的所有特征（自变量）组成的矩阵。
# y：因变量向量，训练数据中的目标变量（因变量）组成的向量。
# alpha：ElasticNet 的混合参数，控制 L1 和 L2 正则化的比例。在这里，alpha = 0.5 表示 L1 和 L2 正则化的比例均衡（即 50% 的 L1 和 50% 的 L2）。
# 交叉验证 (Cross-Validation)
# 目的：通过将数据集分成多个子集（通常称为折叠），然后对每个子集进行训练和验证，从而评估模型的性能。
# 优势：通过交叉验证可以减少模型对特定训练集的依赖，提高模型在新数据上的泛化能力。

> ElasticNet 正则化
# ElasticNet 是一种结合了 L1 正则化（Lasso）和 L2 正则化（Ridge）的线性回归模型。
# 
# L1 正则化（Lasso）：可以产生稀疏模型（即某些系数可以被完全缩小到零），有助于特征选择。
# L2 正则化（Ridge）：可以防止模型过拟合，通过缩小系数来减少模型复杂度。
# alpha 参数：
# 
# alpha = 0：相当于 Ridge 回归，只使用 L2 正则化。
# alpha = 1：相当于 Lasso 回归，只使用 L1 正则化。
# 0 < alpha < 1：ElasticNet 正则化，L1 和 L2 正则化的混合。

> 交叉验证结果
# cv.glmnet 函数返回的对象包含交叉验证的结果，包括每个 λ 值对应的均方误差 (MSE)。
# 通过 cv.glmnet 的结果，我们可以选择使交叉验证误差最小的 λ 值，从而构建最优的 ElasticNet 模型。

正则化，岭回归一些补充解释
直观解释
假设你在用模型预测房价，并且你有很多变量，比如房子的面积、卧室数量、位置等等。如果你用普通的回归方法，可能会得到一个过于复杂的模型，过度拟合你的数据。
L1 正则化：就像在选择变量，自动去掉那些不太重要的变量。比如，如果你发现位置比卧室数量更重要，L1 正则化可能会将卧室数量的系数缩小到零，只保留位置和其他重要变量。
L2 正则化：则是将所有变量的影响都缩小一些，但不会完全去掉任何变量。这样，你的模型会更加稳定，不会过度依赖某些变量。
选择正则化方法
如果你希望自动选择变量，得到一个稀疏模型，可以选择 L1 正则化。
如果你希望处理多重共线性，并且希望所有变量都对模型有贡献，可以选择 L2 正则化。
有时候，结合两者的优势，使用 ELASTIC NET 正则化（同时包括 L1 和 L2）也是一个不错的选择。
````

# SVR
````R
model_svr <- svm(ConversionRate ~ ., data = trainData)

# 支持向量回归 (SVR)：
# 
# SVR 是支持向量机 (SVM) 的一种扩展，专门用于回归任务。
# 它通过寻找最优的超平面，将训练数据映射到高维空间，并尽可能地保持数据点与超平面之间的距离在某个范围内，从而实现回归预测。
# 核心思想：
# 
# SVR 的目标是找到一个函数，使得大多数数据点与该函数的偏差在一个指定的阈值（ε）内。
# 通过引入惩罚参数（C），SVR 允许一些数据点偏离阈值，但会对偏离程度进行惩罚，从而平衡模型的复杂性和预测能力。
# 参数解释
# 公式：ConversionRate ~ .
# 
# ConversionRate：因变量（目标变量）。
# ~ .：使用所有其他变量（自变量）来预测因变量。
# 数据：data = trainData
# 
# trainData：包含训练数据的 data.frame，包括自变量和因变量。
# SVR 的优势
# 高维特征处理能力：SVR 能够处理高维空间中的数据，有效捕捉复杂数据结构。
# 鲁棒性：SVR 对噪声和异常值有较好的鲁棒性。
# 非线性映射：通过使用核函数，SVR 能够处理非线性关系的数据。
````

# GBR
````R
model_gbr <- gbm(ConversionRate ~ ., data = trainData, distribution = "gaussian", n.trees = 100)
# 解释：
# 
# gbm 函数用于构建梯度提升模型。
# ConversionRate ~ .：使用所有其他变量（自变量）来预测 ConversionRate（因变量）。
# data = trainData：指定用于训练模型的数据集。
# distribution = "gaussian"：指定损失函数类型为高斯分布，适用于回归任务。
# n.trees = 100：指定提升树的数量为 100。
# 梯度提升回归 (GBR) 简介
# 梯度提升回归 (GBR)：
# 
# GBR 是一种集成学习方法，通过结合多个弱学习器（通常是决策树）来提高模型的预测性能。
# 每一步都在前一步的基础上，最小化残差（即预测值与真实值之间的差距）。
# 核心思想：
# 
# 通过迭代地添加树来纠正之前模型的错误，逐步减少预测误差。
# 每棵树在训练时关注之前树未能很好拟合的数据点，分配更高的权重给这些数据点。
# 参数解释
# 公式：ConversionRate ~ .
# 
# ConversionRate：因变量（目标变量）。
# ~ .：使用所有其他变量（自变量）来预测因变量。
# 数据：data = trainData
# 
# trainData：包含训练数据的 data.frame，包括自变量和因变量。
# 分布：distribution = "gaussian"
# 
# 指定损失函数类型为高斯分布，适用于回归任务。
# 树的数量：n.trees = 100
# 
# 指定提升树的数量为 100。更多的树通常能提高模型的性能，但也增加了计算开销。


> 预测
#pred_bayesian <- predict(model_bayesian, newdata = testData)
#pred_xgb <- predict(model_xgb, newdata = as.matrix(testData[, -which(names(testData) == "ConversionRate")]))
pred_elasticnet <- predict(model_elasticnet, newx = as.matrix(testData[, -which(names(testData) == "ConversionRate")]))
pred_svr <- predict(model_svr, newdata = testData)
pred_gbr <- predict(model_gbr, newdata = testData, n.trees = 100)

> 评估模型效果
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

results <- data.frame(
  Model = c("Bayesian Ridge", "XGBR", "ElasticNet", "SVR", "GBR"),
  RMSE = c(rmse(testData$ConversionRate, pred_bayesian),
           rmse(testData$ConversionRate, pred_xgb),
           rmse(testData$ConversionRate, pred_elasticnet),
           rmse(testData$ConversionRate, pred_svr),
           rmse(testData$ConversionRate, pred_gbr))
)

print(results)
# Bayesian Ridge:
#   
#   RMSE (Root Mean Square Error): 0.05504598
# 这个模型的表现较好，RMSE较低，说明其预测误差较小。
# XGBR (XGBoost Regression):
#   
#   RMSE: 0.05591660
# XGBoost模型的表现也不错，RMSE稍高于Bayesian Ridge，说明其预测误差略大。
# ElasticNet:
#   
#   RMSE: 0.05493810
# ElasticNet模型表现最好，RMSE最低，说明其预测误差最小。
# SVR (Support Vector Regression):
#   
#   RMSE: 0.05750442
# SVR模型的表现相对较差，RMSE最高，说明其预测误差最大。
# GBR (Gradient Boosting Regression):
#   
#   RMSE: 0.05492428
# GBR模型表现很好，RMSE仅次于ElasticNet，说明其预测误差也很小。
# 总结
# 最佳模型: 从RMSE值来看，ElasticNet (RMSE = 0.05493810) 是表现最好的模型，其次是GBR (RMSE = 0.05492428) 和Bayesian Ridge (RMSE = 0.05504598)。
# 建议使用的模型: 由于ElasticNet模型的RMSE最低，建议使用ElasticNet模型进行转化率的预测。
````
# 合并预测结果
````R
testData$Pred_Bayesian <- pred_bayesian
testData$Pred_XGB <- pred_xgb
testData$Pred_ElasticNet <- pred_elasticnet
testData$Pred_SVR <- pred_svr
testData$Pred_GBR <- pred_gbr

> 绘制趋势对比图
ggplot() +
  geom_line(aes(x = 1:nrow(testData), y = testData$ConversionRate), color = "blue", size = 1) +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_Bayesian), color = "red", size = 1) +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_XGB), color = "green", size = 1) +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_ElasticNet), color = "purple", size = 1) +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_SVR), color = "orange", size = 1) +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_GBR), color = "pink", size = 1) +
  labs(title = "Conversion Rate Prediction Comparison", x = "Sample Index", y = "Conversion Rate") +
  theme_minimal()


> SVR模型效果最好
best_model <- model_elasticnet

> 样本数据预测
sample_data <- testData[1, -which(names(testData) == "ConversionRate")]
predicted_value <- predict(best_model, newx = as.matrix(sample_data))
print(predicted_value)


> 绘制趋势对比图
ggplot() +
  geom_line(aes(x = 1:nrow(testData), y = testData$ConversionRate), color = "blue", size = 1, linetype = "solid") +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_Bayesian), color = "red", size = 1, linetype = "dashed") +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_XGB), color = "green", size = 1, linetype = "dotted") +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_ElasticNet), color = "purple", size = 1, linetype = "dotdash") +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_SVR), color = "orange", size = 1, linetype = "longdash") +
  geom_line(aes(x = 1:nrow(testData), y = testData$Pred_GBR), color = "pink", size = 1, linetype = "twodash") +
  labs(title = "Conversion Rate Prediction Comparison",
       x = "Sample Index",
       y = "Conversion Rate") +
  theme_minimal() +
  scale_color_manual(name = "Model",
                     values = c("Actual" = "blue", "Bayesian" = "red", "XGB" = "green", "ElasticNet" = "purple", "SVR" = "orange", "GBR" = "pink"))
````

# 如果存在共线性，需要降维PCR，然后再回归，可用下面方法
````R
> 标准化数值型数据
num_vars <- c("Age", "Income", "AdSpend", "ClickThroughRate", "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks", "PreviousPurchases", "LoyaltyPoints")
trainData_scaled <- trainData
trainData_scaled[num_vars] <- scale(trainData[num_vars])
testData_scaled <- testData
testData_scaled[num_vars] <- scale(testData[num_vars])

> 执行主成分分析 (PCA)
pca <- prcomp(trainData_scaled[, num_vars], scale. = TRUE)
summary(pca)
# Importance of components:
#                           PC1     PC2     PC3     PC4     PC5     PC6     PC7     PC8     PC9    PC10    PC11    PC12
# Standard deviation     1.04071 1.02537 1.02025 1.00885 1.00300 1.00164 0.99556 0.99412 0.98761 0.97944 0.97386 0.96699
# Proportion of Variance 0.09026 0.08762 0.08674 0.08482 0.08383 0.08361 0.08259 0.08236 0.08128 0.07994 0.07903 0.07792
# Cumulative Proportion  0.09026 0.17787 0.26461 0.34943 0.43326 0.51687 0.59946 0.68182 0.76310 0.84304 0.92208 1.00000
# 标准差 (Standard deviation)
# 定义：每个主成分（PC）的标准差，表示该主成分在原数据中的分布范围。
# 含义：标准差越大，主成分在原数据中的变异程度越大，表示该主成分捕捉到了更多的原始数据变异。
# 例如，PC1 的标准差为 1.04071，表示 PC1 在原数据中的分布范围最大。
# PC12 的标准差为 0.96699，表示 PC12 在原数据中的分布范围最小。

# 方差比例 (Proportion of Variance)
# 定义：每个主成分所解释的原始数据方差比例。
# 含义：方差比例越大，该主成分解释的数据变异越多，表示该主成分的重要性越高。
# 例如，PC1 的方差比例为 0.09026，表示 PC1 解释了原始数据 9.026% 的方差。
# PC12 的方差比例为 0.07792，表示 PC12 解释了原始数据 7.792% 的方差。

# 累积方差比例 (Cumulative Proportion)
# 定义：前 k 个主成分所解释的累计方差比例。
# 含义：累积方差比例表示前 k 个主成分解释了原始数据的总变异程度。
# 例如，前两个主成分 (PC1 + PC2) 的累积方差比例为 0.17787，表示前两个主成分解释了原始数据 17.787% 的总方差。
# 前六个主成分 (PC1 + ... + PC6) 的累积方差比例为 0.51687，表示前六个主成分解释了原始数据 51.687% 的总方差。

# 选择主成分的数量
# 原则：选择能够解释大部分变异的主成分数量，同时避免过多的主成分导致模型过于复杂。
# 方法：通常选择累积方差比例达到某个阈值（例如 80% 或 90%）的前几个主成分。


> 选择前 n 个主成分
n_components <- 10  # 根据需要调整 n 的值
trainData_pca <- as.data.frame(pca$x[, 1:n_components])
testData_pca <- as.data.frame(pca$x[, 1:n_components])

> 将因变量添加到降维后的数据中
trainData_pca$ConversionRate <- trainData$ConversionRate
testData_pca$ConversionRate <- testData$ConversionRate

> 构建XGBoost模型
train_matrix <- xgb.DMatrix(data = as.matrix(trainData_pca[, -which(names(trainData_pca) == "ConversionRate")]), label = trainData_pca$ConversionRate)
test_matrix <- xgb.DMatrix(data = as.matrix(testData_pca[, -which(names(testData_pca) == "ConversionRate")]), label = testData_pca$ConversionRate)
head(train_matrix)

> 设置参数
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

> 训练模型
model_xgb <- xgb.train(params = params, data = train_matrix, nrounds = 100)

> 预测
pred_xgb <- predict(model_xgb, newdata = test_matrix)

> 构建支持向量回归 (SVR) 模型
model_svr <- svm(ConversionRate ~ ., data = trainData_pca)

> 预测
pred_svr <- predict(model_svr, newdata = testData_pca)

> 构建梯度提升回归 (GBT) 模型
model_gbr <- gbm(ConversionRate ~ ., data = trainData_pca, distribution = "gaussian", n.trees = 100)

> 预测
pred_gbr <- predict(model_gbr, newdata = testData_pca, n.trees = 100)

> 定义 RMSE 评估函数
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

> 计算和输出 RMSE
rmse_xgb <- rmse(testData_pca$ConversionRate, pred_xgb)
rmse_svr <- rmse(testData_pca$ConversionRate, pred_svr)
rmse_gbr <- rmse(testData_pca$ConversionRate, pred_gbr)

print(paste("XGBoost RMSE:", rmse_xgb))
print(paste("SVR RMSE:", rmse_svr))
print(paste("GBT RMSE:", rmse_gbr))
````
