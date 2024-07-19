# 该算法涉及：XGBoost， RandomForest，SVM

````R
library(caret)
library(dplyr)
library(ROSE)
library(xgboost)
library(randomForest)
library(e1071)
library(pROC)
library(ggplot2)
library(NLP)
library(lattice)
library(stats)
library(base)

> 读取数据
data <- read.csv(file.choose())
> 查看数据结构
str(data)

>  数据清洗和缺失值处理
> 检查缺失值
sum(is.na(data))
> 用中位数填充数值变量的缺失值
num_vars <- sapply(data, is.numeric)
data[num_vars] <- lapply(data[num_vars], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
> 用众数填充非数值变量的缺失值
cat_vars <- sapply(data, is.factor)
Mode <- function(x) { ux <- unique(x); ux[which.max(tabulate(match(x, ux)))] }
data[cat_vars] <- lapply(data[cat_vars], function(x) ifelse(is.na(x), Mode(x), x))
> 检查缺失值是否填充完毕
sum(is.na(data))
> 将因子型变量转换为数值型
data$Geography <- as.numeric(factor(data$Geography))
data$Gender <- as.numeric(factor(data$Gender))
data$Card.Type <- as.numeric(factor(data$Card.Type))
> 删除surname
data <- within(data, rm(`Surname`))
> 将IsActiveMember转换为因子类型
data$IsActiveMember <- as.factor(data$IsActiveMember)
````

# 样本均衡处理
````R
> 使用ROSE库中的ovun.sample函数进行样本均衡处理
data_balanced <- ovun.sample(IsActiveMember ~ ., data = data, method = "both", p = 0.5, seed = 1)$data
# "both" 表示同时使用过采样和欠采样的组合方法来平衡数据集。
# 过采样：增加少数类别样本的数量。
# 欠采样：减少多数类别样本的数量。
# p = 0.5:
#   p 参数是过采样和欠采样的比例。p = 0.5 表示目标变量的正负样本将均衡到一个比例（即正负样本的数量相等）。
# seed = 1:
#   seed 参数用于设置随机种子，以确保每次运行代码时结果是一致的。这有助于结果的可重复性。

> 拆分数据集
set.seed(123)
trainIndex <- createDataPartition(data_balanced$IsActiveMember, p = .7, list = FALSE, times = 1)
# 作用：根据IsActiveMember列的值，生成一个索引，将70%的数据划分为训练集，30%的数据用于其他用途（如测试集）。
trainData <- data_balanced[trainIndex,]
# 从data_balanced数据集中选取70%的数据作为训练集，并将其存储在trainData变量中。
tempData <- data_balanced[-trainIndex,]
# 从data_balanced数据集中排除已经被选为训练集的70%的数据，剩余的30%的数据存储在tempData变量中，通常用于验证或测试模型。
testIndex <- createDataPartition(tempData$IsActiveMember, p = .5, list = FALSE, times = 1)
testData <- tempData[testIndex,]
validationData <- tempData[-testIndex,]
````

# 三种分类模型思路
````R
> Boosting: 使用xgboost
dtrain <- xgb.DMatrix(data = as.matrix(trainData[, -which(names(trainData) == "IsActiveMember")]), label = as.numeric(trainData$IsActiveMember)-1)
dtest <- xgb.DMatrix(data = as.matrix(testData[, -which(names(testData) == "IsActiveMember")]), label = as.numeric(testData$IsActiveMember)-1)
params <- list(objective = "binary:logistic", eval_metric = "auc")
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)
#dtrain 是训练数据的矩阵格式，专门为XGBoost设计的 xgb.DMatrix 类型。
# 将 trainData 中除了 IsActiveMember 列的所有列转换为矩阵。which(names(trainData) == "IsActiveMember") 找到 IsActiveMember 列的位置，而 -which(...) 则是排除该列。
# 将 IsActiveMember 列中的值转换为数值，并将其值减去1（假设原始值是1和2，这样转换后变为0和1），作为标签。
# params 是XGBoost的参数列表。
# objective = "binary:logistic" 表示这是一个二分类问题，使用逻辑回归作为目标函数。
# eval_metric = "auc" 指定评估指标为AUC（Area Under Curve，曲线下面积），用于评估模型性能。
# xgb_model 是训练好的XGBoost模型。
# xgb.train 函数用于训练模型。
# params = params 使用前面定义的参数。
# data = dtrain 使用训练数据。
# nrounds = 100 指定训练的轮数，即模型将进行100次迭代。

> Bagging: 使用随机森林
rf_model <- randomForest(IsActiveMember ~ ., data = trainData, ntree = 100)
# Bagging 基本概念
# Bagging 通过以下步骤来提高模型性能：
# 
# Bootstrap 采样：从训练数据集中随机有放回地抽取多个子集。每个子集的大小与原始训练集相同。
# 训练多个模型：在每个子集上训练一个独立的模型。
# 集成：对于回归问题，通过对所有模型的预测结果取平均值；对于分类问题，通过投票选择最终的预测结果。
# 随机森林
# 随机森林是一种扩展了 Bagging 方法的集成学习技术。它通过构建多个决策树并将它们的结果结合起来进行预测，进一步提高模型的准确性和稳定性。与传统决策树相比，随机森林还引入了以下两点：
# 
# 特征随机性：在每个决策树节点的分裂过程中，随机选择一部分特征，而不是使用所有特征。这样可以降低模型的方差，提高泛化能力。
# 多棵树：通过结合多棵树的预测结果，随机森林能够减少单个树可能出现的过拟合现象。
# 分类问题：
# 
# 每棵树独立地对输入数据进行分类。
# 最终的分类结果是通过对所有树的预测结果进行投票，选择票数最多的类别作为最终输出。
# 回归问题：
# 
# 每棵树独立地对输入数据进行回归预测。
# 最终的预测结果是通过对所有树的预测值取平均值来得到。

> 随机模型: 使用支持向量机
svm_model <- svm(IsActiveMember ~ ., data = trainData, probability = TRUE)
# 这个参数指定是否要计算并返回预测的概率。如果设置为TRUE，模型在进行预测时会返回每个类的概率。
````

# 用混淆矩阵评估三种模型效果
````R
> gboost混淆矩阵
xgb_pred <- predict(xgb_model, dtest)
xgb_pred_label <- ifelse(xgb_pred > 0.5, 1, 0)
xgb_confusion <- confusionMatrix(factor(xgb_pred_label), factor(as.numeric(testData$IsActiveMember)-1))
# 预测：xgb_pred <- predict(xgb_model, dtest)
# 使用训练好的XGBoost模型 xgb_model 对测试数据 dtest 进行预测，生成预测结果 xgb_pred。预测结果通常是概率值，即某样本属于某一类别的概率。
# 
# 标签转换：xgb_pred_label <- ifelse(xgb_pred > 0.5, 1, 0)
# 将概率值转换为二分类标签。假设阈值为0.5，如果预测概率大于0.5，则预测标签为1（表示积极会员）；否则，预测标签为0（表示非积极会员）。
# 
# 生成混淆矩阵：xgb_confusion <- confusionMatrix(factor(xgb_pred_label), factor(as.numeric(testData$IsActiveMember)-1))
# 使用 confusionMatrix 函数生成混淆矩阵。这里，factor(xgb_pred_label) 是模型的预测标签，factor(as.numeric(testData$IsActiveMember)-1) 是测试数据的实际标签。假设 testData$IsActiveMember 是一个二分类变量（例如0和1），在转换时需要减去1（如果需要），以确保标签与预测标签在同一范围内。
# xgboost和dtest
# xgboost（Extreme Gradient Boosting）是一种高效的梯度提升算法。xgboost要求数据以DMatrix格式输入，这是为了优化内存效率和计算速度。

> 随机森林确保预测值和实际值具有相同的因子级别
rf_pred <- predict(rf_model, testData)
rf_confusion <- confusionMatrix(factor(rf_pred), factor(testData$IsActiveMember))

> SVM混淆矩阵
svm_pred <- predict(svm_model, testData, probability = TRUE)
# 参数 probability = TRUE 指示模型返回预测类别的概率。
svm_pred_label <- factor(ifelse(attr(svm_pred, "probabilities")[, 2] > 0.5, 1, 0), levels = levels(testData$IsActiveMember))
# 使用 attr(svm_pred, "probabilities")[, 2] 获取 svm_pred 中类别为 1 的概率。
# 使用 ifelse 函数判断这些概率是否大于 0.5，如果是，则将预测标签设为 1，否则设为 0。
# 使用 factor 将预测标签转换为因子，并且 levels 参数设置为 testData$IsActiveMember 的因子水平，以确保预测标签与真实标签的因子水平一致。
svm_confusion <- confusionMatrix(svm_pred_label, factor(testData$IsActiveMember))
````

# 评估三种模型的AUC, Accuracy，precision，recall，f1结果
````R
> 定义一个函数来计算这些指标
evaluation_metrics <- function(data, label, pred){
  auc <- roc(as.numeric(data$IsActiveMember)-1, as.numeric(pred), direction="<")$auc
  cm <- confusionMatrix(label, factor(as.numeric(data$IsActiveMember)-1))
  accuracy <- cm$overall['Accuracy']
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(list(auc = auc, accuracy = accuracy, precision = precision, recall = recall, f1 = f1))
}

> 计算xgboost模型的指标
xgb_metrics <- evaluation_metrics(testData, factor(xgb_pred_label), xgb_pred)
> 计算随机森林模型的指标
rf_metrics <- evaluation_metrics(testData, factor(rf_pred, levels = levels(testData$IsActiveMember)), predict(rf_model, testData, type = "prob")[,2])
> 计算SVM模型的指标
svm_metrics <- evaluation_metrics(testData, svm_pred_label, attr(svm_pred, "probabilities")[,2])

> 提取各项评估指标
xgb_result <- unlist(xgb_metrics)
rf_result <- unlist(rf_metrics)
svm_result <- unlist(svm_metrics)

> 形成矩阵
results_matrix <- matrix(c(xgb_result, rf_result, svm_result), nrow = 5, byrow = TRUE)
# byrow = TRUE：指定矩阵应按行填充数据。这意味着数据将先填满第一行，然后是第二行，依此类推。

> 添加行列名
colnames(results_matrix) <- c("XGBoost", "RandomForest", "SVM")
rownames(results_matrix) <- c("AUC", "Accuracy", "Precision", "Recall", "F1")

# 显示结果矩阵
# results_matrix
#   | Metric     | XGBoost   | RandomForest | SVM       |
#   |------------|-----------|--------------|-----------|
#   | AUC        | 0.8355359 | 0.7500000    | 0.7430025 |
#   | Accuracy   | 0.7714663 | 0.7569669    | 0.9050264 |
#   | Precision  | 0.2126667 | 0.2071823    | 0.1981506 |
#   | Recall     | 0.2025658 | 0.6915802    | 0.6280000 |
#   | F1         | 0.6307490 | 0.6340819    | 0.6324111 |
#   AUC越大，分类效果越好
#   accuracy越大，说明分类结果越精确 （TP+TN)/四项和
#   precision越大，说明分类越精确 TP/(TP+FP)
#   RECALL越大，说明分类越准确 TP/(TP+FN)
#   F1越大，说明分类越准确 2 * (precision * recall) / (precision + recall)
````  

# 作图输出特征重要性
````R
xgb_importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_importance)

> 随机森林特征重要性
rf_importance <- varImp(rf_model)
ggplot(rf_importance, aes(x = Overall, y = reorder(Variables, Overall))) + geom_bar(stat = "identity")

> 使用线性核函数重新训练SVM模型
svm_linear_model <- svm(IsActiveMember ~ ., data = trainData, kernel = "linear", probability = TRUE)

> 提取SVM模型的系数
svm_coefficients <- t(svm_linear_model$coefs) %*% svm_linear_model$SV
# t()函数对系数进行转置。
# %*% 矩阵乘法，得到每个特征的系数。

> 提取特征名称
feature_names <- names(trainData)[-which(names(trainData) == "IsActiveMember")]

> 计算特征重要性
feature_importance <- colSums(abs(svm_coefficients))

# 创建特征重要性数据框
svm_importance <- data.frame(Feature = feature_names, Importance = feature_importance)

# 按重要性排序
svm_importance <- svm_importance[order(-svm_importance$Importance),]

# 打印特征重要性
print(svm_importance)
  # | Feature            | Importance      |
  # |--------------------|-----------------|
  # | Exited             | 8.228534e-01    |
  # | Complain           | 2.222319e-04    |
  # | Age                | 8.532593e-05    |
  # | EstimatedSalary    | 6.281416e-05    |
  # | NumOfProducts      | 6.012939e-05    |
  # | CreditScore        | 5.068875e-05    |
  # | CustomerId         | 5.046030e-05    |
  # | Point.Earned       | 4.826969e-05    |
  # | Gender             | 3.300029e-05    |
  # | Geography          | 2.913794e-05    |
  # | Satisfaction.Score | 2.455512e-05    |
  # | RowNumber          | 2.191396e-05    |
  # | Balance            | 1.371164e-05    |
  # | Card.Type          | 1.329100e-05    |
  # | HasCrCard          | 1.259428e-05    |
  # | Tenure             | 1.250197e-05    |
````

# 输出树状规则图
````R
library(DiagrammeR)
# 输出xgboost的树状图
xgb.plot.tree(model = xgb_model, trees = 9)
#trees 参数是从零开始索引的，因此 trees = 0 绘制第一棵树，trees = 6 则绘制第七棵树。

# 随机森林树状图
tree_structure <- getTree(rf_model, k = 1, labelVar = TRUE)
print(tree_structure )
````

# 列表推导式，遍历验证集，输出每条记录
````R
apply(validationData, 1, function(x) print(x))

> 对比验证集的真实结果和预测结果
validation_dtest <- xgb.DMatrix(data = as.matrix(validationData[, -which(names(validationData) == "IsActiveMember")]), label = as.numeric(validationData$IsActiveMember)-1)

> 预测验证集结果
xgb_validation_pred <- predict(xgb_model, validation_dtest)
xgb_validation_pred_label <- ifelse(xgb_validation_pred > 0.5, 1, 0)

rf_validation_pred <- predict(rf_model, validationData)
svm_validation_pred <- predict(svm_model, validationData, probability = TRUE)
svm_validation_pred_label <- ifelse(attr(svm_validation_pred, "probabilities")[, 2] > 0.5, 1, 0)

> 打印验证集真实结果和预测结果
comparison <- data.frame(
  Real = as.numeric(validationData$IsActiveMember)-1,
  XGBoost_Pred = xgb_validation_pred_label,
  RandomForest_Pred = as.numeric(rf_validation_pred)-1,
  SVM_Pred = svm_validation_pred_label
)

print(comparison)

> 创建对比数据框
comparison <- data.frame(
  Real = as.numeric(validationData$IsActiveMember)-1,
  XGBoost_Pred = xgb_validation_pred_label,
  RandomForest_Pred = as.numeric(rf_validation_pred)-1,
  SVM_Pred = svm_validation_pred_label
)

library(ggplot2)
library(reshape2)

> 转换 'value' 为因子类型
comparison_long$value <- factor(comparison_long$value, levels = c(0, 1))

> 确保 'Real' 也是因子类型
comparison_long$Real <- factor(comparison_long$Real, levels = c(0, 1))

> 创建新的数据框以便绘图
comparison_long$Type <- ifelse(comparison_long$variable == "Real", "Real", "Predicted")

> 绘制真实结果和预测结果对比图
ggplot(comparison_long, aes(x = Real, fill = interaction(Type, value))) + 
  geom_bar(position = "dodge") + 
  facet_wrap(~ variable, scales = "free_y") + 
  labs(title = "Comparison of Real vs Predicted Results",
       x = "Real Value",
       y = "Count") + 
  scale_fill_manual(values = c("Real.0" = "red", "Real.1" = "blue", "Predicted.0" = "pink", "Predicted.1" = "lightblue")) + 
  theme_minimal() + 
  scale_x_discrete(drop=FALSE) + 
  scale_y_continuous(expand = c(0, 0)) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
````
