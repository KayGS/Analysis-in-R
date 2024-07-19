````R
加载必要的库
library(readr)
library(caret)
library(gbm)
library(ggplot2)
library(lattice)
library(dplyr)
library(plotly)

加载数据
data <- read_csv(file.choose())

spotify_data <- read_csv(file.choose(), locale = locale(encoding = "ISO-8859-1"))

Check for NA in 'Spotify Popularity' column
na_popularity <- is.na(spotify_data$`Spotify Popularity`)

Extract rows with NA in 'Spotify Popularity' as the initial test set
test_set_na <- spotify_data[na_popularity, ]

Create the remaining dataset without the NA rows
data_no_na <- spotify_data[!na_popularity, ]

index <- createDataPartition(data_no_na$`Spotify Popularity`, p = 0.8, list = FALSE)
train_data <- data_no_na[index, ]
test_data <- data_no_na[-index, ]


确保所有预测变量都是数值型并处理缺失值
x_train <- train_data %>%
  select(-`Spotify Popularity`) %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate_all(~ ifelse(is.na(.), 0, .))

y_train <- train_data$`Spotify Popularity`

x_test <- test_data %>%
  select(-`Spotify Popularity`) %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate_all(~ ifelse(is.na(.), 0, .))

y_test <- test_data$`Spotify Popularity`

检查数据集的结构和类型
str(x_train)
str(y_train)
str(x_test)
str(y_test)


设置网格搜索的超参数调优
tune_grid <- expand.grid(interaction.depth = c(1, 3, 5),
                         n.trees = c(50, 100, 150),
                         shrinkage = c(0.01, 0.1),
                         n.minobsinnode = 10)
# 为什么这样写
# 使用expand.grid函数定义的网格是为了在训练过程中探索这些参数的不同组合。网格搜索通过交叉验证来评估每个参数组合的性能，从而找到最优的参数组合。

# interaction.depth：
# 
# 意义：每棵树的最大深度。决定了树的复杂度。
# 作用：树的深度越大，模型的复杂度越高，能够捕捉更多的数据特征。但同时，过深的树可能导致过拟合。
# 为什么选择1, 3, 5：这些值提供了一个从简单模型（较浅的树）到复杂模型（较深的树）的范围，便于网格搜索找到最佳的深度。
# 为什么不只选1或只选5
# 只选1：模型可能过于简单，无法捕捉数据中的复杂关系，导致欠拟合，模型在测试数据上的表现可能不佳。
# 只选5：模型可能过于复杂，尽管能很好地拟合训练数据，但容易过拟合，导致在测试数据上的泛化能力差。

# n.trees：
# 
# 意义：要生成的树的数量。更多的树通常能提高模型的表现。
# 作用：树的数量越多，模型越复杂，训练时间也越长。合适的树数量能够有效提高模型的性能。
# 为什么选择50, 100, 150：这些值提供了从较少的树（简单模型）到较多的树（复杂模型）的选择，帮助找到最优的树数量。

# shrinkage（也称为学习率）：
# 
# 意义：每棵树对模型最终预测结果的贡献比例。控制了每次迭代更新的步长。
# 作用：较小的学习率使得每棵树的贡献较小，通常需要更多的树才能达到同样的效果，但能提高模型的泛化能力。
# 为什么选择0.01, 0.1：这些值提供了常用的学习率范围。0.01是较低的学习率，适合更稳定的模型更新；0.1是中等学习率，适合更快的模型收敛。

# n.minobsinnode：
# 
# 意义：每个叶子节点的最小观察数。
# 作用：这个参数控制了每个叶子节点中必须至少包含多少个样本。较大的值可以防止模型过拟合。
# 为什么选择10：这是一个常用的值，确保每个叶子节点不会过于细分，同时保持足够的样本数量进行预测。

train_control <- trainControl(method = "cv", number = 5)

# method:
#   
#   意义：指定用于模型评估的重采样方法。
# "cv"：代表交叉验证（Cross-Validation）。交叉验证是一种常用的模型评估方法，通过将数据集划分为多个子集，并多次训练和测试模型，以减少模型过拟合和评估的偏差。
# number:
#   
#   意义：指定交叉验证的折数，即将数据集分成多少个子集。
# 5：表示5折交叉验证（5-fold Cross-Validation）。在5折交叉验证中，数据集被分成5个等分，每次使用4个子集进行训练，1个子集进行验证。这一过程重复5次，每个子集都被用作一次验证集。
# 为什么要这样写
# 交叉验证的必要性：
# 
# 减少过拟合：交叉验证通过多次划分和验证，减少了单一训练集导致的过拟合风险。
# 提高模型评估的稳定性：多次验证提供了模型性能的更稳定估计。
# 5折交叉验证的选择：
# 
# 平衡计算成本和评估质量：5折交叉验证在计算成本和评估质量之间取得了平衡。折数越多，评估越稳定，但计算成本越高；折数越少，计算成本低，但评估可能不稳定。5折是一个常见且合理的选择。

使用GridSearchCV训练Gradient Boosting模型
gbm_model <- train(x_train, y_train,
                   method = "gbm",
                   trControl = train_control,
                   tuneGrid = tune_grid,
                   metric = "RMSE",
                   verbose = FALSE)
# caret包的train函数已经集成了网格搜索的功能，相当于Python中的GridSearchCV。
#
# method = "gbm"：
# 
# 意义：指定要训练的模型类型，这里选择的是Gradient Boosting Machine (GBM)。
# 原因：GBM是一种强大的集成学习算法，通过逐步优化损失函数，提升模型的性能。
# 
# trControl = train_control：
# 
# 意义：指定训练控制参数，这里传入的是由trainControl函数生成的对象。
# 原因：控制模型训练过程中的交叉验证方法、重复次数、抽样方式等。确保模型的泛化能力。
# 
# tuneGrid = tune_grid：
# 
# 意义：指定网格搜索的参数空间，这里传入的是由expand.grid函数生成的超参数组合。
# 原因：通过网格搜索，自动找到最优的超参数组合，提升模型的性能。
# 
# metric = "RMSE"：
# 
# 选择适当的评估指标可以有效衡量模型的性能。RMSE反映了预测值和实际值之间的差异，选择RMSE可以帮助优化模型的精度。
# 
# verbose = FALSE：
# 
# 控制信息输出的详细程度。设置为FALSE可以保持控制台的简洁，避免过多的信息干扰。如果需要调试，可以将其设置为TRUE以查看详细的训练过程信息。

获取最佳参数组合和最佳评分
best_params <- gbm_model$bestTune
best_score <- min(gbm_model$results$RMSE)

打印最佳参数和最佳评分
print(best_params)
print(paste("Best RMSE:", best_score))

# 示例输出：
# r
# Copy code
# n.trees interaction.depth shrinkage n.minobsinnode
# 17     100                 5       0.1             10
# 这表示最佳参数组合为：树的数量（n.trees）为100，树的最大深度（interaction.depth）为5，学习率（shrinkage）为0.1，每个叶节点的最小观测数（n.minobsinnode）为10。

查看交叉验证得分
cv_results <- gbm_model$resample
print(cv_results)

# RMSE (Root Mean Square Error): 均方根误差，反映模型预测值与实际值之间的差异，值越小越好。
# Rsquared: 决定系数，反映模型对数据的拟合程度，值越接近1越好。
# MAE (Mean Absolute Error): 平均绝对误差，反映预测值与实际值的平均差异，值越小越好。


在测试集上进行预测
predictions <- predict(gbm_model, x_test)

# GBM 通过多棵决策树的集合来改善预测精度，因此它并不是仅针对单一特征进行分析的，而是综合了所有特征的贡献。
# 
# 在上述实例中，我们使用的是 gbm 方法，这是 Gradient Boosting Machine 的一种实现方式。在模型训练过程中，它会综合考虑所有输入特征，并通过多棵树的集合进行预测。

评估模型
results <- data.frame(Index = 1:length(y_test), Actual = y_test, Predicted = predictions)

绘制真实值（红色）和预测值（绿色）的折线图
ggplot(results, aes(x = Index)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "green")) +
  labs(title = "Actual vs Predicted Spotify Popularity",
       x = "Index",
       y = "Spotify Popularity",
       color = "Legend") +
  theme_minimal()

创建交互式散点图
scatter_plot <- plot_ly(results, x = ~Actual, y = ~Predicted, type = 'scatter', mode = 'markers',
                        marker = list(color = 'blue')) %>%
  layout(title = 'Actual vs Predicted Spotify Popularity',
         xaxis = list(title = 'Actual Spotify Popularity'),
         yaxis = list(title = 'Predicted Spotify Popularity'))

创建交互式残差图
results$Residual <- results$Actual - results$Predicted
residual_plot <- plot_ly(results, x = ~Index, y = ~Residual, type = 'scatter', mode = 'markers',
                         marker = list(color = 'red')) %>%
  layout(title = 'Residuals of Predictions',
         xaxis = list(title = 'Index'),
         yaxis = list(title = 'Residuals'))

显示图表
scatter_plot
residual_plot

加载新的数据集
new_file_path <- "path_to/new_data.csv"
new_data <- read_csv(new_file_path, locale = locale(encoding = "ISO-8859-1"))

确保所有预测变量都是数值型并处理缺失值
x_new <- test_set_na %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate_all(~ ifelse(is.na(.), 0, .))

使用训练好的模型进行预测
new_predictions <- predict(gbm_model, x_new)

将预测结果与新数据结合
new_data_with_predictions <- test_set_na %>%
  mutate(Predicted_Spotify_Popularity = new_predictions)

查看带有预测结果的新数据集
print(new_data_with_predictions)

如果需要，可以将带有预测结果的新数据集保存到文件
write_csv(new_data_with_predictions, "path_to/new_data_with_predictions.csv")

显示一些预测结果
head(new_data_with_predictions$Track,new_data_with_predictions$`Spotify Popularity`,new_data_with_predictions$Artist)

可视化新的预测结果（例如：散点图）
ggplot(new_data_with_predictions, aes(x = 1:nrow(new_data_with_predictions), y = Predicted_Spotify_Popularity)) +
  geom_line(color = "blue") +
  labs(title = "Predicted Spotify Popularity for New Data",
       x = "Index",
       y = "Predicted Spotify Popularity") +
  theme_minimal()
````
