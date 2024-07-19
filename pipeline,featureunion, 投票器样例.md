```R
install.packages(c("caret", "mlr3", "mlr3pipelines", "mlr3learners", "mlr3tuning", "mlr3filters"))
library(caret)
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(mlr3filters)
library(kknn)
```

# 构建组合特征选择器
```R
首先，定义特征选择器和预处理步骤的pipeline。

创建任务
task = tsk("iris")
# 这行代码创建了一个分类任务。tsk("iris")是mlr3包中的一个函数，用于创建一个预定义的数据集任务。

# <TaskClassif:iris> (150 x 5): Iris Flowers
# * Target: Species
# * Properties: multiclass
# * Features (4):
#   - dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
# 输出解释
# <TaskClassif:iris>：
# 
# TaskClassif：表示这是一个分类任务。
# iris：表示使用的是iris数据集。
# (150 x 5): Iris Flowers：
# 
# 150 x 5：表示数据集有150行（样本）和5列（特征和目标）。
# Iris Flowers：这是任务的描述。
# * Target: Species：
# 
# Target：表示目标变量（需要预测的变量）。
# Species：目标变量的名称，在iris数据集中，Species表示鸢尾花的类别。
# * Properties: multiclass：
# 
# Properties：表示任务的属性。
# multiclass：表示这是一个多类分类任务，即目标变量有多个类别。
# * Features (4):：
# 
# Features：表示特征的数量和详细信息。
# 4：表示有4个特征。
# - dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width：
# 
# dbl (4)：表示这4个特征都是双精度浮点数（double）。
# Petal.Length, Petal.Width, Sepal.Length, Sepal.Width：这4个特征的名称，分别是花瓣长度、花瓣宽度、花萼长度和花萼宽度。

定义过滤器和选择器
filter_corr = po("filter", filter = flt("correlation"), filter.cutoff = 0.8)
filter_anova = po("filter", filter = flt("anova"), filter.cutoff = 0.8)
# po("filter", filter = flt("correlation"), filter.cutoff = 0.8)：
# 
# po：表示管道操作符（pipeline operator），用于定义管道中的步骤。
# "filter"：指定这是一个过滤器步骤。
# filter = flt("correlation")：指定使用相关性过滤器（correlation filter）。
# filter.cutoff = 0.8：设定过滤器的阈值为0.8，即选择相关性大于等于0.8的特征。
# po("filter", filter = flt("anova"), filter.cutoff = 0.8)：
# 
# po：同上，表示管道操作符。
# "filter"：同上，指定这是一个过滤器步骤。
# filter = flt("anova")：指定使用ANOVA过滤器。
# filter.cutoff = 0.8：设定ANOVA过滤器的阈值为0.8，即选择ANOVA统计量大于等于0.8的特征。
# 过滤器的作用
# 相关性过滤器（Correlation Filter）：
# 
# 这个过滤器通过计算特征与目标变量之间的相关性来选择特征。相关性高的特征（即相关性大于或等于0.8）会被保留，其余的特征会被过滤掉。
# 相关性度量通常是皮尔逊相关系数，适用于线性关系。
# ANOVA过滤器：
# 
# 这个过滤器通过计算每个特征与目标变量之间的方差分析（ANOVA）统计量来选择特征。统计量高的特征（即ANOVA统计量大于或等于0.8）会被保留，其余的特征会被过滤掉。
# ANOVA适用于分类任务，尤其是多类分类，通过比较组间和组内方差来评估特征的重要性。
# 通过这种方式，可以有效地选择对目标变量有显著影响的特征，从而简化模型、减少过拟合，并提高模型的预测性能。

组合过滤器
combined_filter = gunion(list(filter_corr, filter_anova)) %>>% po("featureunion")
# gunion(list(filter_corr, filter_anova))：gunion函数将多个PipeOp（管道操作）组合成一个列表，使它们并行执行。这一步的目的是将filter_corr和filter_anova组合成一个并行的管道操作列表。
# %>>% po("featureunion")：%>>%操作符用于将一个PipeOp的输出连接到另一个PipeOp的输入。po("featureunion")是一个特征联合（feature union）操作，它将前面的过滤器的输出联合起来，生成一个新的特征集合。具体来说，它将filter_corr和filter_anova的输出特征集合合并成一个特征集合。

定义分类器
learner_logreg = lrn("classif.log_reg")
learner_rpart = lrn("classif.rpart")
learner_knn = lrn("classif.kknn")

在R中，使用`mlr3`包定义分类器的代码如下：

learner_logreg = lrn("classif.log_reg")
learner_rpart = lrn("classif.rpart")
learner_knn = lrn("classif.kknn")
# 
# 这段代码的作用是创建三个不同类型的分类器。下面是对每个分类器的详细解释：
# 
# 1. **Logistic Regression 分类器**
#   ```R
# learner_logreg = lrn("classif.log_reg")
# ```
# - **learner_logreg**：这是一个逻辑回归分类器对象。
# - **lrn("classif.log_reg")**：`mlr3`包中的`lrn`函数用于创建学习器对象。`"classif.log_reg"`表示我们想要创建一个逻辑回归分类器。
# 
# 2. **Decision Tree 分类器**
#   ```R
# learner_rpart = lrn("classif.rpart")
# ```
# - **learner_rpart**：这是一个决策树分类器对象。
# - **lrn("classif.rpart")**：`"classif.rpart"`表示我们想要创建一个决策树分类器。`rpart`是`R`中的一个决策树算法。
# 
# 3. **K-Nearest Neighbors 分类器**
#   ```R
# learner_knn = lrn("classif.kknn")
# ```
# - **learner_knn**：这是一个K近邻分类器对象。
# - **lrn("classif.kknn")**：`"classif.kknn"`表示我们想要创建一个K近邻分类器。`kknn`是`R`中的一个K近邻算法。
# 
# 这些分类器将用于组合模型的pipeline中，构建一个投票分类器。投票分类器通过结合多个分类器的预测结果来提高整体预测性能。

# 在这段代码中，我们构建了一个组合模型的pipeline，包含交叉验证和投票分类器。下面是对每一部分的详细解释：

创建组合模型的pipeline

combined_learner = po("learner_cv", learner = list(learner_logreg, learner_rpart, learner_knn), cv_outer = rsmp("cv", folds = 3)) %>>% 
  po("classif_voting", learners = list(learner_logreg, learner_rpart, learner_knn))

#### 分步解释
# 
# 1. **`po("learner_cv", ...)`**:
#   - 这个部分创建了一个包含交叉验证的学习器（learner）。`po`表示"pipeline operator"，用于定义pipeline的操作。
# - `learner_cv`：这是一个内置操作，用于在pipeline中执行交叉验证。
# - `learner = list(learner_logreg, learner_rpart, learner_knn)`：这里我们定义了三个学习器（逻辑回归、决策树和KNN），并将它们传递给`learner_cv`。
# - `cv_outer = rsmp("cv", folds = 3)`：定义了外部交叉验证的方式，这里使用了三折交叉验证。
# 
# 2. **`%>>%`**:
#   - 这是一个连接符，用于将两个pipeline操作连接起来。前面的操作输出将作为后面操作的输入。
# 
# 3. **`po("classif_voting", learners = list(learner_logreg, learner_rpart, learner_knn))`**:
#   - 这个部分定义了一个投票分类器。`po`再次表示"pipeline operator"。
# - `classif_voting`：这是一个内置操作，用于创建一个投票分类器，它将多个学习器的预测结果进行投票表决。
# - `learners = list(learner_logreg, learner_rpart, learner_knn)`：将之前定义的三个学习器传递给投票分类器。
# 
# #### 详细说明
# 
# - **交叉验证（learner_cv）**：
# - 在这里，我们将多个学习器（逻辑回归、决策树和KNN）进行三折交叉验证。交叉验证的目的是评估模型的性能，以减少过拟合的风险。
# 
# - **投票分类器（classif_voting）**：
# - 在交叉验证之后，我们将三个学习器的预测结果进行投票表决。每个学习器对每个样本进行预测，最终的预测结果是根据多数投票决定的。这种方法可以提高模型的整体性能和鲁棒性，因为它结合了多个模型的优势。
# 
# ### 总结
# 
# 这段代码创建了一个组合模型的pipeline，其中包括：
# - 使用交叉验证对三个不同的学习器进行评估。
# - 将这些学习器的预测结果进行投票表决，以获得最终的预测结果。
# 
# 这个组合模型的pipeline通过结合多个学习器和交叉验证，旨在提高模型的性能和泛化能力。

创建单个学习器的交叉验证（通过管道操作符）
cv_learner_logreg = po("learner_cv", learner = learner_logreg)
cv_learner_rpart = po("learner_cv", learner = learner_rpart)
cv_learner_knn = po("learner_cv", learner = learner_knn)

创建组合学习器 平均分类 软投票
# 我们使用 po("classifavg") 操作符来平均多个分类器的概率（软投票，按概率）。这个操作符会自动处理投票机制并输出最终的分类结果。
# Create learners with unique IDs
po_rpart = po("learner", learner = lrn("classif.rpart"), id = "rpart_po")
po_knn = po("learner", learner = lrn("classif.kknn"), id = "knn_po")
po_svm = po("learner", learner = lrn("classif.svm"), id = "svm_po")
# 逻辑回归分类器（classif.log_reg）不支持多类分类任务（multiclass），因为它只能处理二分类任务。我们需要选择支持多类分类的学习器来进行投票分类器的构建。

# Combine the learners using gunion
combined_learner = gunion(list(po_svm, po_rpart, po_knn))

# Create a PipeOp for average probabilities (soft voting)
po_avg = po("classifavg")

# Create the full graph
graph = combined_learner %>>% po_avg

# Convert the graph to a GraphLearner
graph_learner = GraphLearner$new(graph)

# Print the combined learner
print(graph_learner)

# Load a task
task = tsk("iris")

# Train the combined learner
graph_learner$train(task)

# Make predictions
prediction = graph_learner$predict(task)
print(prediction)

hardvoting报错 需要自写函数
# classif.voting 使用投票机制来决定最终的分类结果（label按得票多），适合于希望通过投票机制提高分类准确性的场景。
library(data.table)

# Create learners with unique IDs
po_rpart = po("learner", learner = lrn("classif.rpart"), id = "rpart_po")
po_knn = po("learner", learner = lrn("classif.kknn"), id = "knn_po")
po_svm = po("learner", learner = lrn("classif.svm"), id = "svm_po")

# Combine the learners using gunion
combined_learner = gunion(list(po_svm, po_rpart, po_knn))


# Define the custom PipeOp for hard voting
PipeOpHardVoting = R6::R6Class("PipeOpHardVoting",
                               inherit = mlr3pipelines::PipeOp,
                               public = list(
                                 initialize = function(id = "hardvoting") {
                                   super$initialize(id, param_set = ParamSet$new())
                                   self$param_set$values$innum = 3
                                   self$param_set$values$outnum = 1
                                 },
                                 predict = function(inputs) {
                                   # Collect the responses from each learner
                                   preds = lapply(inputs, function(input) input$response)
                                   # Ensure all predictions are valid
                                   preds = as.data.table(do.call(cbind, preds))
                                   # Majority vote function
                                   majority_vote = function(x) {
                                     tbl = table(x)
                                     names(tbl)[which.max(tbl)]
                                   }
                                   prediction = apply(preds, 1, majority_vote)
                                   # Return a PredictionClassif object
                                   PredictionClassif$new(row_ids = inputs[[1]]$row_ids, truth = inputs[[1]]$truth, response = prediction)
                                 }
                               )
)

# Instantiate the custom PipeOp
po_hardvoting = PipeOpHardVoting$new()

# Create the full graph
graph = combined_learner %>>% po_hardvoting

# Convert the graph to a GraphLearner
graph_learner = GraphLearner$new(graph)

# Print the combined learner
print(graph_learner)

# Load a task
task = tsk("iris")

# Train the combined learner
graph_learner$train(task)

# Make predictions
prediction = graph_learner$predict(task)
print(prediction)


# 创建单个学习器的管道操作符
po_logreg = po("learner", learner = learner_logreg)
po_rpart = po("learner", learner = learner_rpart)
po_knn = po("learner", learner = learner_knn)


# 创建组合学习器 投票器
combined_learner = gunion(list(po_logreg, po_rpart, po_knn)) %>>% po("hardvoting")

classif.voting
classif.voting使用投票机制来结合多个基础分类器的预测结果。投票机制有两种主要类型：硬投票和软投票。
 
硬投票：每个基础分类器对每个样本给出一个分类标签，最终的分类结果是得到最多票数的类别。
软投票：每个基础分类器对每个样本给出一个类别概率分布，最终的分类结果是将这些概率分布相加，然后选择概率最高的类别。
在mlr3中，classif.voting通常用于硬投票。

组合特征选择器和分类器的pipeline
final_pipeline = combined_filter %>>% combined_learner

# 具体来说：
# 
# 第一部分（combined_filter）：
# 
# 这部分是特征选择器。它应用了两个不同的特征选择方法，然后将这些方法的结果组合在一起。这一步的输出是经过特征选择后的数据。
# 第二部分（combined_learner）：
# 
# 这部分是分类器。它接收经过特征选择的数据，并应用多个分类器进行学习和预测。通过交叉验证和投票分类，最终输出预测结果。
# 通过将combined_filter和combined_learner组合成一个流水线（final_pipeline），我们实现了一个包含特征选择和分类器的完整模型。这个流水线可以在整个数据集上运行，从特征选择到最终的分类预测，所有步骤自动串联起来。
# 
# 流水线执行流程
# 数据输入：数据首先输入到流水线的起点（combined_filter）。
# 特征选择：数据经过特征选择步骤，得到经过筛选的特征。
# 分类器训练和预测：筛选后的特征数据传递给分类器，分类器进行训练和预测。
# 最终输出：最终的分类结果输出。

### 训练和评估模型

# 设置 resampling
resampling = rsmp("cv", folds = 5)

# 评估模型
rr = resample(task, graph_learner, resampling)

当然，下面是对这两行代码的详细解释：

### 设置 resampling

resampling = rsmp("cv", folds = 5)


# 这一行代码的作用是设置交叉验证 (cross-validation) 的重抽样策略。具体解释如下：
# 
# - `rsmp`：这是`mlr3`包中的一个函数，用于创建重抽样描述符 (resampling descriptor)。
# - `"cv"`：表示使用交叉验证 (cross-validation) 作为重抽样方法。
# - `folds = 5`：表示将数据集分成5折 (folds) 进行交叉验证。每一折会被用作一次验证集，其余的折作为训练集。
# 
# ### 评估模型
# ```r
# rr = resample(task, final_pipeline, resampling)
# ```
# 
# 这一行代码的作用是使用定义好的重抽样策略来评估模型。具体解释如下：
# 
# - `resample`：这是`mlr3`包中的一个函数，用于对模型进行重抽样评估。
# - `task`：表示我们之前创建的机器学习任务（例如，`iris`数据集的分类任务）。
# - `final_pipeline`：表示我们创建的最终pipeline，包含了特征选择器和分类器。
# - `resampling`：表示我们定义的重抽样策略（5折交叉验证）。
# 
# 这个函数的主要目的是在不同的训练集和验证集上对模型进行训练和评估，以获得模型在不同数据分割下的性能表现。
# 
# ### 总结
# - 交叉验证是一种评估模型性能的常用方法，可以帮助我们更好地了解模型在未见过的数据上的表现。
# - `rsmp("cv", folds = 5)` 设置了使用5折交叉验证。
# - `resample(task, final_pipeline, resampling)` 使用定义好的交叉验证策略来评估模型的性能。
# 
# 评估结果保存在变量 `rr` 中，可以进一步分析和汇总这些结果，例如计算平均准确率等。

# 打印结果
rr$aggregate(msr("classif.acc"))

### 解释代码

# 1. **任务创建**：我们使用`tsk("iris")`来创建一个任务。这里使用了`iris`数据集。
# 2. **特征选择**：我们定义了两个特征选择器，分别使用相关性过滤器和ANOVA过滤器。然后，我们使用`gunion`和`po("featureunion")`将它们组合起来。
# 3. **分类器**：我们定义了三个分类器：逻辑回归、决策树和KNN。
# 4. **组合模型**：我们创建了一个组合模型的pipeline，其中包含了交叉验证和投票分类器。
# 5. **组合特征选择器和分类器的pipeline**：我们将特征选择器和分类器组合成一个最终的pipeline。
# 6. **训练和评估模型**：我们使用交叉验证来评估模型，并打印分类准确率。
# 
# 通过这种方式，您可以在R中使用pipeline和feature union构建一个voting classifier和组合特征选择器
````
