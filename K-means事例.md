# 数据读取和标准化
````r
# 检查是否有缺失值
colSums(is.na(data[, c("Order_ID", "Store_Latitude", "Store_Longitude", "Drop_Latitude", 
                       "Drop_Longitude", "Order_Date", "Order_Time", "Weather"
                       , "Traffic", "Vehicle", "Area"
                       , "Delivery_Time", "Category", "Agent_Age", "Agent_Rating"
                       , "Delivery_Time")]))

# 处理多列缺失值，用列的均值替换缺失值
data <- data.frame(lapply(data[, c("Order_ID", "Store_Latitude", "Store_Longitude", "Drop_Latitude", 
                                   "Drop_Longitude", "Order_Date", "Order_Time", "Weather"
                                   , "Traffic", "Vehicle", "Area"
                                   , "Delivery_Time", "Category", "Agent_Age", "Agent_Rating"
                                   , "Delivery_Time")], function(x) {
  ifelse(is.na(x), mean(x, na.rm = TRUE), x)
}))
````

# 标准化数值列（假设需要标准化的数值列为column1, column2, column3）
````r
data_num <- data[, c("Agent_Age", "Agent_Rating", "Delivery_Time")]
data_num_scaled <- scale(data_num)
````

# 给标准化后的列重新命名
````r
colnames(data_num_scaled) <- c("Agent_Age_scaled", "Agent_Rating_scaled", "Delivery_Time_scaled")
````

# 将标准化后的数据与原始数据合并
````r
data <- cbind(data, data_num_scaled)
summary(data)
str(data)
````

# K-means聚类和模型效果评估
````r
library(cluster)
library(factoextra)
library(ggplot2)

data_geo <- data[,c("Store_Latitude", "Store_Longitude", "Drop_Latitude", 
                    "Drop_Longitude")]
````

# K-means聚类
````r
set.seed(123)
kmeans_result <- kmeans(na.omit(data[,c("Store_Latitude", "Store_Longitude", "Drop_Latitude", 
                                     "Drop_Longitude")]), centers = 3, nstart = 25)
````
> 它指定K-means算法应该运行多少次，每次用不同的随机初始聚类中心。
> 然后选择总平方和（Total Within-Cluster Sum of Squares, WCSS）最小的那次结果。
> 通过设置nstart = 25，算法将运行25次，选择最佳的一次结果。这有助于避免由于随机初始化而导致的局部最优解。
> 

# 评估模型效果, 计算轮廓系数
````r
sil <- silhouette(kmeans_result$cluster, dist(na.omit(data[,c("Store_Latitude", "Store_Longitude", "Drop_Latitude", 
                                                              "Drop_Longitude")])))
silhouette_score <- mean(sil[, 3])
````
> silhouette：这个函数用于计算每个数据点的轮廓系数（Silhouette coefficient）。轮廓系数是用来评估聚类效果的一种指标。
> kmeans_result$cluster：这是K-means算法的聚类结果。它包含了每个数据点所属的聚类标签（即每个数据点被分配到哪个簇）。
> dist(data)：这是计算数据点之间的距离矩阵。dist函数计算数据集中所有点之间的欧氏距离（或其他指定距离度量）。
> 这一行代码的目的是计算每个数据点的轮廓系数，它衡量了数据点与其所属簇内其他点的相似度（紧密度）与它和最近簇内点的相似度（分离度）。
> 
> silhouette_score <- mean(sil[, 3])：
> sil：这是之前计算的轮廓系数矩阵。sil是一个矩阵，其中每一行代表一个数据点。
> sil[, 3]：选择矩阵的第三列。对于每个数据点，第三列包含了其轮廓系数值。
> mean(sil[, 3])：计算所有数据点轮廓系数的平均值。
> 这一行代码的目的是计算所有数据点轮廓系数的平均值，即整体的轮廓系数。这是评估聚类效果的一个总体指标。轮廓系数的值介于-1到1之间，值越高表示聚类效果越好。
> >0,5时，聚类较优

# 总结
> silhouette(kmeans_result$cluster, dist(data))：计算每个数据点的轮廓系数，用于评估每个点在其簇中的位置是否合理。
> mean(sil[, 3])：计算所有数据点轮廓系数的平均值，提供一个聚类效果的总体评估指标。
> 轮廓系数用于衡量数据点在同一簇内的紧密度与不同簇之间的分离度。通过这些步骤，我们可以了解聚类的效果并决定是否需要调整聚类参数或使用不同的聚类方法。

# 计算Calinski-Harabasz指数
````r
calinski_harabasz_score <- fviz_nbclust(data, kmeans, method = "silhouette")$data$y
````

# 交叉验证方法（通过重复多次聚类来评估稳定性）
````r
set.seed(123)
kmeans_repeat <- replicate(10, kmeans(data_geo, centers = 3, nstart = 25)$tot.withinss)
cv_score <- mean(kmeans_repeat)
````

> 这段代码使用交叉验证方法通过重复多次K-means聚类来评估模型的稳定性。每个步骤的具体意义如下：
> 
> **set.seed(123)**:
> 设置随机种子为123。这确保了随机过程（如初始聚类中心的选择）在每次运行时都是相同的，从而保证结果的可重复性。
> 
> **kmeans_repeat <- replicate(10, kmeans(data_geo, centers = 3, nstart = 25)$tot.withinss)**:
> `replicate(10, ...)`：这一部分表示将括号内的表达式重复执行10次，并将结果存储在一个向量中。
> `kmeans(data_geo, centers = 3, nstart = 25)`：执行K-means聚类算法，使用`data_geo`数据集，设置聚类中心的数量为3，`nstart = 25`表示每次运行K-means时使用25次不同的随机初始点来选择最好的结果。
> `$tot.withinss`：这一部分提取K-means结果中的总组内平方和（Total Within Sum of Squares），它是衡量聚类效果的一个指标，值越小表示聚类效果越好。
>  `kmeans_repeat`：这是一个向量，包含10次K-means聚类的总组内平方和。
> 
>  cv_score <- mean(kmeans_repeat)**:
>  计算`kmeans_repeat`向量中所有值的平均值，即10次K-means聚类的总组内平方和的平均值。这个平均值（cv_score）作为模型稳定性的一个评估指标，用于判断K-means算法在不同初始条件下的稳定性。
>  CV-score 越小，说明K-means聚类的效果越好。

````r
cat("轮廓系数:", silhouette_score, "\n")
#cat("Calinski-Harabasz指数:", calinski_harabasz_score, "\n")
cat("交叉验证得分:", cv_score, "\n")
cat("\n",silhouette_score,"\n",cv_score)
````

# 合并数据和特征 合并聚类结果
````r
data$cluster <- kmeans_result$cluster
````

# 计算每个聚类的样本量和占比
````r
cluster_count <- table(data$cluster)
cluster_percentage <- prop.table(cluster_count) * 100

cluster_summary <- data.frame(Cluster = names(cluster_count),
                              Count = as.vector(cluster_count),
                              Percentage = as.vector(cluster_percentage))

print(cluster_summary)

str(data)
````

# 计算不同聚类数值型特征 计算数值型特征的均值和标准差
````r
numeric_features <- data[, c("Agent_Age", "Agent_Rating", "Delivery_Time", "cluster")]
numeric_summary <- aggregate(. ~ cluster, data = numeric_features, 
                             FUN = function(x) c(mean = mean(x), sd = sd(x)))
> ~（波浪号）：表示公式接口。左边的.表示对所有列进行操作，右边的cluster表示按cluster列进行分组。
print(numeric_summary)
````
````r
library(ggplot2)
library(reshape2)
````

# 计算数值型特征的均值和标准差
````r
numeric_features <- data[, c("Agent_Age", "Agent_Rating", "Delivery_Time", "cluster")]
numeric_summary <- aggregate(. ~ cluster, data = numeric_features, 
                             FUN = function(x) c(mean = mean(x), sd = sd(x)))
````

# 将数据重塑以适合绘图
````r
numeric_summary_melt <- melt(numeric_summary, id.vars = "cluster")
numeric_summary_melt <- data.frame(cluster = numeric_summary_melt$cluster,
                                   feature = sub("\\.mean", "", numeric_summary_melt$variable),
                                   value = numeric_summary_melt$value)
````

# 绘制数值型特征的均值图
````r
p1 <- ggplot(numeric_summary_melt, aes(x = cluster, y=value, fill = feature)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Numerical Feature Means by Cluster", x = "Cluster", y = "Mean Value") +
  theme_minimal()
````

# 计算分类型特征的频率分布（假设分类型特征为category1, category2）
````r
categorical_features <- data[, c("Weather", "Traffic", "Vehicle", "Area", "Category","cluster")]
categorical_summary <- aggregate(. ~ cluster, data = categorical_features, 
                                 FUN = function(x) prop.table(table(x)))
````

# 计算每个cluster中不同weather的占比
````r
weather_summary <- data %>%
  group_by(cluster, Weather) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

# 计算每个cluster中不同Traffic的占比
traffic_summary <- data %>%
  group_by(cluster, Traffic) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

# 计算每个cluster中不同Vehicle的占比
vehicle_summary <- data %>%
  group_by(cluster, Vehicle) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

# 计算每个cluster中不同Area的占比
area_summary <- data %>%
  group_by(cluster, Area) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

# 计算每个cluster中不同Category的占比
category_summary <- data %>%
  group_by(cluster, Category) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))
````

# 合并所有特征的数据
````r
weather_summary <- weather_summary %>% rename(value = Weather)
traffic_summary <- traffic_summary %>% rename(value = Traffic)
vehicle_summary <- vehicle_summary %>% rename(value = Vehicle)
area_summary <- area_summary %>% rename(value = Area)
category_summary <- category_summary %>% rename(value = Category)
````

# 添加特征名称
````r
weather_summary <- weather_summary %>% mutate(feature = "Weather")
traffic_summary <- traffic_summary %>% mutate(feature = "Traffic")
vehicle_summary <- vehicle_summary %>% mutate(feature = "Vehicle")
area_summary <- area_summary %>% mutate(feature = "Area")
category_summary <- category_summary %>% mutate(feature = "Category")
````

# 合并所有数据框
````r
all_summary <- bind_rows(weather_summary, traffic_summary, vehicle_summary, area_summary, category_summary)

# 绘制整合图表并标记数字
p<- ggplot(all_summary, aes(x = value, y = proportion, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)), 
            position = position_dodge(width = 0.9), vjust = -0.25, size = 3) +
  facet_grid(cluster ~ feature) +
  labs(title = "Proportion of Different Features in Each Cluster", x = "Value", y = "Proportion") +
  theme_minimal()
  
  library(patchwork) 
  
q<- ggplot(numeric_summary_melt, aes(x = value, y = feature, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Numerical Feature Means by Cluster", x = "Mean Value", y = "Cluster") +
  theme_minimal()

# 合并图表
combined_plot <- p1 + p2 + plot_layout(ncol = 1)

# 展示合并后的图表
print(combined_plot)
````

# 计算数值型特征的均值和标准差
````r
numeric_summary <- data %>%
  group_by(cluster) %>%
  summarise(across(c(Agent_Age, Agent_Rating, Delivery_Time), 
                   list(mean = ~mean(.), sd = ~sd(.)), .names = "{col}_{fn}"))

# 将数据重塑以适合绘图
numeric_summary_melt <- numeric_summary %>%
  pivot_longer(cols = -cluster, names_to = c("feature", ".value"), names_sep = "_")

# 查看重塑后的数据
print(numeric_summary_melt)
````


# 计算数值型特征的均值和标准差
````r
numeric_summary <- data %>%
  group_by(cluster) %>%
  summarise(across(c(Agent_Age, Agent_Rating, Delivery_Time), 
                   list(mean = ~mean(.), sd = ~sd(.)), .names = "{col}_{fn}"))

# 将数据重塑以适合绘图
numeric_summary_melt <- numeric_summary %>%
  pivot_longer(cols = -cluster, names_to = c("feature", ".value"), names_sep = "_")

# 查看重塑后的数据
print(numeric_summary_melt)
````

# 合并所有类别特征的数据
````r
all_summary <- bind_rows(
  category_summary %>% mutate(feature = "Category"),
  weather_summary %>% mutate(feature = "Weather"),
  traffic_summary %>% mutate(feature = "Traffic"),
  vehicle_summary %>% mutate(feature = "Vehicle"),
  area_summary %>% mutate(feature = "Area")
)

# 添加数值型特征
numeric_summary_melt <- numeric_summary_melt %>%
  mutate(proportion = mean, value = feature) %>%
  select(cluster, value, proportion, feature)

# 合并数值型和类别型特征的数据
all_summary <- bind_rows(all_summary, numeric_summary_melt)

library(ggplot2)

# 绘制合并后的图表
p <- ggplot(all_summary, aes(x = value, y = proportion, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)), 
            position = position_dodge(width = 0.9), vjust = -0.25, size = 3) +
  facet_grid(cluster ~ feature, scales = "free_y") +
  labs(title = "Proportion and Mean of Different Features in Each Cluster", 
       x = "Feature", y = "Proportion / Mean") +
  theme_minimal()

# 展示图表
print(p)
````























#创建一个Dashboard来展示所有结果
library(shiny)

ui <- fluidPage(
  titlePanel("K-means聚类分析Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("聚类评估指标"),
      verbatimTextOutput("silhouette_score"),
      #verbatimTextOutput("calinski_harabasz_score"),
      verbatimTextOutput("cv_score"),
      h4("聚类样本量和占比"),
      tableOutput("cluster_summary")
    ),
    mainPanel(
      h4("数值型特征汇总"),
      tableOutput("numeric_summary"),
      h4("分类型特征汇总"),
      tableOutput("categorical_summary")
    )
  )
)

server <- function(input, output) {
  output$silhouette_score <- renderText({ paste("轮廓系数:", silhouette_score) })
  #output$calinski_harabasz_score <- renderText({ paste("Calinski-Harabasz指数:", calinski_harabasz_score) })
  output$cv_score <- renderText({ paste("交叉验证得分:", cv_score) })
  output$cluster_summary <- renderTable({cluster_summary})
  output$numeric_summary <- renderTable({numeric_summary})
  output$categorical_summary <- renderTable({categorical_summary})
}

shinyApp(ui = ui, server = server)
