````r
library(magick)
library(imager)
library(magrittr)
library(dplyr)
library(stats)
library(graphics)
library(base)
library(arules)
````

# 读取图像
````r
# 确保路径正确并且文件存在
image_path <- file.choose()
if (file.exists(image_path)) {
  image <- load.image(image_path)
} else {
  stop("图像文件不存在，请检查路径")
}

# 读取图像
image_path <- "/mnt/data/20240413154039_1.jpg"
image <- image_read(image_path)
print(image)
````

# 检查图像是否成功读取
````r
if (is.null(image)) {
  stop("无法读取图像。请检查文件路径和文件格式。")
  ig<-image_read(image_path)
}
````

# 图像旋转处理
````r
rotated_image <- image_rotate(image, 45) # 旋转45度
print(rotated_image)

# 将magick图像转换为imager图像
im_image <- image_read(image_path)
im_image <- as.cimg(im_image)

filenames <- list.files(file.choose(), pattern="*.jpg", full.names = T)
if(!is.null(filenames)){
  for(idx in filenames) {
    im <- idx
    print(im)
    loaded_image <- load.image(im);
    
    im1 <-grayscale(loaded_image)



    # 灰度图像转换
    if (spectrum(image) == 1) {
      gray_image <- image
    } else {
      gray_image <- grayscale(image)
    }
    plot(gray_image, main = "Grayscale Image")
````

# 边缘检测
````r
edge_image <- imgradient(gray_image, "xy") %>% enorm()
plot(edge_image)
````

# 图像二值化
````
binary_image <- threshold(image, "50%")
plot(binary_image)
````

# 图像平滑处理
````
smoothed_image <- isoblur(gray_image, 2) # 使用高斯模糊，sigma = 2
plot(smoothed_image)
````

# 形态学处理（例如腐蚀和膨胀）
````r
# 膨胀
dilated_image <- morphology(binary_image,  method = "convolve",
                            kernel = "Gaussian",)
plot(dilated_image, main = "Dilated Image")
````

# 腐蚀
````r
eroded_image <- morphology(binary_image, kernel = "box", size = 3, operation = "erode")
plot(eroded_image)

# 显示原始图像
plot(image, main = "Original Image")
````

# 图像锐化处理
````r
# 使用unsharp masking技术
sharpened_image <- imsharpen(image,10, type = "diffusion", edge = 1, alpha = 0, sigma = 0)

# 显示锐化后的图像
plot(sharpened_image, main = "Sharpened Image")
````
