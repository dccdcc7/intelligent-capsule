import numpy as np

def conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

# 输入尺寸
input_size = 200

# 第一个卷积层
conv1_kernel_size = 5
conv1_padding = 1
conv1_stride = 2

# 计算第一个卷积层的输出尺寸
output_size_conv1 = conv_output_size(input_size, conv1_kernel_size, conv1_padding, conv1_stride)
print(f"Output size after first convolution: {output_size_conv1} x {output_size_conv1}")

# 池化层
pool_kernel_size = 3
pool_padding = 0
pool_stride = 1

# 计算池化层的输出尺寸
output_size_pool = conv_output_size(output_size_conv1, pool_kernel_size, pool_padding, pool_stride)
print(f"Output size after pooling: {output_size_pool} x {output_size_pool}")

# 第二个卷积层
conv2_kernel_size = 3
conv2_padding = 1
conv2_stride = 1

# 计算第二个卷积层的输出尺寸
output_size_conv2 = conv_output_size(output_size_pool, conv2_kernel_size, conv2_padding, conv2_stride)
print(f"Output size after second convolution: {output_size_conv2} x {output_size_conv2}")