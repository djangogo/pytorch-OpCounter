from OpCounter.conv import Conv2d


c = Conv2d(30, 10, 3)
print(c.__flops__)