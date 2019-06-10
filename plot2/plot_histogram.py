import matplotlib.pyplot as plt
import numpy as np
x = [1,2]   #横坐标
y = [3,4]   #第一个纵坐标
y1 = [5,6]   #第二个纵坐标

# MAD-GAN
a1 =  [0.075999999999999998, 0.0, 0.69399999999999995, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
       0.0, 0.86099999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.56200000000000006, 1.0, 1.0, 1.0, 0.106]
# oc-svm
a2 =  [0.0, 0.0, 0.40899999999999997, 0.0, 0.748, 0.751, 0.75, 0.748, 0.75, 0.0, 0.0, 0.93500000000000005, 0.0, 0.0, 0.0]
#
# a3 =
mado_wadi = [0.0, 0.0, 0.40899999999999997, 0.0, 0.748, 0.751, 0.75, 0.748, 0.75, 0.0, 0.0, 0.93500000000000005, 0.0, 0.0, 0.0]
madgan_wadi =  [0.0, 1.0, 0.64400000000000002, 0.0, 0.0, 0.0, 0.92000000000000004, 1.0, 1.0, 0.0, 0.0, 0.0060000000000000001, 0.19900000000000001, 1.0, 1.0]

swat_anomaly_len =[939, 442, 382, 389, 195, 428, 963, 720, 720, 232, 430, 275, 716, 258, 394, 720, 462,
                  696, 320, 611, 1444, 35899, 120, 1170, 366, 600, 443, 100, 480, 539, 468, 280, 400, 296, 1689]

wadi_anomaly_len =[600, 1740, 850, 600, 1740, 850, 300, 202, 576, 87, 806, 629, 360, 202, 576]
s1 = 0
s2 = 0
sn1 = 0
sn2 = 0
sn3 = 0
sn4 = 0
for i, element in enumerate(wadi_anomaly_len):
    if i in [4,5,8,13]:
        s1 += element
        sn1 += mado_wadi[i] * element
        sn2 += madgan_wadi[i] * element
    else:
        s2 += element
        sn3 += mado_wadi[i] * element
        sn4 += madgan_wadi[i] * element
print(s1, s2)
print(sn1, sn2)

mado = [sn3/s2, sn1/s1]
madgan = [sn4/s2, sn2/s1]

print(len(a1), len(wadi_anomaly_len))
List3 = np.multiply(np.array(mado_wadi),np.array(wadi_anomaly_len))
print(List3.tolist())



# plt.title('WADI')
name_list = ['Single Point Attack', 'Multi Point Attacks']
num_list1 =[sn3/s2, sn1/s1]
num_list = [sn4/s2, sn2/s1]
x = list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='MAD-GAN', fc="#c9d9d3")
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='TAO-Learner', tick_label=name_list, fc="#718dbf")
plt.legend()
plt.show()


x = np.arange(len(mado))  #首先用第一个的长度作为横坐标
width = 0.2    #设置柱与柱之间的宽度
fig,ax = plt.subplots()
ax.bar(x,madgan,width,alpha = 0.9, color="#c9d9d3", label='MAD-GAN',tick_label=['single','multi'])
ax.bar(x+width,mado,width,alpha = 0.9,color="#718dbf", label='TAO-Learner',tick_label=['single','multi'])
ax.set_xticks(x +width/2)#将坐标设置在指定位置
ax.set_xticklabels(x)#将横坐标替换成

plt.show()

p = figure(x_range=fruits, y_range=(0, 10), plot_height=350, title="Fruit Counts by Year",tools="")

p.vbar(x=y, top='2015', width=0.2, source=source,color="#c9d9d3", legend=value("2015")) #用dodge的方法把3个柱状图拼到了一起
p.vbar(x=y1, top='2016', width=0.2, source=source,color="#718dbf", legend=value("2016"))
# p.vbar(x=dodge('index',  0.25, range=p.x_range), top='2017', width=0.2, source=source,color="#e84d60", legend=value("2017"))
# 绘制多系列柱状图       0.25和width=0.2是柱状图之间的空隙间隔，都是0.2了就没有空隙了
# dodge(field_name, value, range=None) → 转换成一个可分组的对象，value为元素的位置（配合width设置）
# value(val, transform=None) → 按照年份分为dict

p.xgrid.grid_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
# 其他参数设置

show(p)
