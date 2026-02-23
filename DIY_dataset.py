import warnings
import numpy as np
import torch
import math

torch.set_default_dtype(torch.float64)  # 精度默认为double类型


def make_disk_data(twist=2, sizepara=41):
    x1 = np.array([[0]])
    y1 = np.array([[0]])
    x2 = np.array([[0]])
    y2 = np.array([[0]])
    colors = np.array([[0]])
    for x0 in list(np.linspace(-1, 1, sizepara)):
        for y0 in list(np.linspace(-1, 1, sizepara)):
            r = np.sqrt(x0 ** 2 + y0 ** 2)
            theta = np.angle(complex(x0, y0))
            if 0 < r <= 1:
                x1 = np.concatenate((x1, np.array([[r * np.cos(theta)]])), axis=0)
                y1 = np.concatenate((y1, np.array([[r * np.sin(theta)]])), axis=0)
                x2 = np.concatenate((x2, np.array([[r * np.cos(twist * theta)]])), axis=0)
                y2 = np.concatenate((y2, np.array([[r * np.sin(twist * theta)]])), axis=0)
                colors = np.concatenate((colors, np.array([[r]])), axis=0)

    put = torch.tensor(np.concatenate((x1, y1), axis=1), dtype=torch.float64)
    output = torch.tensor(np.concatenate((x2, y2), axis=1), dtype=torch.float64)
    print('数据集已生成，扭转次数为%d' % twist)
    return put, output, colors


def make_ring_data(r=2, sizepara=1000, mindist=2, in_dim=3, out_dim=2):
    # sizepara is the num of segments
    data1 = np.zeros([sizepara, 3])
    data2 = np.zeros([sizepara, 3])
    data3 = np.zeros([sizepara, 2])
    data4 = np.zeros([sizepara, 2])
    data5 = np.zeros([sizepara, 1])
    data6 = np.zeros([sizepara, 1])
    colors = np.zeros([2 * sizepara, 1])
    thetalist = np.linspace(start=0, stop=2*np.pi, num=sizepara, endpoint=False)
    for i in range(sizepara):
        data1[i, :] = [-r / 2 + r * np.cos(thetalist[i]), r * np.sin(thetalist[i]), 0]
        data2[i, :] = [r / 2 + r * np.cos(thetalist[i]), 0, r * np.sin(thetalist[i])]
        data3[i, :] = [-(mindist/2 + r) + r * np.cos(thetalist[i]), r * np.sin(thetalist[i])]
        data4[i, :] = [(mindist/2 + r) + r * np.cos(thetalist[i]), r * np.sin(thetalist[i])]
        data5[i] = 1 + (2 - 1) * i / (sizepara - 1)
        data6[i] = -1 - (2 - 1) * i / (sizepara - 1)
        colors[i] = thetalist[i]
        colors[i+sizepara] = thetalist[i] + 2 * np.pi
    if in_dim == 3 and out_dim == 2:
        put = torch.tensor(np.concatenate((data1, data2), axis=0), dtype=torch.float64)
        output = torch.tensor(np.concatenate((data3, data4), axis=0), dtype=torch.float64)
    elif in_dim == 2 and out_dim == 3:
        put = torch.tensor(np.concatenate((data3, data4), axis=0), dtype=torch.float64)
        output = torch.tensor(np.concatenate((data1, data2), axis=0), dtype=torch.float64)
    elif in_dim == 1 and out_dim == 3:
        put = torch.tensor(np.concatenate((data5, data6), axis=0), dtype=torch.float64)
        output = torch.tensor(np.concatenate((data1, data2), axis=0), dtype=torch.float64)
    elif in_dim == 3 and out_dim == 1:
        put = torch.tensor(np.concatenate((data1, data2), axis=0), dtype=torch.float64)
        output = torch.tensor(np.concatenate((data5, data6), axis=0), dtype=torch.float64)
    else:
        put = torch.tensor(np.concatenate((data1, data2), axis=0), dtype=torch.float64)
        output = torch.tensor(np.concatenate((data3, data4), axis=0), dtype=torch.float64)
        print('输入维数错误，已按照in_dim=3, out_dim=2处理')
    print('数据集已生成，环的半径为%f, 输入维数为%d, 输出维数为%d' % (r, in_dim, out_dim))
    return put, output, colors


def make_belt_data(r1=3, r2=7, sizepara=101, inner=False, sizepara2=51):
    x1 = np.array([[0]])
    y1 = np.array([[0]])
    x2 = np.array([[0]])
    y2 = np.array([[0]])
    colors = np.array([[0]])
    for x0 in list(np.linspace(-r2, r2, sizepara)):
        for y0 in list(np.linspace(-r2, r2, sizepara)):
            r = np.sqrt(x0 ** 2 + y0 ** 2)
            if r1 <= r <= r2:
                theta = np.angle(complex(x0, y0))
                x1 = np.concatenate((x1, np.array([[r * np.cos(theta)]])), axis=0)
                y1 = np.concatenate((y1, np.array([[r * np.sin(theta)]])), axis=0)
                x2 = np.concatenate((x2, np.array([[(r1 + r2 - r) * np.cos(theta)]])), axis=0)
                y2 = np.concatenate((y2, np.array([[(r1 + r2 - r) * np.sin(theta)]])), axis=0)
                colors = np.concatenate((colors, np.array([[r]])), axis=0)
    if inner:
        for x0 in list(np.linspace(-r1, r1, sizepara2)):
            for y0 in list(np.linspace(-r1, r1, sizepara2)):
                r = np.sqrt(x0 ** 2 + y0 ** 2)
                if 0 <= r <= r1:
                    theta = np.angle(complex(x0, y0))
                    x1 = np.concatenate((x1, np.array([[r * np.cos(theta)]])), axis=0)
                    y1 = np.concatenate((y1, np.array([[r * np.sin(theta)]])), axis=0)
                    x2 = np.concatenate((x2, np.array([[(r1 + r2 - r) * np.cos(theta)]])), axis=0)
                    y2 = np.concatenate((y2, np.array([[(r1 + r2 - r) * np.sin(theta)]])), axis=0)
                    colors = np.concatenate((colors, np.array([[r1-0.1]])), axis=0)
    x1 = x1[1:, :]
    y1 = y1[1:, :]
    x2 = x2[1:, :]
    y2 = y2[1:, :]
    colors = colors[1:, :]

    put = torch.tensor(np.concatenate((x1, y1), axis=1), dtype=torch.float64)
    output = torch.tensor(np.concatenate((x2, y2), axis=1), dtype=torch.float64)
    print('数据集已生成，环的内半径为%f，外半径为%f' % (r1, r2))
    return put, output, colors


def make_knot_data(p=2, q=3, sizepara=1001, bias=0):
    data1 = np.zeros([sizepara, 1])
    data2 = np.zeros([sizepara, 3])
    colors = np.zeros([sizepara, 1])
    thetalist = np.linspace(start=bias, stop=2*np.pi*p/math.gcd(p, q)-bias, num=sizepara, endpoint=True)
    for i in range(sizepara):
        data1[i] = i * (1 - 0)/(sizepara - 1)
        data2[i, :] = [(2 + np.cos(q * thetalist[i] / p)) * np.cos(thetalist[i]),
                       (2 + np.cos(q * thetalist[i] / p)) * np.sin(thetalist[i]),
                       np.sin(q * thetalist[i] / p)]
        colors[i] = thetalist[i]

    line = torch.tensor(data1, dtype=torch.float64)
    knot = torch.tensor(data2, dtype=torch.float64)
    print('数据集已生成，为(%d, %d)环面扭结' % (p, q))
    return line, knot, colors


def make_klein_data(zmax=1, sizepara=1001):
    thetalist = np.linspace(start=0, stop=2 * np.pi, num=sizepara, endpoint=True)
    data1 = np.zeros([sizepara * 101, 3])
    data2 = np.zeros([sizepara * 101, 3])
    colors = np.zeros([sizepara * 101, 1])

    for i in range(101):
        for j in range(sizepara):
            data1[i * sizepara + j] = [np.cos(thetalist[j]), np.sin(thetalist[j]), zmax/100 * i]
            data2[i * sizepara + j] = [np.cos(thetalist[j])-np.sin(2*np.pi*(zmax/100*i)), np.sin(thetalist[j]),
                                       np.sin(np.pi*(zmax/100*i))]
            colors[i * sizepara + j] = zmax/100 * i
    put = torch.tensor(data1, dtype=torch.float64)
    output = torch.tensor(data2, dtype=torch.float64)
    print('数据集已生成，克莱因瓶zmax=%f' % zmax)
    return put, output, colors


def make_S1_data(r=1, sizepara=10000):
    x1 = np.zeros((sizepara, 1))
    y1 = np.zeros((sizepara, 1))
    x2 = np.zeros((sizepara, 1))
    y2 = np.zeros((sizepara, 1))
    colors = np.zeros([sizepara, 1])
    thetalist = np.linspace(start=0, stop=2 * np.pi, num=sizepara, endpoint=True)
    for i in range(sizepara):
        theta = thetalist[i]
        x1[i] = np.array([[r * np.cos(theta)]])
        y1[i] = np.array([[r * np.sin(theta)]])
        x2[i] = np.array([[r * np.cos(2 * theta)]])
        y2[i] = np.array([[r * np.sin(2 * theta)]])
        colors[i] = np.array([[theta]])
    put = torch.tensor(np.concatenate((x1, y1), axis=1), dtype=torch.float64)
    output = torch.tensor(np.concatenate((x2, y2), axis=1), dtype=torch.float64)
    print('数据集已生成，S1的r=%f' % r)
    return put, output, colors

def make_polyline_data(tail=1, a=2, sizepara=1001):
    put_np = np.linspace(start=0, stop=1, num=sizepara, endpoint=True).reshape(sizepara, 1)
    output_np = np.zeros([sizepara, 2])
    colors = np.linspace(start=0, stop=1, num=sizepara, endpoint=True).reshape(sizepara, 1)
    t_list = np.linspace(start=0, stop=2 * tail+ 4 * a, num=sizepara, endpoint=True)
    for index in range(sizepara):
        t= t_list[index]
        if 0 <= t < tail + a:
            output_np[index, :] = [t - tail, 0]
        elif tail + a <= t < tail + 2 * a:
            output_np[index, :] = [a, t - (tail + a)]
        elif tail + 2 * a <= t < tail + 3 * a:
            output_np[index, :] = [tail + 3 * a - t, a]
        elif tail + 3 * a <= t <= 2 * tail + 4 * a:
            output_np[index, :] = [0, 4 * a + tail - t]
    put = torch.tensor(put_np, dtype=torch.float64)
    output = torch.tensor(output_np, dtype=torch.float64)
    return put, output, colors

def make_outrot_data(r1=1, r2=2, sizepara=101, rottheta=np.pi/3):
    # sizepara is the num of segments

    x1 = np.array([[0]])
    y1 = np.array([[0]])
    x2 = np.array([[0]])
    y2 = np.array([[0]])
    colors = np.array([[4]])
    for x0 in list(np.linspace(-r2, r2, sizepara)):
        for y0 in list(np.linspace(-r2, r2, sizepara)):
            r = np.sqrt(x0 ** 2 + y0 ** 2)
            theta = np.angle(complex(x0, y0))
            if 0 < r <= r1:
                x1 = np.concatenate((x1, np.array([[r * np.cos(theta)]])), axis=0)
                y1 = np.concatenate((y1, np.array([[r * np.sin(theta)]])), axis=0)
                x2 = np.concatenate((x2, np.array([[r * np.cos(theta)]])), axis=0)
                y2 = np.concatenate((y2, np.array([[r * np.sin(theta)]])), axis=0)
                colors = np.concatenate((colors, np.array([[4]])), axis=0)
            if r1 < r <= r2:
                x1 = np.concatenate((x1, np.array([[r * np.cos(theta)]])), axis=0)
                y1 = np.concatenate((y1, np.array([[r * np.sin(theta)]])), axis=0)
                x2 = np.concatenate((x2, np.array([[r * np.cos(theta+(r-r1)/(r2-r1)*rottheta)]])), axis=0)
                y2 = np.concatenate((y2, np.array([[r * np.sin(theta+(r-r1)/(r2-r1)*rottheta)]])), axis=0)
                colors = np.concatenate((colors, np.array([[theta]])), axis=0)

    put = torch.tensor(np.concatenate((x1, y1), axis=1), dtype=torch.float64)
    output = torch.tensor(np.concatenate((x2, y2), axis=1), dtype=torch.float64)
    print('数据集已生成，内半径和外半径为%f, %f' % (r1, r2))
    return put, output, colors

def make_swissroll_data(radiusrange=2, thetapara=np.pi, height=1, sizepara=None, in_out = 23):
    if sizepara is None:
        sizepara = [21, 6]
    x1 = np.zeros((sizepara[0] * sizepara[1], 1))
    x2 = np.zeros((sizepara[0] * sizepara[1], 1))
    x3 = np.zeros((sizepara[0] * sizepara[1], 1))
    x4 = np.zeros((sizepara[0] * sizepara[1], 1))
    heightlist = np.linspace(start=0, stop=height, num=sizepara[1], endpoint=True).reshape(sizepara[1], 1)
    radiuslist = np.linspace(start=0, stop=radiusrange, num=sizepara[0], endpoint=True).reshape(sizepara[0], 1)
    for h in range(sizepara[1]):
        for i in range(sizepara[0]):
            x1[h * sizepara[0] + i] = heightlist[h]
            x2[h * sizepara[0] + i] = radiuslist[i] * np.cos(radiuslist[i] * thetapara)
            x3[h * sizepara[0] + i] = radiuslist[i] * np.sin(radiuslist[i] * thetapara)
            x4[h * sizepara[0] + i] = radiuslist[i]
    temp1 = np.concatenate((x1, x2), axis=1)
    if in_out == 23:
        put = torch.tensor(np.concatenate((x4, x1), axis=1), dtype=torch.float64)
        output = torch.tensor(np.concatenate((temp1, x3), axis=1), dtype=torch.float64)
    elif in_out == 32:
        put = torch.tensor(np.concatenate((temp1, x3), axis=1), dtype=torch.float64)
        output = torch.tensor(np.concatenate((x4, x1), axis=1), dtype=torch.float64)
    else:
        warnings.warn('输入输出维数错误')
    colors = x4
    print('数据集已生成，大小为%d乘%d' % (sizepara[0], sizepara[1]))
    return put, output, colors

def make_thickSn_data(r1=3, r2=7, sndim=2, sizepara=21):
    assert r1<=r2
    axisdata = np.linspace(start=-r2, stop=r2, endpoint=True, num=sizepara)
    data_raw = axisdata.reshape((sizepara, 1))
    for i in range(1, sndim+1):
        data_temp1 = axisdata[0] * np.ones((np.pow(sizepara, i), 1))
        data_temp_new = np.concatenate((data_temp1, data_raw), axis=1)
        for j in axisdata[1:]:
            data_temp1 = j * np.ones((np.pow(sizepara, i), 1))
            data_temp2 = np.concatenate((data_temp1, data_raw), axis=1)
            data_temp_new = np.concatenate((data_temp_new, data_temp2), axis=0)
        data_raw = data_temp_new.copy()
    data_sphere = np.ones((1, sndim+1))
    for k in range(np.size(data_raw, 0)):
        print('data_sphere创建进度：%d/%d' % (k+1, np.pow(sizepara, sndim+1)))
        if r1 <= np.linalg.norm(data_raw[k, :], ord=2) <= r2:
            data_sphere = np.vstack((data_sphere, data_raw[k, :]))
    data_sphere = np.delete(data_sphere, 0, axis=0)
    data_sphere_reverse = np.copy(data_sphere)

    colors = []
    for m in range(np.size(data_sphere_reverse, 0)):
        r = np.linalg.norm(data_sphere[m, :], ord=2)
        colors.append(r)
        data_sphere_reverse[m, :] = data_sphere_reverse[m, :] * (r1+r2-r) / r
    colors = np.array(colors)

    sphere1 = torch.tensor(data_sphere, dtype=torch.float64)
    sphere2 = torch.tensor(data_sphere_reverse, dtype=torch.float64)

    return sphere1, sphere2, colors

date_str = '26-02-24'
if __name__ == '__main__':
    print('最新更改日期：%s' % date_str)
    print('作者：周琦')
    print('联系方式：2517036487@qq.com')
else:
    print('已导入数据集', __name__, '最新更改日期:', date_str)