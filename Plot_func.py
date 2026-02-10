import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import warnings
import TopoDataSyn
import TopoDataSyn.NN_model as NN_model
from itertools import islice
from collections import OrderedDict

from matplotlib.lines import lineStyles

torch.set_default_dtype(torch.float64)  # 精度默认为double类型

def axis_visualize(ax, axisnum=2, linewidth=3, linestyle='-'):
    assert axisnum == 3 or axisnum == 2
    if axisnum >=3:
        z_bound1, z_bound2 = ax.get_zbound()
    if axisnum >=2:
        y_bound1, y_bound2 = ax.get_ybound()
    if axisnum >=1:
        x_bound1, x_bound2 = ax.get_xbound()

    if axisnum ==3:
        if x_bound1 <= 0 and x_bound2 >= 0:
            if y_bound1 <= 0 and y_bound2 >= 0:
                line_z = np.zeros([100, 3])
                line_z[:, 0] = np.zeros([100])
                line_z[:, 1] = np.zeros([100])
                line_z[:, 2] = np.linspace(z_bound1, z_bound2, 100, endpoint=True)
                ax.plot(line_z[:, 0], line_z[:, 1], line_z[:, 2], linewidth=linewidth, c='r', linestyle=linestyle, label='z-axis')
            if z_bound1 <= 0 and z_bound2 >= 0:
                line_y = np.zeros([100, 3])
                line_y[:, 0] = np.zeros([100])
                line_y[:, 1] = np.linspace(y_bound1, y_bound2, 100, endpoint=True)
                line_y[:, 2] = np.zeros([100])
                ax.plot(line_y[:, 0], line_y[:, 1], line_y[:, 2], linewidth=linewidth, c='r', linestyle=linestyle, label='y-axis')
        if y_bound1 <= 0 and y_bound2 >= 0:
            if z_bound1 <= 0 and z_bound2 >= 0:
                line_x = np.zeros([100, 3])
                line_x[:, 0] = np.linspace(x_bound1, x_bound2, 100, endpoint=True)
                line_x[:, 1] = np.zeros([100])
                line_x[:, 2] = np.zeros([100])
                ax.plot(line_x[:, 0], line_x[:, 1], line_x[:, 2], linewidth=linewidth, c='r', linestyle=linestyle, label='x-axis')

    if axisnum ==2:
        if x_bound1 <= 0 and x_bound2 >= 0:
            line_y = np.zeros([100, 2])
            line_y[:, 1] = np.linspace(y_bound1, y_bound2, 100, endpoint=True)
            ax.plot(line_y[:, 0], line_y[:, 1], linewidth=linewidth, c='r', linestyle=linestyle, label='y-axis')
        if y_bound1 <= 0 and y_bound2 >= 0:
            line_x = np.zeros([100, 2])
            line_x[:, 0] = np.linspace(x_bound1, x_bound2, 100, endpoint=True)
            ax.plot(line_x[:, 0], line_x[:, 1], linewidth=linewidth, c='r', linestyle=linestyle, label='x-axis')


def diskgif(size_option=None, iteration=100000, frameduration=1000, figdata=None, figcolor=None, output_path='./Disk.gif'):
    if size_option is None:
        size_option = [-1, 1, -5, 5, -1, 1]  # 图片大小默认值
    a1, b1, a2, b2, a3, b3 = size_option
    image_list = []
    figdata_0, figdata_1, figdata_2 = figdata
    for i in range(int(iteration / frameduration)):
        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax1.set_xlim(a1, b1)
        ax1.set_ylim(a1, b1)
        ax1.scatter(figdata_0[i][:, 0], figdata_0[i][:, 1], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)
        plt.title('INPUT')
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.set_xlim(a2, b2)
        ax2.set_ylim(a2, b2)
        ax2.set_zlim(a2, b2)
        ax2.scatter(figdata_1[i][:, 0], figdata_1[i][:, 1], figdata_1[i][:, 2], s=0.5, c=figcolor,
                    cmap=plt.cm.gist_rainbow)
        ax3 = fig.add_subplot(234)
        ax3.set_xlim(a3, b3)
        ax3.set_ylim(a3, b3)
        ax3.scatter(figdata_2[i][:, 0], figdata_2[i][:, 1], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)

        plt.title('itrations=%s' % (i * frameduration))
        plt.savefig('temp.png')
        image_list.append(imageio.imread('temp.png'))
        plt.close()
        imageio.mimsave(output_path, image_list, 'GIF', duration=0.25)


def Plot_mesh(net, mesh1, mesh2, linecolor='k', linewidth=0.5):
    '''
    Datatype:
    net -> torch.nn
    mesh1, mesh2 -> list
    Return:
    None
    '''

    for xp in mesh1:
        for j in range(len(mesh2)-1):
            P1 = net(torch.tensor([xp, mesh2[j]], dtype=torch.float64)).detach().cpu()
            P2 = net(torch.tensor([xp, mesh2[j+1]], dtype=torch.float64)).detach().cpu()
            plt.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], c=linecolor)

    for yp in mesh2:
        for i in range(len(mesh1) - 1):
            P1 = net(torch.tensor([mesh1[i], yp], dtype=torch.float64)).detach().cpu()
            P2 = net(torch.tensor([mesh1[i+1], yp], dtype=torch.float64)).detach().cpu()
            plt.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], c=linecolor)


def Plot_NN(figdata=None, figcolor=None, save_path=['./test.png']):
    """
    Datatype:
    figdata -> list of torch.tensor
    figcolor -> numpy.array
    output_path -> list of strings
    Return:
    None
    """

    figdata_in, figdata_out, figdata_test = figdata  # datatype: torch.tensor
    # 如果是1维线性数据，则扩容成2维
    if figdata_in.size()[1] == 1:
        warnings.warn('input data is 1-dimensional')
        figdata_in = torch.cat((figdata_in, torch.zeros(figdata_in.size()[0], 1)), dim=1)
    if figdata_out.size()[1] == 1:
        warnings.warn('output data is 1-dimensional')
        figdata_out = torch.cat((figdata_out, torch.zeros(figdata_out.size()[0], 1)), dim=1)
    if figdata_test.size()[1] == 1:
        warnings.warn('test data is 1-dimensional')
        figdata_test = torch.cat((figdata_test, torch.zeros(figdata_test.size()[0], 1)), dim=1)

    plt.figure(figsize=(16, 5))

    if figdata_in.size()[1] >= 3:
        if figdata_in.size()[1] >= 4:
            warnings.warn("the dimension of input data is over 3, only the selected 3 dimensions will be plotted")
        ax_in = plt.subplot2grid((1, 3), (0, 0), projection='3d')
        ax_in.set_zlabel('z')
    else:
        ax_in = plt.subplot2grid((1, 3), (0, 0))
    ax_in.set_xlabel('x')
    ax_in.set_ylabel('y')
    ax_in.set_aspect('equal')
    ax_in.set_rasterized(True)  # 将此图层栅格化
    if figdata_in.size()[1] >= 3:
        ax_in.scatter(figdata_in[:, 0], figdata_in[:, 1], figdata_in[:, 2], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)
    else:
        ax_in.scatter(figdata_in[:, 0], figdata_in[:, 1], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)

    if figdata_out.size()[1] >= 3:
        if figdata_out.size()[1] >= 4:
            warnings.warn("the dimension of output data is over 3, only the first 3 dimensions will be plotted")
        ax_out = plt.subplot2grid((1, 3), (0, 1), projection='3d')
        ax_out.set_zlabel('z')
    else:
        ax_out = plt.subplot2grid((1, 3), (0, 1))
    ax_out.set_xlabel('x')
    ax_out.set_ylabel('y')
    ax_out.set_aspect('equal')
    ax_out.set_rasterized(True)  # 将此图层栅格化
    if figdata_out.size()[1] >= 3:
        ax_out.scatter(figdata_out[:, 0], figdata_out[:, 1], figdata_out[:, 2], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)
    else:
        ax_out.scatter(figdata_out[:, 0], figdata_out[:, 1], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)

    if figdata_test.size()[1] >= 3:
        if figdata_test.size()[1] >= 4:
            warnings.warn("the dimension of test data is over 3, only the first 3 dimensions will be plotted")
        ax_test = plt.subplot2grid((1, 3), (0, 2), projection='3d')
        ax_test.set_zlabel('z')
    else:
        ax_test = plt.subplot2grid((1, 3), (0, 2))
    ax_test.set_xlabel('x')
    ax_test.set_ylabel('y')
    ax_test.set_aspect('equal')
    ax_test.set_rasterized(True)  # 将此图层栅格化
    if figdata_test.size()[1] >= 3:
        ax_test.scatter(figdata_test[:, 0], figdata_test[:, 1], figdata_test[:, 2], s=0.5, c=figcolor,
                       cmap=plt.cm.gist_rainbow)
    else:
        ax_test.scatter(figdata_test[:, 0], figdata_test[:, 1], s=0.5, c=figcolor, cmap=plt.cm.gist_rainbow)

    for p in save_path:
        plt.savefig('%s' % p, bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()


def Plot_layers(modelpath, data_in, acf=torch.nn.ELU(), figcolor=None, fig_row_col=[2, 4], save_path=[], layer_select=[1, -1],
                activated_pre=False, *, axis_visible = False, cmap='viridis', dim_index=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                layerwise_paranum=2):
    # 如果是1维线性数据，则扩容成2维
    if data_in.size()[1] == 1:
        warnings.warn('input data is 1-dimensional')
        pic_data_in = torch.cat((data_in, torch.zeros(data_in.size()[0], 1)), dim=1).detach().numpy()
    else:
        pic_data_in = data_in.detach().numpy()

    plt.figure(figsize=(16, 10))

    if data_in.size()[1] >= 3:
        if data_in.size()[1] >= 4:
            warnings.warn("the dimension of input data is over 3, only the first 3 dimensions will be plotted")
        ax_in = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (0, 0), projection='3d')
        ax_in.set_zlabel('z')
    else:
        ax_in = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (0, 0))
    ax_in.set_xlabel('x')
    ax_in.set_ylabel('y')
    ax_in.set_aspect('equal')
    ax_in.set_rasterized(True)  # 将此图层栅格化
    ax_in.set_title('INPUT')
    if data_in.size()[1] >= 3:
        ax_in.scatter(pic_data_in[:, dim_index[0][0]], pic_data_in[:, dim_index[0][1]], pic_data_in[:, dim_index[0][2]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_in, 3)
    else:
        ax_in.scatter(pic_data_in[:, dim_index[0][0]], pic_data_in[:, dim_index[0][1]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_in, 2)

    old_dict = torch.load(modelpath, map_location='cpu', weights_only=True)['net']
    layer = []

    depth = (len(old_dict) - 2) // layerwise_paranum
    w_in = old_dict['net1.0.weight'].size()[1]
    w_out = old_dict['net2.0.bias'].size()[0]

    for i in range(depth):
        layer.append(old_dict['net1.%d.bias' % (2 * i)].size()[0])

    layer1 = layer_select[0]
    if layer_select[1]==-1:
        layer2 = depth
    else:
        layer2 = layer_select[1]
    print('本次输出%d到%d层的图像' % (layer1, layer2))
    assert layer2 <= depth
    # 显示从a层到b层的图像

    if activated_pre:
        print('本次将输出每一层激活前的图像')
        assert 2 * (layer2 - layer1 + 1) + 2 <= fig_row_col[0] * fig_row_col[1]
    else:
        assert (layer2 - layer1 + 1) + 2 <= fig_row_col[0] * fig_row_col[1]

    for d in range(layer1, layer2+1):
        if activated_pre:
            a, b = 2 * (d + 1 - layer1) // fig_row_col[1], 2 * (d + 1 - layer1) % fig_row_col[1]
            a1, b1 = (2 * (d + 1 - layer1) - 1)// fig_row_col[1], (2 * (d + 1 - layer1) - 1) % fig_row_col[1]
        else:
            a, b = (d + 1 - layer1) // fig_row_col[1], (d + 1 - layer1) % fig_row_col[1]

        islice_re = islice(old_dict, layerwise_paranum * d)  # islice()获取前n个元素
        new_dict = OrderedDict()  # 准备一个新字典存放要取的元素
        for i in islice_re:
            new_dict[i] = old_dict[i]  # 从旧字典取值赋值到新字典
        new_dict['net2.0.weight'] = torch.rand(w_out, layer[d-1])
        new_dict['net2.0.bias'] = torch.rand(w_out)

        net = NN_model.Netpre(w_in=w_in, w_out=w_out, layer=layer[0:d], acf=acf)
        net.load_state_dict(new_dict)
        pic_data_hid = net(data_in).detach().numpy()

        if activated_pre:
            del net.net1[-1]
            pic_data_hid_pre = net(data_in).detach().numpy()

        if layer[d-1] >= 3:
            if layer[d-1] >= 4:
                warnings.warn("the dimension of hidden layer data is over 3, only the selected 3 dimensions will be plotted")
            ax_hid = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a, b), projection='3d')
            ax_hid.set_zlabel('z')
            if activated_pre:
                ax_hid_pre = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a1, b1), projection='3d')
                ax_hid_pre.set_zlabel('z')
        else:
            ax_hid = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a, b))
            if activated_pre:
                ax_hid_pre = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a1, b1))
        ax_hid.set_xlabel('x')
        ax_hid.set_ylabel('y')
        ax_hid.set_aspect('equal')
        ax_hid.set_rasterized(True)  # 将此图层栅格化
        ax_hid.set_title('Layer%d' % d)
        if activated_pre:
            ax_hid_pre.set_xlabel('x')
            ax_hid_pre.set_ylabel('y')
            ax_hid_pre.set_aspect('equal')
            ax_hid_pre.set_rasterized(True)  # 将此图层栅格化
            ax_hid_pre.set_title('Layer%d(not activated)' % d)
        if layer[d-1] >= 3:
            ax_hid.scatter(pic_data_hid[:, dim_index[1][0]], pic_data_hid[:, dim_index[1][1]], pic_data_hid[:, dim_index[1][2]], s=0.5, c=figcolor, cmap=cmap)
            if axis_visible:
                axis_visualize(ax_hid, 3)
        else:
            ax_hid.scatter(pic_data_hid[:, dim_index[1][0]], pic_data_hid[:, dim_index[1][1]], s=0.5, c=figcolor, cmap=cmap)
            if axis_visible:
                axis_visualize(ax_hid, 2)


        if activated_pre:
            if layer[d - 1] >= 3:
                ax_hid_pre.scatter(pic_data_hid_pre[:, dim_index[1][0]], pic_data_hid_pre[:, dim_index[1][1]],
                               pic_data_hid_pre[:, dim_index[1][2]], s=0.5, c=figcolor, cmap=cmap)
                if axis_visible:
                    axis_visualize(ax_hid_pre, 3)
            else:
                ax_hid_pre.scatter(pic_data_hid_pre[:, dim_index[1][0]], pic_data_hid_pre[:, dim_index[1][1]], s=0.5, c=figcolor,
                               cmap=cmap)
                if axis_visible:
                    axis_visualize(ax_hid_pre, 2)


    if activated_pre:
        a, b = (2 * (layer2 + 1 - layer1) + 1) // fig_row_col[1], (2 * (layer2 + 1 - layer1) + 1) % fig_row_col[1]
    else:
        a, b = (1 * (layer2 + 1 - layer1) + 1) // fig_row_col[1], (1 * (layer2 + 1 - layer1) + 1) % fig_row_col[1]

    net = NN_model.Net(w_in=w_in, w_out=w_out, layer=layer, acf=acf)
    net.load_state_dict(old_dict)
    pic_data_out = net(data_in)
    if pic_data_out.size()[1] == 1:
        warnings.warn('output data is 1-dimensional')
        pic_data_out = torch.cat((pic_data_out, torch.zeros(pic_data_out.size()[0], 1)), dim=1).detach().numpy()
    else:
        pic_data_out = pic_data_out.detach().numpy()

    if w_out >= 3:
        if w_out >= 4:
            warnings.warn(
                "the dimension of output data is over 3, only the selected 3 dimensions will be plotted")
        ax_out = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a, b), projection='3d')
        ax_out.set_zlabel('z')
    else:
        ax_out = plt.subplot2grid((fig_row_col[0], fig_row_col[1]), (a, b))
    ax_out.set_xlabel('x')
    ax_out.set_ylabel('y')
    ax_out.set_aspect('equal')
    ax_out.set_rasterized(True)  # 将此图层栅格化
    ax_out.set_title('OUTPUT')

    if w_out >= 3:
        ax_out.scatter(pic_data_out[:, dim_index[2][0]], pic_data_out[:, dim_index[2][1]], pic_data_out[:, dim_index[2][2]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_out, 3)
    else:
        ax_out.scatter(pic_data_out[:, dim_index[2][0]], pic_data_out[:, dim_index[2][1]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_out, 2)

    for p in save_path:
        plt.savefig('%s' % p, bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()

def Plot_action_diagram(modelpath, data_in, acf=torch.nn.ELU(), figcolor=None, save_path=[], layer_select=1,
                      layerwise_paranum=2, dim_index=[[0, 1, 2], [0, 1, 2]], axis_visible=False, cmap='viridis'):
    '''
    Datatype:
    loss_record -> list of float
    save_path -> string
    Return:
    None
    '''
    print('本次输出第%d层的作用示意图' % layer_select)
    ### 准备画图数据pic_data_in, pic_data_pre, pic_data_out
    # 如果是1维线性数据，则扩容成2维
    if data_in.size()[1] == 1:
        warnings.warn('input data is 1-dimensional')
        pic_data_in = torch.cat((data_in, torch.zeros(data_in.size()[0], 1)), dim=1).detach().numpy()
    else:
        pic_data_in = data_in.detach().numpy()

    plt.figure(figsize=(16, 10))

    old_dict = torch.load(modelpath, map_location='cpu', weights_only=True)['net']
    layer = []

    w_in = old_dict['net1.0.weight'].size()[1]
    w_out = old_dict['net2.0.bias'].size()[0]

    for i in range(layer_select):
        layer.append(old_dict['net1.%d.bias' % (2 * i)].size()[0])

    # 创建第layer_select层的网络
    new_dict = OrderedDict()  # 准备一个新字典存放要取的元素
    islice_re = islice(old_dict, layerwise_paranum * layer_select)
    net = torch.nn.Sequential(torch.nn.Linear(w_in, layer[-1]))
    net[0].weight.data = old_dict['net1.%d.weight' % (2 * (layer_select - 1))]
    net[0].bias.data = old_dict['net1.%d.bias' % (2 * (layer_select - 1))]
    pic_data_pre = net(data_in).detach().numpy()
    print('加权矩阵', net[0].weight.data)
    print('偏置矩阵', net[0].bias.data)
    np_matrix = net[0].weight.data.detach().numpy()
    U, Sigma, Vt = np.linalg.svd(np_matrix)
    print('奇异值分解为: U=', U, 'Sigma=', Sigma ,'Vt=', Vt)
    e_val, e_vec = np.linalg.eigh(np_matrix)
    print('特征值分解为: e_val=', e_val, 'e_vec=', e_vec)


    net.add_module('1', acf)
    if layerwise_paranum == 3:
        net[1].weight.data = old_dict['net1.%d.weight' % (2 * (layer_select - 1) + 1)]
    pic_data_out = net(data_in).detach().numpy()
    print('激活函数', net[1])

    ### 开始画图
    if data_in.size()[1] >= 3:
        if data_in.size()[1] >= 4:
            warnings.warn("the dimension of input data is over 3, only the first 3 dimensions will be plotted")
        ax_in = plt.subplot2grid((1, 3), (0, 0), projection='3d')
        ax_in.set_zlabel('z')
    else:
        ax_in = plt.subplot2grid((1, 3), (0, 0))
    ax_in.set_xlabel('x')
    ax_in.set_ylabel('y')
    ax_in.set_aspect('equal')
    ax_in.set_rasterized(True)  # 将此图层栅格化
    ax_in.set_title('INPUT')

    if data_in.size()[1] >= 3:
        ax_in.scatter(pic_data_in[:, dim_index[0][0]], pic_data_in[:, dim_index[0][1]], pic_data_in[:, dim_index[0][2]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_in, 3)
    else:
        ax_in.scatter(pic_data_in[:, dim_index[0][0]], pic_data_in[:, dim_index[0][1]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_in, 2)

    if layer[-1] >= 3:
        if layer[-1] >= 4:
            warnings.warn("the dimension of output data is over 3, only the first 3 dimensions will be plotted")
        ax_pre = plt.subplot2grid((1, 3), (0, 1), projection='3d')
        ax_pre.set_zlabel('z')
        ax_out = plt.subplot2grid((1, 3), (0, 2), projection='3d')
        ax_out.set_zlabel('z')
    else:
        ax_pre = plt.subplot2grid((1, 3), (0, 1))
        ax_out = plt.subplot2grid((1, 3), (0, 2))

    ax_pre.set_xlabel('x')
    ax_pre.set_ylabel('y')
    ax_pre.set_aspect('equal')
    ax_pre.set_rasterized(True)  # 将此图层栅格化
    ax_pre.set_title('OUTPUT(not activated)')

    ax_out.set_xlabel('x')
    ax_out.set_ylabel('y')
    ax_out.set_aspect('equal')
    ax_out.set_rasterized(True)  # 将此图层栅格化
    ax_out.set_title('OUTPUT')

    if layer[-1] >= 3:
        ax_pre.scatter(pic_data_pre[:, dim_index[1][0]], pic_data_pre[:, dim_index[1][1]], pic_data_pre[:, dim_index[1][2]], s=0.5, c=figcolor, cmap=cmap)
        ax_out.scatter(pic_data_out[:, dim_index[1][0]], pic_data_out[:, dim_index[1][1]], pic_data_out[:, dim_index[1][2]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_pre, 3)
            axis_visualize(ax_out, 3)
    else:
        ax_pre.scatter(pic_data_pre[:, dim_index[1][0]], pic_data_pre[:, dim_index[1][1]], s=0.5, c=figcolor, cmap=cmap)
        ax_out.scatter(pic_data_out[:, dim_index[1][0]], pic_data_out[:, dim_index[1][1]], s=0.5, c=figcolor, cmap=cmap)
        if axis_visible:
            axis_visualize(ax_pre, 2)
            axis_visualize(ax_out, 2)

    for p in save_path:
        plt.savefig('%s' % p, bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()

date_str = '26-02-06'
if __name__ == '__main__':
    print('最新更改日期：%s' % date_str)
    print('作者：周琦')
    print('联系方式：2517036487@qq.com')
else:
    print('已导入模型', __name__, '最新更改日期:', date_str)

