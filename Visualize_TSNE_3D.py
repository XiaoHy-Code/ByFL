import math
import pandas as pd
import numpy as np
import pyecharts.options as opts
import pyecharts.charts as pych
from pyecharts.faker import Faker
import pathlib


folder_str = r'G:\FL\By-FL\logs\2022-06-09\16.42.21\visualize_TSNE_3D'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

createVars = locals()#以字典类型返回当前位置所有局部变量，后续DataFrame
for fp in folder.iterdir():#迭代文件夹
    if fp.match('*.csv'):#re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]
        features = pd.read_csv(fp, header=None)
        features = np.array(features)

        data = []
        for t in features:
            data.append(t.tolist())

        xaxis3d_opts = opts.Axis3DOpts(type_="value")
        yaxis3d_opts = opts.Axis3DOpts(type_="value")
        grid3d_opts = opts.Grid3DOpts(width=100, height=100, depth=100)

        scatter = pych.Scatter3D().add("", data=data, xaxis3d_opts=xaxis3d_opts, yaxis3d_opts=yaxis3d_opts,
                                        grid3d_opts=grid3d_opts)
        scatter_opts = scatter.set_global_opts(
            title_opts=opts.TitleOpts(title="Scatter-多维度数据"),
            visualmap_opts=opts.VisualMapOpts(dimension=3, max_=50, min_=0),
            toolbox_opts=opts.ToolboxOpts(is_show=True)).set_series_opts(

            markpoint_opts=opts.MarkPointOpts(symbol='triangle')
        )

        scatter_opts.render(folder_str+'/'+varname+'.html')


# features = pd.read_csv(r'G:\FL\By-FL\logs\2022-06-01\17.35.10\visualize_TSNE\TSNE_round95.csv')
# features = np.array(features)
#
# data = []
# for t in features:
#     data.append(t.tolist())
#
# xaxis3d_opts = opts.Axis3DOpts(Faker.clock, type_="value")
# yaxis3d_opts = opts.Axis3DOpts(Faker.week_en, type_="value")
# grid3d_opts = opts.Grid3DOpts(width=100, height=100, depth=100)
#
# scatter = pych.Scatter3D().add("", data=data, xaxis3d_opts=xaxis3d_opts, yaxis3d_opts=yaxis3d_opts, grid3d_opts=grid3d_opts)
# scatter_opts = scatter.set_global_opts(visualmap_opts=opts.VisualMapOpts(dimension=2, max_=30, min_=0))
# scatter_opts.render("line3d_rectangular_projection.html")


