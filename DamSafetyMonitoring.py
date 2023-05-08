# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import xlwings as xw
import os
import time
import datetime


class DamSafetyMonitoring:
    def __init__(self, list_PathInput, list_isHeader):
        # 读入数据
        self.path_UpLevel = list_PathInput[0]
        self.path_DownLevel = list_PathInput[1]
        self.path_AirTempt = list_PathInput[2]
        self.path_DamTempt = list_PathInput[3]
        self.path_RainFall = list_PathInput[4]
        self.path_MonitPnt = list_PathInput[5]
        self.path_WaterTempt = list_PathInput[6]
        self.data_Exist = np.zeros(7)
        print(self.data_Exist)
        if (list_PathInput[0][-4:] == ".csv"):
            if (list_isHeader[0] == 1):
                self.data_UpLevel = pd.read_csv(list_PathInput[0], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_UpLevel.index = pd.DatetimeIndex(self.data_UpLevel.index)
                self.data_UpLevel.columns = ["UpLevel"]
            else:
                self.data_UpLevel = pd.read_csv(list_PathInput[0], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_UpLevel.index = pd.DatetimeIndex(self.data_UpLevel.index)
                self.data_UpLevel.columns = ["UpLevel"]
            self.data_Exist[0] = 1
            print("上游水位为：")
            print(self.data_UpLevel)
        elif (list_PathInput[0][-4:] == ".xls" or list_PathInput[0][-5:] == ".xlsx"):
            if (list_isHeader[0] == 1):
                self.data_UpLevel = pd.read_excel(list_PathInput[0], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_UpLevel.index = pd.DatetimeIndex(self.data_UpLevel.index)
                self.data_UpLevel.columns = ["UpLevel"]
            else:
                self.data_UpLevel = pd.read_excel(list_PathInput[0], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_UpLevel.index = pd.DatetimeIndex(self.data_UpLevel.index)
                self.data_UpLevel.columns = ["UpLevel"]
            self.data_Exist[0] = 1
            print("上游水位为：")
            print(self.data_UpLevel)

        if (list_PathInput[1][-4:] == ".csv"):
            if (list_isHeader[1] == 1):
                self.data_DownLevel = pd.read_csv(list_PathInput[1], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_DownLevel.index = pd.DatetimeIndex(self.data_DownLevel.index)
                self.data_DownLevel.columns = ["DownLevel"]
            else:
                self.data_DownLevel = pd.read_csv(list_PathInput[1], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_DownLevel.index = pd.DatetimeIndex(self.data_DownLevel.index)
                self.data_DownLevel.columns = ["DownLevel"]
            self.data_Exist[1] = 1
            print("下游水位为：")
            print(self.data_DownLevel)
        elif (list_PathInput[1][-4:] == ".xls" or list_PathInput[1][-5:] == ".xlsx"):
            if (list_isHeader[1] == 1):
                self.data_DownLevel = pd.read_excel(list_PathInput[1], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_DownLevel.index = pd.DatetimeIndex(self.data_DownLevel.index)
                self.data_DownLevel.columns = ["DownLevel"]
            else:
                self.data_DownLevel = pd.read_excel(list_PathInput[1], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_DownLevel.index = pd.DatetimeIndex(self.data_DownLevel.index)
                self.data_DownLevel.columns = ["DownLevel"]
            self.data_Exist[1] = 1
            print("下游水位为：")
            print(self.data_DownLevel)

        if (list_PathInput[2][-4:] == ".csv"):
            if (list_isHeader[2] == 1):
                self.data_AirTempt = pd.read_csv(list_PathInput[2], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_AirTempt.index = pd.DatetimeIndex(self.data_AirTempt.index)
                self.data_AirTempt.columns = ["AirTempt"]
            else:
                self.data_AirTempt = pd.read_csv(list_PathInput[2], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_AirTempt.index = pd.DatetimeIndex(self.data_AirTempt.index)
                self.data_AirTempt.columns = ["AirTempt"]
            self.data_Exist[2] = 1
            print("气温为：")
            print(self.data_AirTempt)
        elif (list_PathInput[2][-4:] == ".xls" or list_PathInput[2][-5:] == ".xlsx"):
            if (list_isHeader[2] == 1):
                self.data_AirTempt = pd.read_excel(list_PathInput[2], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_AirTempt.index = pd.DatetimeIndex(self.data_AirTempt.index)
                self.data_AirTempt.columns = ["AirTempt"]
            else:
                self.data_AirTempt = pd.read_excel(list_PathInput[2], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_AirTempt.index = pd.DatetimeIndex(self.data_AirTempt.index)
                self.data_AirTempt.columns = ["AirTempt"]
            self.data_Exist[2] = 1
            print("气温为：")
            print(self.data_AirTempt)

        if (list_PathInput[3][-4:] == ".csv"):
            if (list_isHeader[3] == 1):
                self.data_DamTempt = pd.read_csv(list_PathInput[3], index_col=0, encoding="gbk")
                self.data_DamTempt.index = pd.DatetimeIndex(self.data_DamTempt.index)
            else:
                self.data_DamTempt = pd.read_csv(list_PathInput[3], header=None, index_col=0, encoding="gbk")
                self.data_DamTempt.index = pd.DatetimeIndex(self.data_DamTempt.index)
            self.data_Exist[3] = 1
            print("坝体温度为：")
            print(self.data_DamTempt)
        elif (list_PathInput[3][-4:] == ".xls" or list_PathInput[3][-5:] == ".xlsx"):
            if (list_isHeader[3] == 1):
                self.data_DamTempt = pd.read_excel(list_PathInput[3], index_col=0, encoding="gbk")
                self.data_DamTempt.index = pd.DatetimeIndex(self.data_DamTempt.index)
            else:
                self.data_DamTempt = pd.read_excel(list_PathInput[3], header=None, index_col=0, encoding="gbk")
                self.data_DamTempt.index = pd.DatetimeIndex(self.data_DamTempt.index)
            self.data_Exist[3] = 1
            print("坝体温度为：")
            print(self.data_DamTempt)

        if (list_PathInput[4][-4:] == ".csv"):
            if (list_isHeader[4] == 1):
                self.data_RainFall = pd.read_csv(list_PathInput[4], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_RainFall.index = pd.DatetimeIndex(self.data_RainFall.index)
                self.data_RainFall.columns = ["RainFall"]
            else:
                self.data_RainFall = pd.read_csv(list_PathInput[4], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_RainFall.index = pd.DatetimeIndex(self.data_RainFall.index)
                self.data_RainFall.columns = ["RainFall"]
            self.data_Exist[4] = 1
            print("降雨量为：")
            print(self.data_RainFall)
        elif (list_PathInput[4][-4:] == ".xls" or list_PathInput[4][-5:] == ".xlsx"):
            if (list_isHeader[4] == 1):
                self.data_RainFall = pd.read_excel(list_PathInput[4], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_RainFall.index = pd.DatetimeIndex(self.data_RainFall.index)
                self.data_RainFall.columns = ["RainFall"]
            else:
                self.data_RainFall = pd.read_excel(list_PathInput[4], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_RainFall.index = pd.DatetimeIndex(self.data_RainFall.index)
                self.data_RainFall.columns = ["RainFall"]
            self.data_Exist[4] = 1
            print("降雨量为：")
            print(self.data_RainFall)

        if (list_PathInput[5][-4:] == ".csv"):
            if (list_isHeader[5] == 1):
                self.data_MonitPnt = pd.read_csv(list_PathInput[5], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_MonitPnt.index = pd.DatetimeIndex(self.data_MonitPnt.index)
                name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-4]
                self.data_MonitPnt.columns = ["%s" % name_MonitPnt]
            else:
                self.data_MonitPnt = pd.read_csv(list_PathInput[5], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_MonitPnt.index = pd.DatetimeIndex(self.data_MonitPnt.index)
                name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-4]
                self.data_MonitPnt.columns = ["%s" % name_MonitPnt]
            self.data_Exist[5] = 1
            print("测点数据为：")
            print(self.data_MonitPnt)
        elif (list_PathInput[5][-4:] == ".xls" or list_PathInput[5][-5:] == ".xlsx"):
            if (list_isHeader[5] == 1):
                self.data_MonitPnt = pd.read_excel(list_PathInput[5], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_MonitPnt.index = pd.DatetimeIndex(self.data_MonitPnt.index)
                if (self.path_MonitPnt[-4:] == "xlsx"):
                    name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-5]
                elif(self.path_MonitPnt[-3:] == "xls"):
                    name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-4]
                self.data_MonitPnt.columns = ["%s" % name_MonitPnt]
            else:
                self.data_MonitPnt = pd.read_excel(list_PathInput[5], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_MonitPnt.index = pd.DatetimeIndex(self.data_MonitPnt.index)
                if (self.path_MonitPnt[-4:] == "xlsx"):
                    name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-5]
                elif(self.path_MonitPnt[-3:] == "xls"):
                    name_MonitPnt = os.path.split(self.path_MonitPnt)[1][:-4]
                self.data_MonitPnt.columns = ["%s" % name_MonitPnt]
            self.data_Exist[5] = 1
            print("测点数据为：")
            print(self.data_MonitPnt)

        if (list_PathInput[6][-4:] == ".csv"):
            if (list_isHeader[6] == 1):
                self.data_WaterTempt = pd.read_csv(list_PathInput[6], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_WaterTempt.index = pd.DatetimeIndex(self.data_WaterTempt.index)
                self.data_WaterTempt.columns = ["WaterTempt"]
            else:
                self.data_WaterTempt = pd.read_csv(list_PathInput[6], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_WaterTempt.index = pd.DatetimeIndex(self.data_WaterTempt.index)
                self.data_WaterTempt.columns = ["WaterTempt"]
            self.data_Exist[6] = 1
            print("水温为：")
            print(self.data_WaterTempt)
        elif (list_PathInput[6][-4:] == ".xls" or list_PathInput[6][-5:] == ".xlsx"):
            if (list_isHeader[6] == 1):
                self.data_WaterTempt = pd.read_excel(list_PathInput[6], index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_WaterTempt.index = pd.DatetimeIndex(self.data_WaterTempt.index)
                self.data_WaterTempt.columns = ["WaterTempt"]
            else:
                self.data_WaterTempt = pd.read_excel(list_PathInput[6], header=None, index_col=0, encoding="gbk").iloc[:, 0:1]
                self.data_WaterTempt.index = pd.DatetimeIndex(self.data_WaterTempt.index)
                self.data_WaterTempt.columns = ["WaterTempt"]
            self.data_Exist[6] = 1
            print("水温为：")
            print(self.data_WaterTempt)
        print(self.data_Exist)

# # 整合环境量
#     def data_Merge(self):
#         self.data_Merge = pd.DataFrame()


#  计算标准化法式方程中各预报因子的简单相关系数
    def get_Regre_Cof(self, X, Y):
        S_xy = np.dot(X-np.mean(X), Y-np.mean(Y))
        S_xx = np.var(X)*len(X)
        S_yy = np.var(Y)*len(Y)
        # 计算预报因子的简单相关系数r_xy
        r_xy = 1.0*S_xy/pow(S_xx*S_yy, 0.5)
        return r_xy

# 计算标准化法方程式的系数矩阵
    def get_Original_Matrix(self, data, num_Col):
        # 构造法式方程,并初始化
        r = np.ones((num_Col, num_Col))
        # 建立标准化法式方程式的系数矩阵
        for i in range(num_Col):
            for j in range(num_Col):
                r[i, j] = self.get_Regre_Cof(data[:, i], data[:, j])
        return r

# 计算偏回归平方和
    def get_Vari_Contri(self, r, num_Col):
        # 创建矩阵存储偏回归平方和，并初始化
        V = np.ones((1, num_Col-1))
        for i in range(num_Col - 1):
            V[0, i] = pow(r[i, num_Col - 1], 2)/r[i, i]
        return V

# 选择因子是否进入方程
    def select_Factor(self, r, v, k, p, num_Col, num_Row):
        # 计算引入因子的F统计量
        vari_f = (num_Row - p - 2)*v[0, k]/(r[num_Col-1, num_Col-1] - v[0, k])
        return vari_f

# 剔除因子是否留在方程
    def delete_Factor(self, r, v, k, p, num_Col, num_Row):
        # 计算剔除因子的F统计量
        vari_f = (num_Row - p - 1)*abs(v[0, k])/r[num_Col-1, num_Col-1]
        return vari_f

# 通过矩阵转换公式来计算各部分增广矩阵的元素值
    def convert_Matrix(self, r, k, num_Col):
        # 创建矩阵存储新的增广矩阵元素，并初始化
        r1 = np.ones((num_Col, num_Col))
        for i in range(num_Col):
            for j in range(num_Col):
                if (i != k and j != k):
                    r1[i, j] = r[i, j] - r[i, k]*r[k, j]/r[k, k]
                elif (i != k and j == k):
                    r1[i, j] = -r[i, k]/r[k, k]
                elif (i == k and j != k):
                    r1[i, j] = r[k, j]/r[k, k]
                else:
                    r1[i, j] = 1.0/r[k, k]
        return r1

# 获取要需剔除因子的最最小值及索引
    def get_vmin_index(self, v, judge_Factor, num_Col):
        v_min = min(v[0, np.argwhere(judge_Factor[0] == 1)])[0]
        index_v_min = np.argwhere(v[0] == v_min)[0]
        return v_min, index_v_min

# 获取要引入因子的最大值及索引
    def get_vmax_index(self, v, judge_Factor, num_Col):
        v_max = max(v[0, np.argwhere(judge_Factor[0] == 0)])[0]
        index_v_max = np.argwhere(v[0] == v_max)[0]
        return v_max, index_v_max

# 计算回归方程各预报因子的系数
    def get_Coeff(self, data, judge_Factor, r, num_Col):
        coeff_NormFactor = np.zeros((1, num_Col-1))
        coeff_OriFactor = np.zeros((1, num_Col))
        coeff_NormFactor[0][np.argwhere(judge_Factor[0] == 1)] = r[np.argwhere(judge_Factor[0] == 1), num_Col-1]
        coeff_OriFactor[0][1:] = coeff_NormFactor[0]*pow(np.var(data[:, num_Col-1])/np.var(data[:, :num_Col-1], axis=0), 0.5)
        coeff_OriFactor[0][0] = np.mean(data[:, num_Col-1]) - np.dot(coeff_OriFactor[0][1:], np.mean(data[:, :num_Col-1], axis=0))
        return coeff_NormFactor, coeff_OriFactor

# 计算复相关系数，剩余标准差， F检验， 剩余平方和
    def get_Vari_RSQF(self, data, judge_Factor, r, num_Col, num_Row):
        vari_R = pow(1-r[num_Col-1, num_Col-1], 0.5)
        vari_S = np.std(data[:, num_Col-1])*pow(num_Row, 0.5)*pow(r[num_Col-1, num_Col-1]/(num_Row - sum(judge_Factor[0] == 1) - 1), 0.5)
        vari_Q = r[num_Col-1, num_Col-1]*np.var(data[:, num_Col-1])*num_Row
        vari_F = (np.var(data[:, num_Col-1])*num_Row - vari_Q)/(sum(judge_Factor[0] == 1)*pow(vari_S, 2))
        return vari_R, vari_S, vari_Q, vari_F

# 生成预报因子
    def generate_Factor(self, judge_GenerateFactor, order_UpLevel, height_DamBase, earlier_UpLevel, order_DownLevel, earlier_DownLevel, earlier_UpliftDamBase, earlier_UpliftDam, order_DeformAdjust, order_TemptHarmWare, param_ExpAging,  param1_HyperAging, param2_HyperAging, order_AgingMulti, order_AgingHarmWare, eariler_RainFall, time_Start):
        # 判断是否存在测点数据
        self.data_Factor = pd.DataFrame(index=self.data_MonitPnt.index)
        if(time_Start != ''):
            time_Start = pd.to_datetime(time_Start)
        else:
            time_Start = self.data_Factor.index[0]
        print(time_Start)
        if(self.data_Exist[5] == 1):
            # 上游水位因子
            if (judge_GenerateFactor[0] == 1):
                if (self.data_Exist[0] == 1):
                    for i in range(1, order_UpLevel+1):
                        self.data_Factor["H_UpLevel^%i" % i] = ((self.data_UpLevel.iloc[:, 0][self.data_Factor.index] - height_DamBase)**i - (self.data_UpLevel.iloc[:, 0][time_Start] - height_DamBase)**i).values.copy()
                        print(self.data_Factor["H_UpLevel^%i" % i])
                    if (judge_GenerateFactor[1] == 1):
                        eachEariler_UpLevel = re.split(",", earlier_UpLevel)
                        num_EarilerUpLevel = len(eachEariler_UpLevel)
                        sum_EarilerUpLevel = 0
                        for i in range(1, num_EarilerUpLevel+1):
                            # try:
                            sum_EarilerUpLevel = sum_EarilerUpLevel + int(eachEariler_UpLevel[i-1])
                            arr_H_UpLevel = np.zeros((self.data_Factor.shape[0], int(eachEariler_UpLevel[i-1])))
                            arr_H0_UpLevel = np.zeros((self.data_Factor.shape[0], int(eachEariler_UpLevel[i-1])))
                            for j in range(int(eachEariler_UpLevel[i-1])):
                                arr_H_UpLevel[:, j] = self.data_UpLevel.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=sum_EarilerUpLevel-j)]
                                arr_H0_UpLevel[:, j] = self.data_UpLevel.iloc[:, 0].loc[time_Start - pd.Timedelta(days=sum_EarilerUpLevel-j)]
                            self.data_Factor["H0%i_UpLevel" % i] = np.mean(arr_H_UpLevel, axis=1) - np.mean(arr_H0_UpLevel, axis=1)
                            print(self.data_Factor["H0%i_UpLevel" % i])
                        print(eachEariler_UpLevel)
                        print("上游水位前期项项数为:%i" % num_EarilerUpLevel)
                else:
                    print("请输入上游水位")

            # 下游水位因子
            if (judge_GenerateFactor[2] == 1):
                if (self.data_Exist[1] == 1):
                    for i in range(1, order_DownLevel+1):
                        self.data_Factor["H_DownLevel^%i" % i] = ((self.data_DownLevel.iloc[:, 0][self.data_Factor.index] - height_DamBase)**i - (self.data_DownLevel.iloc[:, 0][time_Start] - height_DamBase)**i).values.copy()
                    if (judge_GenerateFactor[3] == 1):
                        eachEariler_DownLevel = re.split(",", earlier_DownLevel)
                        num_EarilerDownLevel = len(eachEariler_DownLevel)
                        sum_EarilerDownLevel = 0
                        for i in range(1, num_EarilerDownLevel+1):
                            # try:
                            sum_EarilerDownLevel = sum_EarilerDownLevel + int(eachEariler_DownLevel[i-1])
                            arr_H_DownLevel = np.zeros((self.data_Factor.shape[0], int(eachEariler_DownLevel[i-1])))
                            arr_H0_DownLevel = np.zeros((self.data_Factor.shape[0], int(eachEariler_DownLevel[i-1])))
                            for j in range(int(eachEariler_DownLevel[i-1])):
                                arr_H_DownLevel[:, j] = self.data_DownLevel.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=sum_EarilerDownLevel-j)]
                                arr_H0_DownLevel[:, j] = self.data_DownLevel.iloc[:, 0][time_Start - pd.Timedelta(days=sum_EarilerDownLevel-j)]
                            self.data_Factor["H0%i_DownLevel" % i] = np.mean(arr_H_DownLevel, axis=1) - np.mean(arr_H0_DownLevel, axis=1)
                            print(self.data_Factor["H0%i_DownLevel" % i])
                        print(eachEariler_DownLevel)
                        print("下游水位前期项项数为:%i" % num_EarilerDownLevel)
                else:
                    print("请输入下游水位")

            # 坝基扬压力项
            if (judge_GenerateFactor[4] == 1):
                if (self.data_Exist[0] == 1):
                    if (earlier_UpliftDamBase > 0):
                        arr_H_Uplift = np.zeros((self.data_Factor.shape[0], earlier_UpliftDamBase))
                        for j in range(earlier_UpliftDamBase):
                            arr_H_Uplift[:, j] = self.data_UpLevel.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=j+1)]
                        self.data_Factor["H_UpliftBase"] = self.data_UpLevel.iloc[:, 0][self.data_Factor.index].values.copy() - np.mean(arr_H_Uplift, axis=1)
                else:
                    print("请输入上游水位（库水位）")

            # 坝身扬压力项
            if (judge_GenerateFactor[5] == 1):
                if (self.data_Exist[0] == 1):
                    if (earlier_UpliftDam > 0):
                        arr_H0_Uplift = np.zeros((self.data_Factor.shape[0], earlier_UpliftDam))
                        for j in range(earlier_UpliftDam):
                            arr_H0_Uplift[:, j] = self.data_UpLevel.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=j+1)]
                        self.data_Factor["H_UpliftDam"] = pow(self.data_UpLevel.iloc[:, 0][self.data_Factor.index].values.copy() - np.mean(arr_H0_Uplift, axis=1), 2)
                else:
                    print("请输入上游水位（库水位）")

            # 拱坝坝体变形重调整项
            if (judge_GenerateFactor[6] == 1):
                if (self.data_Exist[0] == 1):
                    arr_H_DeformAdjust = np.zeros((self.data_Factor.shape[0], 30))
                    for j in range(30):
                        arr_H_DeformAdjust[:, j] = self.data_UpLevel.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=j+1)] - height_DamBase
                    for j in range(1, order_DeformAdjust+1):
                        self.data_Factor["H^%i_DeformAdjust" % j] = pow(np.mean(arr_H_DeformAdjust, axis=1), j)
                        print(self.data_Factor["H^%i_DeformAdjust" % j])
                else:
                    print("请输入上游水位（库水位）")

            # 坝体温度项使用各温度计测值作为因子
            if (judge_GenerateFactor[7] == 1):
                if (self.data_Exist[3] == 1):
                    num_Thermometer = self.data_DamTempt.shape[1]
                    for i in range(num_Thermometer):
                        self.data_Factor["T%i_Dam" % (i+1)] = self.data_DamTempt.iloc[:, i][self.data_Factor.index]
                else:
                    print("请输入坝体温度")

            # 坝体温度项使用等效温度作为因子
            if (judge_GenerateFactor[8] == 1):
                if (self.data_Exist[3] == 1):
                    num_EquTemptGradi = self.data_DamTempt.shape[1]
                    for i in range(num_EquTemptGradi):
                        self.data_Factor["T0%i_Dam" % (i+1)] = self.data_DamTempt.iloc[:, i][self.data_Factor.index]
                else:
                    print("请输入坝体温度（等效温度和梯度）")

            # 气温项

            # 水温项

            # 周期温度因子项
            if (judge_GenerateFactor[9] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                for i in range(1, order_TemptHarmWare+1):
                    self.data_Factor["T_sin%i" % (i)] = np.sin(2.0*np.pi*i*t/365) - np.sin(2.0*np.pi*i/365)
                    self.data_Factor["T_cos%i" % (i)] = np.cos(2.0*np.pi*i*t/365) - np.cos(2.0*np.pi*i/365)
                    print(self.data_Factor["T_sin%i" % i])
                    print(self.data_Factor["T_cos%i" % i])

            # 指数时效
            if (judge_GenerateFactor[10] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                self.data_Factor["Aging_Exp"] = 1 - np.exp(-param_ExpAging*(t/100.0-1/100))

            # 双曲时效
            if (judge_GenerateFactor[11] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                self.data_Factor["Aging_Hyper"] = param1_HyperAging*(t/100 - 1/100)/(param2_HyperAging + (t/100 - 1/100))

            # 多项式时效
            if (judge_GenerateFactor[12] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                for i in range(1, order_AgingMulti+1):
                    self.data_Factor["Aging%i_Multi" % i] = (t/100 - 1/100)**i
                    print(self.data_Factor["Aging%i_Multi" % i])

            # 对数时效
            if (judge_GenerateFactor[13] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                self.data_Factor["Aging_Log"] = np.log(t/100) - np.log(1/100)
                print(self.data_Factor["Aging_Log"])

            # 周期时效项
            if (judge_GenerateFactor[14] == 1):
                t = (self.data_Factor.index - time_Start).days.values.copy()
                if(time_Start == self.data_Factor.index[0]):
                    t[0] = 1
                for i in range(1, order_AgingHarmWare+1):
                    self.data_Factor["Aging_sin%i" % (i)] = np.sin(2.0*np.pi*i*t/365) - np.sin(2.0*np.pi*i/365)
                    self.data_Factor["Aging_cos%i" % (i)] = np.cos(2.0*np.pi*i*t/365) - np.cos(2.0*np.pi*i/365)
                    print(self.data_Factor["Aging_sin%i" % i])
                    print(self.data_Factor["Aging_cos%i" % i])

            # 降雨分量
            if (judge_GenerateFactor[15] == 1):
                eachEariler_RainFall = re.split(",", eariler_RainFall)
                num_EarilerRainFall = len(eachEariler_RainFall)
                if(eachEariler_RainFall[0] == "0"):
                    self.data_Factor["R_0"] = (self.data_RainFall.iloc[:, 0][self.data_Factor.index] - self.data_RainFall.iloc[:, 0][time_Start]).values.copy()
                    print(self.data_Factor["R_0"])
                    sum_EarilerRainFall = 0
                    for i in range(1, num_EarilerRainFall):
                        # try:
                        sum_EarilerRainFall = sum_EarilerRainFall + int(eachEariler_RainFall[i])
                        arr_R_RainFall = np.zeros((self.data_Factor.shape[0], int(eachEariler_RainFall[i])))
                        arr_R0_RainFall = np.zeros((self.data_Factor.shape[0], int(eachEariler_RainFall[i])))
                        for j in range(int(eachEariler_RainFall[i])):
                            arr_R_RainFall[:, j] = self.data_RainFall.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=sum_EarilerRainFall-j)]
                            arr_R0_RainFall[:, j] = self.data_RainFall.iloc[:, 0][time_Start - pd.Timedelta(days=sum_EarilerRainFall-j)]
                        self.data_Factor["R_%i" % i] = np.mean(arr_R_RainFall, axis=1) - np.mean(arr_R0_RainFall, axis=1)
                        print(self.data_Factor["R_%i" % i])
                    print(eachEariler_RainFall)
                    print("降雨量前期项项数:%i" % num_EarilerRainFall)
                else:
                    sum_EarilerRainFall = 0
                    for i in range(1, num_EarilerRainFall+1):
                        # try:
                        sum_EarilerRainFall = sum_EarilerRainFall + int(eachEariler_RainFall[i-1])
                        arr_R_RainFall = np.zeros((self.data_Factor.shape[0], int(eachEariler_RainFall[i-1])))
                        arr_R0_RainFall = np.zeros((self.data_Factor.shape[0], int(eachEariler_RainFall[i-1])))
                        for j in range(int(eachEariler_RainFall[i-1])):
                            arr_R_RainFall[:, j] = self.data_RainFall.iloc[:, 0][self.data_Factor.index - pd.Timedelta(days=sum_EarilerRainFall-j)]
                            arr_R0_RainFall[:, j] = self.data_RainFall.iloc[:, 0][time_Start - pd.Timedelta(days=sum_EarilerRainFall-j)]
                        self.data_Factor["R_%i" % i] = np.mean(arr_R_RainFall, axis=1) - np.mean(arr_R0_RainFall, axis=1)
                        print(self.data_Factor["R_%i" % i])
                    print(eachEariler_RainFall)
                    print("降雨量前期项项数:%i" % num_EarilerRainFall)

            self.data_Factor["data_Measure"] = self.data_MonitPnt.iloc[:, 0]
            print(self.data_Factor)

# 逐步回归
    def StepwiseRegression(self, F_in, F_out):
        data = self.data_Factor.values.copy()
        num_Col = data.shape[1]
        num_Row = data.shape[0]
        r = np.ones((num_Col, num_Col))
        v = np.ones((1, num_Col-1))
        judge_Factor = np.zeros((1, num_Col-1))
        sum_One = sum(judge_Factor[0] == 1)
        while(sum_One <= num_Col-1):
            if (sum_One == 0):
                r = self.get_Original_Matrix(data, num_Col)
                print(r)
                v = self.get_Vari_Contri(r, num_Col)
                print(v)
                v_max, index_v_max = self.get_vmax_index(v, judge_Factor, num_Col)
                print(v_max, index_v_max)
                vari_F_in = self.select_Factor(r, v, index_v_max, sum_One, num_Col, num_Row)
                print(vari_F_in)
                if (vari_F_in > F_in):
                    judge_Factor[0][index_v_max] = 1
                    r = self.convert_Matrix(r, index_v_max, num_Col)
                    coeff_NormFactor, coeff_OriFactor = self.get_Coeff(data, judge_Factor, r, num_Col)
                    vari_R, vari_S, vari_Q, vari_F = self.get_Vari_RSQF(data, judge_Factor, r, num_Col, num_Row)
                    sum_One += 1
                else:
                    break
                print(judge_Factor)
                print(r)
                print(coeff_NormFactor, coeff_OriFactor)
                print(vari_R, vari_S, vari_Q, vari_F)
                print(sum_One)
            elif(1 <= sum_One <= num_Col-2):
                v = self.get_Vari_Contri(r, num_Col)
                print(v)
                v_min, index_v_min = self.get_vmin_index(v, judge_Factor, num_Col)
                print(v_min, index_v_min)
                vari_F_out = self.delete_Factor(r, v, index_v_min, sum_One, num_Col, num_Row)
                print(vari_F_out)
                v_max, index_v_max = self.get_vmax_index(v, judge_Factor, num_Col)
                print(v_max, index_v_max)
                vari_F_in = self.select_Factor(r, v, index_v_max, sum_One, num_Col, num_Row)
                print(vari_F_in)
                if (vari_F_out <= F_out):
                    judge_Factor[0][index_v_min] = 0
                    r = self.convert_Matrix(r, index_v_min, num_Col)
                    coeff_NormFactor, coeff_OriFactor = self.get_Coeff(data, judge_Factor, r, num_Col)
                    vari_R, vari_S, vari_Q, vari_F = self.get_Vari_RSQF(data, judge_Factor, r, num_Col, num_Row)
                    sum_One -= 1
                    print(judge_Factor)
                    print(r)
                    print(coeff_NormFactor, coeff_OriFactor)
                    print(vari_R, vari_S, vari_Q, vari_F)
                    print(sum_One)
                else:
                    if (vari_F_in > F_out):
                        judge_Factor[0][index_v_max] = 1
                        r = self.convert_Matrix(r, index_v_max, num_Col)
                        coeff_NormFactor, coeff_OriFactor = self.get_Coeff(data, judge_Factor, r, num_Col)
                        vari_R, vari_S, vari_Q, vari_F = self.get_Vari_RSQF(data, judge_Factor, r, num_Col, num_Row)
                        sum_One += 1
                        print(judge_Factor)
                        print(r)
                        print(coeff_NormFactor, coeff_OriFactor)
                        print(vari_R, vari_S, vari_Q, vari_F)
                        print(sum_One)
                    else:
                        break
            else:
                v = self.get_Vari_Contri(r, num_Col)
                print(v)
                v_min, index_v_min = self.get_vmin_index(v, judge_Factor, num_Col)
                print(v_min, index_v_min)
                vari_F_out = self.delete_Factor(r, v, index_v_min, sum_One, num_Col, num_Row)
                print(vari_F_out)
                if (vari_F_out <= F_out):
                    judge_Factor[0][index_v_min] = 0
                    r = self.convert_Matrix(r, index_v_min, num_Col)
                    coeff_NormFactor, coeff_OriFactor = self.get_Coeff(data, judge_Factor, r, num_Col)
                    vari_R, vari_S, vari_Q, vari_F = self.get_Vari_RSQF(data, judge_Factor, r, num_Col, num_Row)
                    sum_One -= 1
                    print(judge_Factor)
                    print(r)
                    print(coeff_NormFactor, coeff_OriFactor)
                    print(vari_R, vari_S, vari_Q, vari_F)
                    print(sum_One)
                else:
                    break
        self.coeff_NormFactor = coeff_NormFactor
        self.coeff_OriFactor = coeff_OriFactor
        self.vari_R = vari_R
        self.vari_S = vari_S
        self.vari_Q = vari_Q
        self.vari_F = vari_F

# 计算各分量（水压分量，温度分量，时效，降雨分量等）
    def CalaulateComponent(self):
        name_DataFactor = self.data_Factor.columns
        indexcomp_Hydralic = []
        indexcomp_Tempterature = []
        indexcomp_Aging = []
        indexcomp_RainFall = []
        for i in range(len(name_DataFactor)):
            if (name_DataFactor[i][0] == "H"):
                indexcomp_Hydralic.append(i)
            elif(name_DataFactor[i][0] == "T"):
                indexcomp_Tempterature.append(i)
            elif(name_DataFactor[i][0] == "A"):
                indexcomp_Aging.append(i)
            elif(name_DataFactor[i][0] == "R"):
                indexcomp_RainFall.append(i)
        print(indexcomp_Hydralic)
        print(indexcomp_Tempterature)
        print(indexcomp_Aging)
        print(indexcomp_RainFall)

        self.comp_DataHydralic = self.data_Factor.iloc[:, indexcomp_Hydralic]
        self.coeff_Hydralic = self.coeff_OriFactor[0][list(map(lambda x: int(x), list(np.ones(np.array(indexcomp_Hydralic).shape)+np.array(indexcomp_Hydralic))))]
        self.comp_Hydralic = pd.DataFrame(np.dot(self.comp_DataHydralic, self.coeff_Hydralic))
        self.comp_Hydralic.index = self.data_Factor.index
        self.comp_Hydralic.columns = ["comp_Hydralic"]
        print(self.comp_Hydralic)

        self.comp_DataTempterature = self.data_Factor.iloc[:, indexcomp_Tempterature]
        self.coeff_Tempterature = self.coeff_OriFactor[0][list(map(lambda x: int(x), list(np.ones(np.array(indexcomp_Tempterature).shape)+np.array(indexcomp_Tempterature))))]
        self.comp_Tempterature = pd.DataFrame(np.dot(self.comp_DataTempterature, self.coeff_Tempterature))
        self.comp_Tempterature.index = self.data_Factor.index
        self.comp_Tempterature.columns = ["comp_Tempterature"]
        print(self.comp_Tempterature)

        self.comp_DataAging = self.data_Factor.iloc[:, indexcomp_Aging]
        self.coeff_Aging = self.coeff_OriFactor[0][list(map(lambda x: int(x), list(np.ones(np.array(indexcomp_Aging).shape)+np.array(indexcomp_Aging))))]
        self.comp_Aging = pd.DataFrame(np.dot(self.comp_DataAging, self.coeff_Aging))
        self.comp_Aging.index = self.data_Factor.index
        self.comp_Aging.columns = ["comp_Aging"]
        print(self.comp_Aging)

        self.comp_DataRainFall = self.data_Factor.iloc[:, indexcomp_RainFall]
        self.coeff_RainFall = self.coeff_OriFactor[0][list(map(lambda x: int(x), list(np.ones(np.array(indexcomp_RainFall).shape)+np.array(indexcomp_RainFall))))]
        self.comp_RainFall = pd.DataFrame(np.dot(self.comp_DataRainFall, self.coeff_RainFall))
        self.comp_RainFall.index = self.data_Factor.index
        self.comp_RainFall.columns = ["comp_RainFall"]
        print(self.comp_RainFall)

        indexcomp_DataFactor = [indexcomp_Hydralic, indexcomp_Tempterature, indexcomp_Aging, indexcomp_RainFall]
        self.comp_Data = [self.comp_Hydralic, self.comp_Tempterature, self.comp_Aging, self.comp_RainFall]

        self.comp_All = pd.DataFrame(index=self.data_Factor.index)
        for i in range(4):
            if(len(indexcomp_DataFactor[i]) > 0):
                self.comp_All = pd.concat([self.comp_All, self.comp_Data[i]], axis=1)
        self.data_Fitting = self.coeff_OriFactor[0][0] + np.sum(self.comp_All.values.copy(), axis=1)
        self.data_Residual = self.data_Factor.values.copy()[:, -1] - self.data_Fitting
        self.data_Fitting = pd.DataFrame(self.data_Fitting)
        self.data_Residual = pd.DataFrame(self.data_Residual)
        self.data_Fitting.index = self.data_Factor.index
        self.data_Residual.index = self.data_Factor.index
        self.data_Fitting.columns = ["data_Fitting"]
        self.data_Residual.columns = ["data_Residual"]
        print(self.comp_All)
        print(self.data_Fitting)
        print(self.data_Residual)

# txt输出和matplotlib出图
    def Figure_Output(self):
        self.data_Measuring = self.data_Factor.iloc[:, -1]
        fig_FittingMeasuring = plt.subplot(2, 1, 1)
        fig_Residual = plt.subplot(2, 1, 2, sharex=fig_FittingMeasuring)
        plt.show()
        # plt.plot(self.data_Factor.index, self.data_CompAllActual)
        # plt.figure(figsize=(10, 4))
        # plt.plot(self.data_Factor.index, self.data_Fitting, label="Fitting")
        # plt.plot(self.data_Factor.index, self.data_Measuring, label="Measuring")
        # plt.grid(True)
        # plt.xlabel("Time(year)", fontsize=10)
        # plt.ylabel("Displacement(mm)", fontsize=10)
        # plt.title("Fitting&Measuring", fontsize=12)
        # plt.legend()
        # # plt.plot(self.data_Factor.index, self.data_Residual)
        # plt.show()

# excel输出
    def ExcelFile_Output(self, path_Output):
        if (self.path_MonitPnt[-4:] == "xlsx"):
            filename = os.path.split(self.path_MonitPnt)[1][:-5]
        elif(self.path_MonitPnt[-3:] == "csv"):
            filename = os.path.split(self.path_MonitPnt)[1][:-4]
        elif(self.path_MonitPnt[-3:] == "xls"):
            filename = os.path.split(self.path_MonitPnt)[1][:-4]
        print(filename)

        filepath = pd.ExcelWriter(os.path.join(path_Output, r"coeff_comp_factor_%s.xlsx" % filename))
        # 输出系数
        self.coeff_Output = pd.DataFrame(self.coeff_OriFactor[0])
        index_coeff = []
        for i in range(self.coeff_Output.shape[0]):
            index_coeff.append("b%i" % i)
        self.coeff_Output.index = index_coeff
        self.coeff_Output.columns = ["Coefficient"]
        self.coeff_Output.loc["R"] = self.vari_R
        self.coeff_Output.loc["S"] = self.vari_S
        self.coeff_Output.loc["Q"] = self.vari_Q
        self.coeff_Output.loc["F"] = self.vari_F
        self.coeff_Output.to_excel(excel_writer=filepath, sheet_name="Coefficient")
        print(self.coeff_Output)

        # 输出各测点的分量
        self.comp_Output = pd.concat([self.comp_All, self.data_Fitting, self.data_Residual, self.data_MonitPnt], axis=1)
        self.comp_Output.to_excel(excel_writer=filepath, sheet_name="Comp")

        # 输出各测点因子
        self.data_Factors = self.data_Factor
        self.data_Factors.to_excel(excel_writer=filepath, sheet_name="Factor")

        # 输出环境量及各分量

        filepath.save()

        excel_State = xw.App(visible=True, add_book=False)
        filepath_Open = excel_State.books.open(os.path.join(path_Output, r"coeff_comp_factor_%s.xlsx" % filename))

        sheet_Comp = filepath_Open.sheets("Comp")
        sheet_Comp.range("A2:A%i" % sheet_Comp.used_range.last_cell.row).api.NumberFormat = "yyyy-m-d"

        sheet_Factor = filepath_Open.sheets("Factor")
        sheet_Factor.range("A2:A%i" % sheet_Comp.used_range.last_cell.row).api.NumberFormat = "yyyy-m-d"

        compFitMeasure_LineChart = sheet_Comp.charts.add(left=600, top=0, width=600, height=300)
        compFitMeasure_LineChart.set_source_data(sheet_Comp.range("A1:%s" % sheet_Comp.used_range.last_cell.get_address()))
        compFitMeasure_LineChart.chart_type = "line"
        compFitMeasure_LineChart.api[1].Legend.Position = -4160
        compFitMeasure_LineChart.api[1].PlotArea.Border.LineStyle = 1
        compFitMeasure_LineChart.api[1].Axes(2).Crosses = 4
        compFitMeasure_LineChart.api[1].Axes(1).Crosses = 4
        compFitMeasure_LineChart.api[1].Axes(2).HasMajorGridlines = True
        compFitMeasure_LineChart.api[1].Axes(1).HasMajorGridlines = True
        compFitMeasure_LineChart.api[1].Axes(2).HasMinorGridlines = False
        compFitMeasure_LineChart.api[1].Axes(1).HasMinorGridlines = False
        compFitMeasure_LineChart.api[1].Axes(2).MajorGridlines.Border.LineStyle = -4115
        compFitMeasure_LineChart.api[1].Axes(1).MajorGridlines.Border.LineStyle = -4115
        compFitMeasure_LineChart.api[1].Axes(2).MinorGridlines.Border.LineStyle = -4115
        compFitMeasure_LineChart.api[1].Axes(1).MinorGridlines.Border.LineStyle = -4115
        compFitMeasure_LineChart.api[1].Axes(2).HasTitle = True
        compFitMeasure_LineChart.api[1].Axes(1).HasTitle = True
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Text = "位移（mm）"
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Text = "时间"
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Name = "宋体"
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Name = "Times New Roman"
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Name = "宋体"
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Name = "Times New Roman"
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Size = 10
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Size = 10
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Bold = True
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Bold = True
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Italic = False
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Italic = False
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Strikethrough = False
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Strikethrough = False
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Subscript = False
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Subscript = False
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Superscript = False
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Superscript = False
        compFitMeasure_LineChart.api[1].Axes(1).AxisTitle.Font.Underline = -4142
        compFitMeasure_LineChart.api[1].Axes(2).AxisTitle.Font.Underline = -4142
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Name = "Times New Roman"
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Name = "Times New Roman"
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Size = 10
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Size = 10
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Bold = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Bold = False
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Italic = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Italic = False
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Strikethrough = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Strikethrough = False
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Subscript = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Subscript = False
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Superscript = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Superscript = False
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Font.Underline = -4142
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Font.Underline = -4142
        compFitMeasure_LineChart.api[1].Axes(2).TickLabelPosition = 4
        compFitMeasure_LineChart.api[1].Axes(1).TickLabelPosition = 4
        compFitMeasure_LineChart.api[1].Axes(1).CategoryType = 3
        compFitMeasure_LineChart.api[1].Axes(1).BaseUnit = 0
        compFitMeasure_LineChart.api[1].Axes(1).BaseUnitIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(1).MajorUnitIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(1).MajorUnit = 2
        compFitMeasure_LineChart.api[1].Axes(1).MajorUnitScale = 2
        compFitMeasure_LineChart.api[1].Axes(1).MinorUnitIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(1).MinorUnit = 1
        compFitMeasure_LineChart.api[1].Axes(1).MinorUnitScale = 2
        compFitMeasure_LineChart.api[1].Axes(2).MajorUnitIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(2).MajorUnit = 0.5
        compFitMeasure_LineChart.api[1].Axes(2).MinorUnitIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(2).MinorUnit = 0.2
        compFitMeasure_LineChart.api[1].Axes(1).MaximumScale = datetime.datetime.strptime("2005-1-1", r"%Y-%m-%d").__sub__(datetime.datetime.strptime('1899-12-30', r'%Y-%m-%d')).days
        compFitMeasure_LineChart.api[1].Axes(1).MinimumScale = datetime.datetime.strptime("2005-1-1", r"%Y-%m-%d").__sub__(datetime.datetime.strptime('1899-12-30', r'%Y-%m-%d')).days
        compFitMeasure_LineChart.api[1].Axes(1).MaximumScaleIsAuto = True
        compFitMeasure_LineChart.api[1].Axes(1).MinimumScaleIsAuto = False
        compFitMeasure_LineChart.api[1].Axes(2).MaximumScale = 0
        compFitMeasure_LineChart.api[1].Axes(2).MinimumScale = 0
        compFitMeasure_LineChart.api[1].Axes(2).MaximumScaleIsAuto = True
        compFitMeasure_LineChart.api[1].Axes(2).MinimumScaleIsAuto = True
        compFitMeasure_LineChart.api[1].Axes(1).MajorTickMark = 3
        compFitMeasure_LineChart.api[1].Axes(2).MajorTickMark = 3
        compFitMeasure_LineChart.api[1].Axes(1).MinorTickMark = -4142
        compFitMeasure_LineChart.api[1].Axes(2).MinorTickMark = -4142
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Orientation = -4105
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Orientation = -4105
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.Offset = 100
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.Offset = 100
        compFitMeasure_LineChart.api[1].Axes(1).TickLabels.NumberFormatLinked = False
        compFitMeasure_LineChart.api[1].Axes(2).TickLabels.NumberFormatLinked = False

        # comp_MarkStyle = []
        for i in range(1, sheet_Comp.used_range.last_cell.column):
            compFitMeasure_LineChart.api[1].FullSeriesCollection(i).MarkerStyle = i-1
            compFitMeasure_LineChart.api[1].FullSeriesCollection(i).MarkerSize = 2
            compFitMeasure_LineChart.api[1].FullSeriesCollection(i).Format.Line.Weight = 0.75
            compFitMeasure_LineChart.api[1].FullSeriesCollection(i).Format.Line.Visible = -1
            compFitMeasure_LineChart.api[1].FullSeriesCollection(i).Format.Line.DashStyle = 1
            # compFitMeasure_LineChart.api[1].FullSeriesCollection(i).Format.Line.ForeColor.RGB = 4145832

        # compFitMeasure_LineChart.api[1].Axes(1).NumberFormatLocal = "yyyy" 
        # compFitMeasure_LineChart.api[1].HasTitle = True
        # print(compFitMeasure_LineChart.api[1].Legend)
        # print(sheet_Comp.used_range.last_cell)

        filepath_Open.save()
        filepath_Open.close()
        excel_State.quit()


        # excel_Active = openpyxl.load_workbook(os.path.join(path_Output, r"coeff_comp_factor_%s.xlsx" % filename))

        # comp_Sheet = excel_Active.get_sheet_by_name("Comp")
        # for i in range(1, self.comp_Output.shape[0]+2):
        #     comp_Sheet["A%i" % i].number_format = openpyxl.styles.numbers.FORMAT_DATE_YYYYMMDD2
        # factor_Sheet = excel_Active.get_sheet_by_name("Factor")
        # for i in range(1, self.comp_Output.shape[0]+2):
        #     factor_Sheet["A%i" % i].number_format = openpyxl.styles.numbers.FORMAT_DATE_YYYYMMDD2

        # # 输出画图
        # comp_Sheet = excel_Active.get_sheet_by_name("Comp")

        # compFitMeasure_LineChart = openpyxl.chart.LineChart()
        # compFitMeasure_LineChart.width = 20
        # compFitMeasure_LineChart.height = 12
        # compFitMeasure_LineChart.plot_area.graphicalProperties

        # data_compFitMeasure = openpyxl.chart.Reference(comp_Sheet, min_col=2, min_row=1, max_col=self.comp_Output.shape[1]+1, max_row=self.comp_Output.shape[0]+1)
        # compFitMeasure_LineChart.add_data(data_compFitMeasure,  titles_from_data=True)
        # date_compFitMeasure = openpyxl.chart.Reference(comp_Sheet, min_col=1, min_row=2, max_row=self.comp_Output.shape[0])
        # compFitMeasure_LineChart.set_categories(date_compFitMeasure)
        # compFitMeasure_LineChart.legend.position = "t"

        # compFitMeasure_LineChart.y_axis.auto = True
        # compFitMeasure_LineChart.y_axis.title = "位移(mm)"
        # compFitMeasure_LineChart.y_axis.axPos = "l"
        # compFitMeasure_LineChart.y_axis.crosses = "min"
        # compFitMeasure_LineChart.y_axis.majorGridlines = openpyxl.chart.axis.ChartLines(openpyxl.chart.shapes.GraphicalProperties(ln=openpyxl.drawing.line.LineProperties(prstDash="dash")))
        # compFitMeasure_LineChart.y_axis.majorTickMark = "out"

        # compFitMeasure_LineChart.x_axis.auto = True
        # compFitMeasure_LineChart.x_axis.axPos = 'b'
        # compFitMeasure_LineChart.x_axis.crosses = "min"
        # compFitMeasure_LineChart.x_axis.number_format = openpyxl.styles.numbers.FORMAT_DATE_YYYYMMDD2
        # compFitMeasure_LineChart.x_axis.majorTimeUnit = "years"
        # compFitMeasure_LineChart.x_axis.title = "日期"
        # compFitMeasure_LineChart.x_axis.majorGridlines = openpyxl.chart.axis.ChartLines(openpyxl.chart.shapes.GraphicalProperties(ln=openpyxl.drawing.line.LineProperties(prstDash="dash")))
        # compFitMeasure_LineChart.x_axis.majorTickMark = "out"

        # compFitMeasure_TempChart = openpyxl.chart.LineChart()
        # compFitMeasure_TempChart.width = 20
        # compFitMeasure_TempChart.height = 12
        # compFitMeasure_TempChart.x_axis.crosses = "max"
        # compFitMeasure_TempChart.y_axis.crosses = "max"
        # compFitMeasure_TempChart.x_axis.axId = 10
        # compFitMeasure_TempChart.y_axis.axId = 20
        # compFitMeasure_TempChart.x_axis.majorGridlines = openpyxl.chart.axis.ChartLines(openpyxl.chart.shapes.GraphicalProperties(ln=openpyxl.drawing.line.LineProperties(noFill=True)))
        # compFitMeasure_TempChart.y_axis.majorGridlines = openpyxl.chart.axis.ChartLines(openpyxl.chart.shapes.GraphicalProperties(ln=openpyxl.drawing.line.LineProperties(noFill=True)))
        # # compFitMeasure_TempChart
        # # compFitMeasureTemp_Chart.add_data(data_compFitMeasure,  titles_from_data=False)
        # compFitMeasure_TempChart.set_categories(date_compFitMeasure)

        # compFitMeasure_LineChart += compFitMeasure_TempChart

        # comp_Sheet.add_chart(compFitMeasure_LineChart, "K3")
        # excel_Active.save(os.path.join(path_Output, r"coeff_comp_factor_%s.xlsx" % filename))


if __name__ == "__main__":

    path_UpLevel = ""
    path_DownLevel = ""
    path_AirTempt = ""
    path_DamTempt = ""
    path_RainFall = ""
    path_MonitPnt = ""
    path_WaterTempt = ""

    isHeader_UpLevel = 0
    isHeader_DownLevel = 0
    isHeader_AirTempt = 0
    isHeader_DamTempt = 0
    isHeader_RainFall = 0
    isHeader_MonitPnt = 0
    isHeader_WaterTempt = 0

    path_UpLevel = r"C:\Users\gonhjian\Desktop\up.xlsx"
    # path_DownLevel = r"C:\Users\gonhjian\Desktop\坝上水位.csv"
    # path_AirTempt = r"C:\Users\gonhjian\Desktop\4水库大坝历史安全监测资料及整编资料标准化处理 - 副本\1环境量监测数据\气温.csv"
    # path_RainFall = r"C:\Users\gonhjian\Desktop\4水库大坝历史安全监测资料及整编资料标准化处理 - 副本\1环境量监测数据\降雨量.csv"
    path_MonitPnt = r"C:\Users\gonhjian\Desktop\X22-1.xlsx"
    # path_WaterTempt = r"C:\Users\gonhjian\Desktop\4水库大坝历史安全监测资料及整编资料标准化处理 - 副本\1环境量监测数据\水温.csv"

    # path_UpLevel = r"C:\Users\gonhjian\Desktop\岳城.xlsx"
    isHeader_UpLevel = 1
    # isHeader_DownLevel = 1
    # isHeader_AirTempt = 1
    # isHeader_RainFall = 1
    isHeader_MonitPnt = 1
    # isHeader_WaterTempt = 1

    list_PathInput = [path_UpLevel, path_DownLevel, path_AirTempt, path_DamTempt, path_RainFall, path_MonitPnt, path_WaterTempt]
    list_isHeader = [isHeader_UpLevel, isHeader_DownLevel, isHeader_AirTempt, isHeader_DamTempt, isHeader_RainFall, isHeader_MonitPnt, isHeader_WaterTempt]

    data_All = DamSafetyMonitoring(list_PathInput, list_isHeader)

    judge_GenerateFactor = np.zeros(16)
    order_UpLevel = 1
    order_DownLevel = 1
    earlier_UpLevel = ""
    height_DamBase = 50
    earlier_DownLevel = ""
    earlier_UpliftDamBase = 3
    earlier_UpliftDam = 3
    order_DeformAdjust = 1
    order_TemptHarmWare = 2
    param_ExpAging = 2
    param1_HyperAging = 1
    param2_HyperAging = 1
    order_AgingMulti = 1
    order_AgingHarmWare = 2
    eariler_RainFall = ""
    time_Start = ""

    judge_GenerateFactor[0] = 1
    judge_GenerateFactor[1] = 0
    judge_GenerateFactor[2] = 0
    judge_GenerateFactor[3] = 0
    judge_GenerateFactor[4] = 0
    judge_GenerateFactor[5] = 0
    judge_GenerateFactor[6] = 0
    judge_GenerateFactor[9] = 1
    judge_GenerateFactor[10] = 0
    judge_GenerateFactor[11] = 0
    judge_GenerateFactor[12] = 1
    judge_GenerateFactor[13] = 1
    judge_GenerateFactor[14] = 0
    judge_GenerateFactor[15] = 0
    order_UpLevel = 4
    order_DownLevel = 1
    earlier_UpLevel = "1,1,2,11,15"
    earlier_DownLevel = "1,1,2,11,15"
    order_DeformAdjust = 3
    eariler_RainFall = "0,1,1,2,11,15"
    # time_Start = "1995-01-01"

    data_All.generate_Factor(judge_GenerateFactor, order_UpLevel, height_DamBase, earlier_UpLevel, order_DownLevel, earlier_DownLevel, earlier_UpliftDamBase, earlier_UpliftDam, order_DeformAdjust, order_TemptHarmWare, param_ExpAging, param1_HyperAging, param2_HyperAging, order_AgingMulti, order_AgingHarmWare, eariler_RainFall, time_Start)

    # print(pd.date_range(end="4/5/2018",periods=5))

    F1 = 2.01
    F2 = 2
    data_All.StepwiseRegression(F1, F2)

    data_All.CalaulateComponent()

    # data_All.Figure_Output()

    path_Output = ''

    path_Output = r"C:\Users\gonhjian\Desktop"
    data_All.ExcelFile_Output(path_Output)
