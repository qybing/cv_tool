#! python3
# _*_ coding: utf-8 _*_
# @Time : 2023/6/24 0:24 
# @Author : Jovan
# @File : counting.py
# @desc : 传入两次坐标计算进出
from collections import deque
import math


def line_angle(line):
    '''
    判断线段是水平方向的还是竖直方向的
    Args:
        line: 线段的两个端点坐标[[730, 408], [1150, 408]]
    Returns:

    '''
    x = line[1][0] - line[0][0]
    y = line[1][1] - line[0][1]
    if x != 0:
        # return math.degrees(math.atan2(y, x))
        return math.degrees(math.atan(y / x))
    else:
        return 90
    # return math.degrees(math.atan2(y, x))


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    '''
    # 判断线段AB与线段CD是否相交
    其中A,B分别是线段AB的两个端点
    Args:
        A:
        B:
        C:
        D:
    Returns:相交返回True，不相交返回False
    '''
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def calc_abc_from_line_2d(line):
    x0, y0 = line[0][0], line[0][1]
    x1, y1 = line[1][0], line[1][1]
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def get_line_cross_point(line1, line2):
    '''
    求两个相交的线段的交点坐标
    Args:
        line1: 线段AB，[[730, 408], [1150, 408]]
        line2: 线段CD [[730, 408], [1150, 408]]
    Returns:相交点的坐标
    '''
    a0, b0, c0 = calc_abc_from_line_2d(line1)
    a1, b1, c1 = calc_abc_from_line_2d(line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y


from loguru import logger


class Line:
    """
        支持360度划线设置
        input: dict; for example  line = {'line1': [[730, 408], [1150, 408]], 'line2': [[1280, 435], [1920, 889]]}
    """

    def __init__(self, line):
        self.history = {}
        self.line_angle = {}
        self.line = line
        self.already_counted = deque(maxlen=50)
        self.frame_id = 0
        self.result = {}
        self.idstate = {}
        for i in line:
            self.result[i] = {'up_count': 0, 'down_count': 0}
            temp_line = line[i]
            temp_line_angle = line_angle(temp_line)
            if temp_line_angle < 45 and temp_line_angle > -45:
                line_divide = "updown"
            else:
                line_divide = "leftright"
            self.line_angle[i] = line_divide

    def update(self, pid, cen1, cen2, frame_id):
        midpoint = [int(cen1), int(cen2)]
        if pid not in self.history:
            self.history[pid] = deque(maxlen=2)
        self.frame_id = frame_id
        self.history[pid].append(midpoint)
        previous_midpoint = self.history[pid][0]

        pid_up_count = str(pid) + "_up_count"
        pid_down_count = str(pid) + "_down_count"

        if pid not in self.already_counted:
            self.idstate[pid_up_count] = 0
            self.idstate[pid_down_count] = 0
        for i in self.line:
            temp = self.line[i]
            if intersect(midpoint, previous_midpoint, temp[0], temp[1]):
                cp_xy = get_line_cross_point(temp, [midpoint, previous_midpoint])
                if cp_xy:
                    cp_x, cp_y = cp_xy[0], cp_xy[1]
                    if self.line_angle[i] == "updown":
                        if cp_y < midpoint[1] and self.idstate[pid_down_count] != 1:
                            self.result[i]['down_count'] = self.result[i]['down_count'] + 1
                            self.idstate[pid_down_count] = 1

                        elif cp_y > midpoint[1] and self.idstate[pid_up_count] != 1:
                            self.result[i]['up_count'] = self.result[i]['up_count'] + 1
                            self.idstate[pid_up_count] = 1

                        else:
                            if midpoint[1] < previous_midpoint[1] and self.idstate[pid_up_count] != 1:
                                self.result[i]['up_count'] = self.result[i]['up_count'] + 1
                                self.idstate[pid_up_count] = 1

                            elif midpoint[1] >= previous_midpoint[1] and self.idstate[pid_down_count] != 1:
                                self.result[i]['down_count'] = self.result[i]['down_count'] + 1
                                self.idstate[pid_down_count] = 1
                    else:

                        if cp_x < midpoint[0] and self.idstate[pid_down_count] != 1:
                            self.result[i]['down_count'] = self.result[i]['down_count'] + 1
                            self.idstate[pid_down_count] = 1

                        elif cp_x > midpoint[0] and self.idstate[pid_up_count] != 1:
                            self.result[i]['up_count'] = self.result[i]['up_count'] + 1
                            self.idstate[pid_up_count] = 1
                        else:
                            if midpoint[0] < previous_midpoint[0] and self.idstate[pid_up_count] != 1:
                                self.result[i]['up_count'] = self.result[i]['up_count'] + 1
                                self.idstate[pid_up_count] = 1

                            elif midpoint[0] >= previous_midpoint[0] and self.idstate[pid_down_count] != 1:
                                self.result[i]['down_count'] = self.result[i]['down_count'] + 1
                                self.idstate[pid_down_count] = 1

                self.already_counted.append(pid)

    def getdata(self):
        return self.result


if __name__ == '__main__':
    # line1 = {'line1': [[730, 408], [1150, 408]], 'line2': [[1280, 435], [1920, 889]]}
    line1 = {'line1': [[730, 408], [1150, 408]]}
    class_line = Line(line1)
    class_line.update(1, 953, 252, 1)
    class_line.update(1, 950, 607, 2)
    class_line.update(1, 950, 610, 3)
    class_line.update(2, 953, 252, 1)
    class_line.update(2, 950, 607, 2)
    # class_line.update(2, 950, 607, 2)
    # class_line.update(2, 953, 252, 1)
    result = class_line.getdata()
    print(result)
    # class_line = Line(line1)
    # class_line.update(1, 1080, 310, 3)
    # result = class_line.getdata()
    # print(result)
