# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:26:42 2022

@author: Kowloon Chen
"""

import copy
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import Dubins_Path
from RRTstar import RRTStar

show_animation = True

def plot_arrow(x, y, yaw, arrow_length=1.0,
               origin_point_plot_style="xr",
               head_width=0.1, fc="r", ec="k", **kwargs):
    """
    Plot an arrow or arrows based on 2D state (x, y, yaw)

    Parameters
    ----------
    x : a float or array_like
        a value or a list of arrow origin x position.
    y : a float or array_like
        a value or a list of arrow origin y position.
    yaw : a float or array_like
        a value or a list of arrow yaw angle (orientation).
    arrow_length : a float (optional)
        arrow length. default is 1.0
    origin_point_plot_style : str (optional)
        origin point plot style. If None, not plotting.
    head_width : a float (optional)
        arrow head width. default is 0.1
    fc : string (optional)
        face color
    ec : string (optional)
        edge color
    """
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, head_width=head_width,
                       fc=fc, ec=ec, **kwargs)
    else:
        plt.arrow(x, y,
                  arrow_length * math.cos(yaw),
                  arrow_length * math.sin(yaw),
                  head_width=head_width,
                  fc=fc, ec=ec,
                  **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


class RRTStarDubins(RRTStar):

    class Node(RRTStar.Node):
        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw  # 偏航
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=5,
                 max_iter=100,
                 play_area=None,
                 connect_circle_dist=50.0,
                 robot_radius=0.0,
                 ):

        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist
        self.robot_radius = robot_radius

        # for dubins path
        self.curvature = 1.0
        self.goal_yaw_th = np.deg2rad(1.0)  # 把角度转换成弧度
        self.goal_xy_th = 0.5

    def planning(self, animation=True, search_until_max_iter=True):
        start_time = time.time()  # 注意这里的时间
        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            ## 随机取点
            rnd = self.get_random_node()
            # 找到列表里邻近的点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            ## steer延申出新点
            new_node = self.steer(self.node_list[nearest_ind], rnd)
            # 判断是否在区域内和是否躲开障碍
            if self.check_if_outside_play_area(new_node, self.play_area) and \
                    self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)  # 根据新点再找方圆connect_circle_dist最相邻的点的索引near_inds
                node_with_updated_parent = self.choose_parent(new_node, near_indexes)       # 将相邻的点集和new_node相连，比较cost(这个时候比较的是从起点到这个新点的距离)，得到更新父节点之后的节点
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent,
                                near_indexes)  # 更改相邻节点的父节点，如果new_node的cost很小，而相邻节点的cost比较大，那么相邻节点的父节点就是new_node
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            print("It costs {} s, current path length: {}".format(time.time() - start_time, self.node_list[-1].cost))
            return self.generate_final_course(last_index)

        return None

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "*b")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # for (ox, oy, size) in self.obstacle_list:
        #     plt.plot(ox, oy, "ok", ms=30 * size)
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = \
            Dubins_Path.plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw,
                to_node.x, to_node.y, to_node.yaw, self.curvature)

        if len(px) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = Dubins_Path.plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        cost = sum([abs(c) for c in course_lengths])

        return from_node.cost + cost

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-math.pi, math.pi)
                            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path


def main():
    print("Start rrt star with dubins planning")

    # ====Search Path with RRT====
    # obstacleList = [
    #     (7, 4, 1),
    #     (2, 3, 2),
    #     (-1, 6, 2),
    #     (8, 10, 2),
    #     (4, 0, 2),
    #     (9, 5, 2),
    #     (5, 12, 1)
    # ]  # [x,y,size(radius)]
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x,y,size(radius)]
    # Set Initial parameters
    start = [0.0, 0.0, np.deg2rad(0.0)]
    goal = [10.0, 13.0, np.deg2rad(60)]

    rrtstar_dubins = RRTStarDubins(start, goal, rand_area=[-2.0, 15.0], obstacle_list=obstacleList)
    path = rrtstar_dubins.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        # Draw final path
        if show_animation:
            rrtstar_dubins.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.001)

            plt.show()


if __name__ == '__main__':
    main()
