# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:32:53 2022

@author: Kowloon Chen
"""

import math
import time
import matplotlib.pyplot as plt
from RRT import RRT

show_animation = True


class RRTStar(RRT):

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,  # 每次衍生的步长
                 path_resolution=0.1,  # 路径分解的长度
                 goal_sample_rate=5,  # 随机点出现在目标点的概率
                 max_iter=300,
                 connect_circle_dist=50.0,
                 play_area=None,  ## 给定的机器人运行场地
                 search_until_max_iter=True,  # RRT*是否需要搜到最后？
                 robot_radius=0.0):  # 把机器人当作一个圆，他有一个半径，在接触障碍物时，需要判断

        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius)

        self.connect_circle_dist = connect_circle_dist  # 这是新参数connect_circle_dist，为了找到新点产生后方圆dist内的相邻点
        self.goal_node = self.Node(goal[0], goal[1])    # 这是rrt中的end
        self.search_until_max_iter = search_until_max_iter  # 这也是新出现的参数
        self.node_list = []
        if play_area is not None:
            self.play_area = RRT.AreaBounds(play_area)
        else:
            self.play_area = None

    def planning(self, animation=True):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        start_time = time.time()  # 注意这里的时间
        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))  # 输出迭代次数和结点list
            rnd_node = self.get_random_node()  # 随机取点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # 找到距离随机点最近的相邻点的索引ind
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)  # 连接相邻的点和随机点，注意expand_distance
            new_node.cost = nearest_node.cost + \
                math.hypot(new_node.x-nearest_node.x,
                           new_node.y-nearest_node.y)          # 计算最近点和起始点的cost距离+新点和最近点的距离，返回欧几里得范数

            if self.check_if_outside_play_area(new_node, self.play_area) and \
                    self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)   # 根据新点再找方圆connect_circle_dist最相邻的点的索引near_inds
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)             # 将相邻的点集和new_node相连，比较cost(这个时候比较的是从起点到这个新点的距离)，得到更新父节点之后的节点
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds) # 更改相邻节点的父节点，如果new_node的cost很小，而相邻节点的cost比较大，那么相邻节点的父节点就是new_node
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd_node)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                next_index = self.search_best_goal_node()
                if next_index is not None:
                    return self.generate_final_course(next_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            print("It costs {} s, current path length: {}".format(time.time() - start_time, self.node_list[-1].cost))
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]    # [x,y,size(radius)]

    # Set Initial parameters
    rrt_star = RRTStar(
        start=[0, 0],
        goal=[5, 12],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        #play_area=[0, 10, 0, 14],
        expand_dis=3,
        max_iter=200,
        robot_radius=0.0)
    path = rrt_star.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            plt.grid(True)
            plt.pause(0.001)

            plt.show()


if __name__ == '__main__':
    main()
