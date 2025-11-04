import numpy as np
import re
import random
import os
from collections import deque, namedtuple, defaultdict
import math
from math import sqrt, atan2
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time

# ====================== 数据结构定义 ======================
Customer = namedtuple('Customer', ['id', 'type', 'x', 'y', 'demand', 
                                   'ready', 'due', 'service'])
Station = namedtuple('Station', Customer._fields)
Depot = namedtuple('Depot', ['id', 'type', 'x', 'y', 'demand', 'ready', 'due', 'service'])

class EVRPTWInstance:
    def __init__(self):
        self.depot = None
        self.customers = []
        self.stations = []
        self.vehicle_params = {
            'fuel_cap': None,
            'load_cap': None,
            'consump_rate': None,
            'charge_rate': None,
            'velocity': None
        }
        self.dist_matrix = None

class Route:
    def __init__(self, nodes = []):
        self.nodes = nodes  # 节点序列 (depot起始和结束)
        self.load = 0
        self.time = 0
        self.fuel = 0

# ====================== 数据加载模块 ======================
def load_instance(file_path):
    """加载Solomon格式实例"""
    instance = EVRPTWInstance()
    param_pattern = re.compile(r"([A-Za-z]+)\s+.*?/(\d+\.?\d*)/")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("StringID") or line.startswith('S0'):
                continue
            
            # 解析参数行
            if line[:2] in ['Q ','C ','r ','g ','v ']:
                match = param_pattern.search(line)
                if match:
                    param, value = match.groups()
                    map_param(instance, param, float(value))
                continue
                
            # 解析数据行
            parts = line.split()
            if len(parts) < 8: continue
            
            node_id = parts[0]
            node_type = parts[1]
            x, y = float(parts[2]), float(parts[3])
            demand = float(parts[4])
            ready, due, service = map(float, parts[5:8])

            if node_type == 'd':
                instance.depot = Depot(node_id, node_type, x, y, demand, ready, due, service)
            elif node_type == 'f':
                instance.stations.append(Station(node_id, node_type, x, y, demand, ready, due, service))
            elif node_type == 'c':
                instance.customers.append(Customer(node_id, node_type, x, y, demand, ready, due, service))

    # 预计算距离矩阵
    build_distance_matrix(instance)
    return instance

def map_param(instance, param, value):
    """映射参数到实例"""
    param_map = {
        'Q': ('fuel_cap', value),
        'C': ('load_cap', value),
        'r': ('consump_rate', value),
        'g': ('charge_rate', 1/value),  # 转换为充电速率
        'v': ('velocity', value)
    }
    if param in param_map:
        key, val = param_map[param]
        instance.vehicle_params[key] = val

def build_distance_matrix(instance):
    nodes = np.array([(n.x, n.y) for n in [instance.depot] + instance.stations + instance.customers])
    dx = nodes[:, 0, None] - nodes[:, 0]
    dy = nodes[:, 1, None] - nodes[:, 1]
    instance.dist_matrix = np.sqrt(dx**2 + dy**2)

# ====================== 算法核心实现 ======================
import random
import math
import copy
import itertools
from collections import deque

class VNSTSolver:
    def __init__(self, instance, predefine_route_number=3):
        self.instance = instance
        self.tabu_list = deque(maxlen=30)
        self.recharging_stations = np.array([[instance.stations[i].x, instance.stations[i].y] for i in range(len(instance.stations))])

        # 退火参数
        self.temp = -1  # 初始温度

        # 禁忌搜索参数
        self.tabu_tenure = 30  # 禁忌表的最大长度
        self.tabu_iter = 100 # 禁忌搜索迭代次数

        # 论文设定的惩罚参数
        self.alpha, self.beta, self.gamma = 10.0, 10.0, 10.0
        self.alpha_min, self.beta_min, self.gamma_min = 0.5, 0.75, 1.0
        self.alpha_max, self.beta_max, self.gamma_max = 5000, 5000, 5000

        # VNS参数
        self.k_max = 15  # 最大邻域数
        self.η_feas = 500  # 非可行性阶段的最大尝试次数
        self.η_dist = 200  # 可行阶段的最大尝试次数
        self.nearest_station = self.battery_to_nearest_rs(instance.depot)
        self.instance_dist_matrix_calculatrion()
        self.attribute_frequency = defaultdict(int)
        self.attribute_total = 0
        self.lambda_div = 1.0
        self.delta_sa = 0.08
        self.predefine_route_number = predefine_route_number
        self.global_value = 1e10
        self.time_matrix = self.dist_matrix / instance.vehicle_params['velocity']

    def time_cost(self, node1, node2):
        i = self.node_id[node1.id]
        j = self.node_id[node2.id]
        return self.time_matrix[i][j]

    def instance_dist_matrix_calculatrion(self):
        self.node_id = {self.instance.depot.id: 0}
        for i in range(len(self.instance.stations)):
            self.node_id[self.instance.stations[i].id] = i + 1
        offset = i + 1
        for i in range(len(self.instance.customers)):
            self.node_id[self.instance.customers[i].id] = i + 1 + offset
        self.dist_matrix = self.instance.dist_matrix

    def battery_to_nearest_rs(self, node):
        self.nearest_station = {self.instance.depot.id: 0}
        for i in range(len(self.instance.stations)):
            self.nearest_station[self.instance.stations[i].id] = 0

        for i in range(len(self.instance.customers)):
            pos = np.array([self.instance.customers[i].x, self.instance.customers[i].y])
            distances = np.linalg.norm(self.recharging_stations - pos, axis=1)
            nearest_station = self.instance.stations[np.argmin(distances)]
            self.nearest_station[self.instance.customers[i].id] = self.instance.vehicle_params['consump_rate'] * np.min(distances) / self.instance.vehicle_params['velocity']

    def fuel_consumption(self, node1, node2):
        return self.instance.vehicle_params['consump_rate'] * self.time_cost(node1,node2)

    def solve(self):
        """基于论文 Figure 1 的 VNS/TS 解决方案"""
        S = self.initial_solution()
        self.global_solution = S[:]
        κ = 1  # 初始邻域编号
        i = 0  # 迭代次数

        feasibilityPhase = True  # 初始进入可行性阶段
        best_solution = copy.deepcopy(S)
        best_value = 1e10
        while feasibilityPhase or (not feasibilityPhase and i < self.η_dist):
            start_time = time.time()
            # print("Step {}, Neighbor Idx: {}".format(i,κ))
            S_prime = self.vns_perturb(S, κ)
            S_double_prime = self.apply_tabu_search(S_prime)
            if self.accept_sa(S_double_prime, S):
                S = S_double_prime[:]
                κ = 1
            else:
                κ = (κ % self.k_max) + 1

            S_value = self.generalized_cost(S, penalty_value=False, p_div_value=False, allow_infeasible=False)
            if self.is_solution_feasible(S) and S_value < best_value:
                best_solution = copy.deepcopy(S)
                best_value = S_value
                if self.global_value > best_value:
                    self.global_value = best_value
                    self.global_solution = best_solution

            if feasibilityPhase:
                if not self.is_solution_feasible(S):
                    if i == self.η_feas:
                        S = self.add_vehicle(S)
                        i -= 1
                else:
                    feasibilityPhase = False
                    i = 0 # 进入非可行阶段
            i += 1
            self.update_penalty_weights(S, i)
        print(time.time() - start_time)
        return self.global_solution 

    def apply_tabu_search(self, S_prime):
        """对 S' 进行禁忌搜索优化"""
        return self._tabu_search(S_prime)

    def accept_sa(self, S_double_prime, S):
        """使用模拟退火准则接受新的解"""
        cost_diff = self.generalized_cost(S_double_prime, penalty_value=False, p_div_value=False, allow_infeasible=False) - self.generalized_cost(S, penalty_value=False, p_div_value=False, allow_infeasible=False)

        if cost_diff <= 0:
            return True  # 直接接受更优解

        if self.temp == -1 and cost_diff > 0:
            # 论文方法：首次遇到劣解时初始化温度
            self.temp = -cost_diff / math.log(0.5)
            self.cooling = (1 - self.delta_sa)
        else:
            # 线性降温
            self.temp *= self.cooling  

        # SA 接受准则
        return random.random() < math.exp(-cost_diff / self.temp)

    def add_vehicle(self, S):
        """添加车辆"""
        new_route = []
        candidate_customer = []
        for route in S:
            if self.is_route_feasible(route):
                new_route.append(route)
                continue

            route_add = self.copy_route(route)
            route_length = len(route_add.nodes)
            search_idx = 1

            while search_idx < route_length - 1 and not self.is_route_feasible(route_add):
                current_node = route_add.nodes[search_idx]
                if self.violates_constraints(route_add, search_idx):
                    if current_node.type == 'c':
                        candidate_customer.append(current_node)
                        route_add.nodes.pop(search_idx)
                        route_length -= 1
                    else:
                        route_add.nodes.pop(search_idx)
                else:                    
                    search_idx += 1

            if len(route_add.nodes) > 2:
                new_route.append(route_add)
            
        route_add = self.create_new_route()
        route_add.nodes.append(self.instance.depot)
        candidate_routes = []

        while len(candidate_customer) > 0:
            current_node = candidate_customer.pop()
            
            if not candidate_routes:
                route_add.nodes.insert(-1, current_node)
                candidate_routes.append(route_add)
            else:
                insert_success = False
                for route in candidate_routes:
                    for insert_pos in reversed(range(1, len(route.nodes) - 1)):
                        route.nodes.insert(insert_pos, current_node)
                        if self.is_route_feasible(route):
                            insert_success = True
                            break
                        else:
                            route.nodes.pop(insert_pos)  # 直接撤回插入
                    if insert_success:
                        break
                
                if not insert_success:
                    route_add = self.create_new_route()
                    route_add.nodes.append(current_node)
                    route_add.nodes.append(self.instance.depot)
                    candidate_routes.append(route_add)
        
        new_route.extend(candidate_routes)
        return new_route

    def violates_constraints(self, route, search_idx):
        """检查删除 search_idx 位置的节点后，是否仍然违反约束"""
        new_route = self.copy_route(route)
        new_route.nodes = new_route.nodes[:search_idx + 1]
        if new_route.nodes[-1].type != 'd':
            new_route.nodes.append(self.instance.depot)
        return self.battery_violation(new_route) or self.time_violation(new_route) or self.load_violation(new_route)

    def battery_violation(self, route, node=None):
        """检查客户是否导致路径电量违规"""
        current_fuel = self.instance.vehicle_params['fuel_cap']

        for i in range(len(route.nodes) - 1):
            from_node = route.nodes[i]
            to_node = route.nodes[i + 1]
            fuel_needed = self.fuel_consumption(from_node, to_node)

            current_fuel -= fuel_needed
            if current_fuel < 0:
                return True  # 电量不足

            # 经过充电站时，补充电量
            if to_node.type == 'f':
                current_fuel = self.instance.vehicle_params['fuel_cap']

        # 访问客户节点
        if node is not None:
            fuel_needed = self.fuel_consumption(route.nodes[-1], node)
            current_fuel -= fuel_needed
            if current_fuel < 0:
                return True

        return False

    def battery_penalty(self, route):
        current_fuel = self.instance.vehicle_params['fuel_cap']
        battery_penalty_value = 0
        gamma_in = 0
        for i in range(len(route.nodes) - 1):
            from_node = route.nodes[i]
            to_node = route.nodes[i + 1]
            fuel_needed = self.fuel_consumption(from_node, to_node)

            gamma_in += fuel_needed
            battery_penalty_value += max(0, gamma_in - self.instance.vehicle_params['fuel_cap'])
                
            # 经过充电站时，补充电量
            if to_node.type == 'f':
                gamma_in = 0
        return battery_penalty_value

    def load_violation(self, route, node=None):
        """检查路径负载约束"""
        if node is not None:
            return sum(node.demand for node in route.nodes) + node.demand > self.instance.vehicle_params['load_cap']
        return sum(node.demand for node in route.nodes) > self.instance.vehicle_params['load_cap']
    
    def load_penalty(self, route):
        return max(0, sum(node.demand for node in route.nodes) - self.instance.vehicle_params['load_cap'])

    def time_violation(self, route, node=None):
        """检查客户是否导致路径时间窗违规"""
        current_time = 0
        battery_use = 0
        vehicle_params = self.instance.vehicle_params
        charge_rate = vehicle_params['charge_rate']
        
        for i in range(len(route.nodes)):
            node_i = route.nodes[i]
            
            # 计算到达时间
            arrival_time = max(current_time, node_i.ready)  # 不能早于时间窗开始
            
            # 检查时间窗是否被违反
            if arrival_time > node_i.due:
                return True  # 违反时间窗
            
            # 更新电池消耗或充电
            if node_i.type == "c":  # 客户点
                if i < len(route.nodes) - 1:
                    battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[route.nodes[i-1].id]]
                current_time = arrival_time + node_i.service  # 增加服务时间

            elif node_i.type == "f":  # 充电站
                battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[route.nodes[i-1].id]]
                current_time += battery_use / charge_rate
                battery_use = 0  # 充电后电池归零

            # 更新行驶时间
            if i < len(route.nodes) - 1:
                current_time += self.time_cost(node_i, route.nodes[i + 1])

        # 如果有新加入的节点 `node`，检查它是否导致时间窗违规
        if node is not None:
            last_node = route.nodes[-1] if route.nodes else None
            if last_node:
                projected_arrival = current_time + self.time_cost(last_node, node)
                if projected_arrival > node.due:
                    return True  # 违反时间窗

        return False

    def time_penalty(self, route):
        """计算时间窗超出部分的惩罚，同时考虑充电站的充电时间"""
        time_penalty_value = 0
        current_time = 0
        battery_use = 0
        vehicle_params = self.instance.vehicle_params
        charge_rate = vehicle_params['charge_rate']

        for i in range(len(route.nodes)):
            node_i = route.nodes[i]

            # 计算到达时间，不能早于时间窗开始
            arrival_time = max(current_time, node_i.ready)

            # 计算并累计时间窗惩罚
            if arrival_time > node_i.due:
                time_penalty_value += max(0, arrival_time - node_i.due)  # 超出部分惩罚
                arrival_time = node_i.due  # 假设在截止时间到达

            # 处理客户节点
            if node_i.type == "c":  
                current_time = arrival_time + node_i.service  # 增加服务时间
                if i < len(route.nodes) - 1:
                    next_node = route.nodes[i + 1]
                    battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[next_node.id]]

            # 处理充电站节点
            elif node_i.type == "f":
                if battery_use > 0:  # 只有电池消耗时才充电
                    charge_time = battery_use / charge_rate
                    current_time += charge_time  # 增加充电时间
                    battery_use = 0  # 充满后电池归零

            # 计算到下一个节点的行驶时间
            if i < len(route.nodes) - 1:
                current_time += self.time_cost(node_i, route.nodes[i + 1])

        return time_penalty_value

    def update_penalty_weights(self, solution, step):
        """根据论文规则动态更新 α, β, γ"""
        delta = 1.2  # 论文实验设定的增长因子
        penalty_update_interval = 2  # 论文设定 τ_penalty = 2

        # 计算当前解的违规情况
        load_violation = sum(self.load_penalty(route) for route in solution)
        tw_violation = sum(self.time_penalty(route) for route in solution)
        batt_violation = sum(self.battery_penalty(route) for route in solution)

        # **检查是否有违规**
        self.load_update = load_violation > 0
        self.tw_update = tw_violation > 0
        self.batt_update = batt_violation > 0

        # **仅在指定步数进行更新**
        if step % penalty_update_interval == 0:
            # **更新 α（容量违规惩罚因子）**
            if self.load_update:
                self.alpha = min(self.alpha * delta, self.alpha_max)  # 违反则增加
            else:
                self.alpha = max(self.alpha / delta, self.alpha_min)  # 满足则减少
            
            # **更新 β（时间窗口违规惩罚因子）**
            if self.tw_update:
                self.beta = min(self.beta * delta, self.beta_max)
            else:
                self.beta = max(self.beta / delta, self.beta_min)

            # **更新 γ（电池违规惩罚因子）**
            if self.batt_update:
                self.gamma = min(self.gamma * delta, self.gamma_max)
            else:
                self.gamma = max(self.gamma / delta, self.gamma_min)

            self.update_reset()

    def update_reset(self):
        self.load_update = False
        self.batt_update = False
        self.tw_update = False

    def initial_solution(self):
        """基于 Schneider et al. (2014) 方法的初始解（极角排序 + 贪心插入 + 批量时间窗排序）"""

        depot = self.instance.depot
        customers = self.instance.customers

        # 选择一个随机点，用于计算极角排序
        random_point = random.choice(customers)
        customers_sorted = sorted(customers, key=lambda c: self.polar_angle(c, depot, random_point))

        # 预定义路径数量（从最佳已知解中获取）
        predefined_routes = self.predefine_route_number

        routes = []
        current_route = self.create_new_route()
        current_route.nodes.append(depot)

        last_route = self.create_new_route()
        unassigned_customers = []  # 存放无法插入 `predefined_routes` 的客户

        for customer in customers_sorted:
            best_position = None
            min_extra_cost = float('inf')

            # 在当前路径中找到最优插入位置（最小增量）
            for i in range(1, len(current_route.nodes)):  # 不能插入到起点
                temp_route = self.copy_route(current_route)
                temp_route.nodes.insert(i, customer)

                if not self.load_violation(temp_route, customer) and not self.time_violation(temp_route, customer):
                    extra_cost = self.generalized_cost([temp_route], penalty_value=False, p_div_value=False, allow_infeasible=True)
                    if extra_cost < min_extra_cost:
                        min_extra_cost = extra_cost
                        best_position = i

            # 插入最优位置，如果找不到合适位置，开启新路径
            if best_position is not None:
                current_route.nodes.insert(best_position, customer)
            else:
                if len(routes) <= predefined_routes:
                    # 关闭当前路径并开启新路径
                    routes.append(current_route)

                    current_route = self.create_new_route()
                    current_route.nodes.append(customer)
                    current_route.nodes.append(depot)  # 新路径的起点
                else:
                    # 超过路径数限制，存入 `unassigned_customers`
                    unassigned_customers.append(customer)

        routes.append(current_route)
        # **批量插入 `last_route`，按时间窗排序**
        if len(unassigned_customers) > 0:
            unassigned_customers.sort(key=lambda c: c.ready)  # 时间窗排序
            last_route.nodes.extend(unassigned_customers)  # 一次性插入
            last_route.nodes.append(depot)
            routes.append(last_route)  # 加入最终解

        while len(routes) < predefined_routes:
            current_route = self.create_new_route()
            current_route.nodes.append(depot)  # 新路径的起点
            routes.append(current_route)

        return routes

    def vns_perturb(self, solution, k):
        """VNS 变邻域扰动（符合论文 Table 2 方法）"""
        # Table 2 中的邻域结构参数
        neighborhood_structure = {
            1: (2, 1),  2: (2, 2),  3: (2, 3),  4: (2, 4),  5: (2, 5),
            6: (3, 1),  7: (3, 2),  8: (3, 3),  9: (3, 4), 10: (3, 5),
            11: (4, 1), 12: (4, 2), 13: (4, 3), 14: (4, 4), 15: (4, 5)
        }

        if k not in neighborhood_structure:
            return solution  # 如果 k 超出范围，则不变
        if len(solution) == 1:
            if random.random() < 0.3:
                return self.extra_exchange(solution)
            return solution

        if len(solution) < neighborhood_structure[k][0]:
            return solution

        num_routes, max_nodes = neighborhood_structure[k]

        return self.cyclic_exchange(solution, num_routes, max_nodes)

    def cyclic_exchange(self, solution, num_routes, max_nodes):
        """Cyclic-Exchange: 在 num_routes 条路径之间进行循环交换子路径 (符合论文 κ 邻域结构)"""

        if len(solution) < num_routes:
            return solution  # 需要至少 num_routes 条路径

        # **优化 1: 只拷贝被修改的部分**
        selected_routes_idx = random.sample(range(len(solution)), num_routes)
        new_solution = [solution[i] for i in range(len(solution))]  # 复制引用，减少拷贝

        selected_routes = [solution[i] for i in selected_routes_idx]  # 选择 num_routes 条路径
        segments, start_positions, end_positions = [], [], []

        # **优化 2: 避免多次访问 .nodes 及随机数计算**
        for route in selected_routes:
            nodes = route.nodes
            num_nodes = len(nodes)
            if num_nodes < 3:  # 必须至少有 depot + 一个客户 + depot
                return solution  

            start = random.randint(1, num_nodes - 2)  # 确保至少有一个客户
            max_chain_length = min(max_nodes, num_nodes - 2)  
            chain_length = random.randint(0, max_chain_length)
            end = min(start + chain_length, num_nodes - 1)

            segments.append(nodes[start:end])
            start_positions.append(start)
            end_positions.append(end)

        # **优化 3: 避免不必要的列表拼接**
        for i in range(num_routes):
            next_i = (i + 1) % num_routes
            route = new_solution[selected_routes_idx[next_i]]
            route.nodes[start_positions[next_i]:end_positions[next_i]] = segments[i]

        return new_solution

    def extra_exchange(self, solution):
        node_idx = random.randint(1, len(solution[0].nodes) - 1)
        while solution[0].nodes[node_idx].type != "c":
            node_idx = random.randint(1, len(solution[0].nodes) - 1)
        node = solution[0].nodes.pop(node_idx)
        route_add = self.create_new_route()
        route_add.nodes.append(node)
        route_add.nodes.append(self.instance.depot)
        solution.append(route_add)
        return solution

    def _tabu_search(self, S):
        """禁忌搜索"""
        best_solution = copy.deepcopy(S)
        current_solution = copy.deepcopy(S)
        tabu_list = deque(maxlen=self.tabu_tenure)
        for iter in range(self.tabu_iter):
            # 生成候选解
            self.route_info = [self.print_route(r) for r in current_solution]
            two_opt_start, two_opt_route_info = self._two_opt(current_solution)
            relocate_start, relocate_route_info = self._relocate(current_solution)
            exchange_start, exchange_route_info = self._exchange(current_solution)
            station_in_re_start, station_in_re_route_info = self._station_insertion(current_solution)

            zip_neighborhood = [two_opt_start, relocate_start, exchange_start, station_in_re_start]
            neighborhood = [item for sublist in zip_neighborhood for item in sublist]
            zip_infos = [two_opt_route_info, relocate_route_info, exchange_route_info, station_in_re_route_info]
            tabu_infos = [item for sublist in zip_infos for item in sublist]

            # 选择最优候选解
            current_candidate = min(neighborhood, key=self.generalized_cost)
            current_candidate_info = tabu_infos[neighborhood.index(current_candidate)]
            current_candidate = self.solution_fix(current_candidate)
            best_solution = min(
                                neighborhood, 
                                key=lambda sol: self.generalized_cost(
                                    sol, penalty_value=False, p_div_value=False, allow_infeasible=False
                                )
                            )
            best_solution_value = self.generalized_cost(best_solution, penalty_value=False, p_div_value=False, allow_infeasible=False)
            best_solution = self.solution_fix(best_solution)

            if best_solution_value < self.global_value:
                global_value = best_solution_value
                self.golbal_solution = best_solution
            
            if current_candidate_info not in tabu_list:
                depot_to_depot = [r for r in current_candidate if len(r.nodes) == 2]
                if len(depot_to_depot) > 1:
                    regular_routes = [r for r in current_candidate if len(r.nodes) > 2]
                    current_candidate = regular_routes.extend(depot_to_depot[0])
                
                current_solution = current_candidate
                tabu_list.append(current_candidate_info)

                # 更新历史路径结构频率
                self.update_diversification_history(current_solution)

                # 更新最优解
                if self.generalized_cost(current_solution, penalty_value=False, p_div_value=False, allow_infeasible=False) < self.generalized_cost(best_solution, penalty_value=False, p_div_value=False, allow_infeasible=False):
                    best_solution = copy.deepcopy(current_solution)
        test_end = time.time()
        return best_solution

    def update_diversification_history(self, S):
        """更新路径结构的历史出现频率"""
        for k, route in enumerate(S):
            for i in range(1, len(route.nodes) - 1):  # 遍历所有客户
                u = route.nodes[i].id
                mu = route.nodes[i - 1].id
                zeta = route.nodes[i + 1].id
                self.attribute_frequency[(u, k, mu, zeta)] += 1  # 增加出现次数
                self.attribute_total += 1

    def generalized_cost(self, S, penalty_value=True, p_div_value=True, allow_infeasible=True, tabu_search=False):
        """统一的目标函数，同时计算距离和各种惩罚项"""

        if not allow_infeasible and not self.is_solution_feasible(S):
            infeasible_cost = 1e10
            return (infeasible_cost, infeasible_cost) if tabu_search else infeasible_cost

        total_distance = sum(
            self.dist_matrix[self.node_id[route.nodes[i].id]][self.node_id[route.nodes[i + 1].id]]
            for route in S for i in range(len(route.nodes) - 1)
        )

        total_penalty = sum(
            self.alpha * self.load_violation(route) +
            self.beta * self.time_penalty(route) +
            self.gamma * self.battery_penalty(route)
            for route in S
        ) if penalty_value else 0

        p_div_penalty = 0
        if p_div_value:
            num_customers = sum(len(route.nodes) - 2 for route in S)
            num_vehicles = len(S)

            penalty_sum = sum(
                self.attribute_frequency.get((route.nodes[i].id, k, route.nodes[i - 1].id, route.nodes[i + 1].id), 0)
                for k, route in enumerate(S)
                for i in range(1, len(route.nodes) - 1)
            )

            p_div_penalty = (self.lambda_div * total_distance * penalty_sum *
                            ((num_customers * num_vehicles) ** 0.5) / (1e-10 + self.attribute_total))

        total_cost = total_distance + total_penalty + p_div_penalty

        if tabu_search:
            # 为禁忌搜索同时返回完整代价和距离
            return total_cost, total_distance
        else:
            return total_cost


    def print_route(self, route):
        """获取路径信息"""
        route_info = []
        for node in route.nodes:
            route_info.append(node.id)
        return '->'.join(route_info)

    def _two_opt(self, solution):
        """2-opt* 交换操作：在两条路径之间交换边"""
        two_opt_solution = []
        two_opt_tabu_list = []
        
        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                for split_1 in range(1, len(solution[i].nodes)-1):
                    for split_2 in range(1, len(solution[j].nodes)-1):
                        
                        # **先进行浅拷贝**
                        new_solution = solution[:]  

                        # **只深拷贝被修改的 `route`**
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        # **进行交换**
                        segment1_head, segment1_tail = new_solution[i].nodes[:split_1], new_solution[i].nodes[split_1:]
                        segment2_head, segment2_tail = new_solution[j].nodes[:split_2], new_solution[j].nodes[split_2:]

                        new_solution[i].nodes = segment1_head + segment2_tail
                        new_solution[j].nodes = segment2_head + segment1_tail

                        # 记录 `tabu_list` 信息
                        route_info = (['Two_opt', self.route_info[i] + str(split_1), self.route_info[j] + str(split_2)])
                        route_info.sort()

                        # **添加到解集中**
                        two_opt_solution.append(new_solution)
                        two_opt_tabu_list.append(route_info)
        
        return two_opt_solution, two_opt_tabu_list

    def _relocate(self, solution):
        """Relocate 操作：将一个客户从一条路径移动到另一条路径"""
        relocate_solution = []
        relocate_tabu_list = []

        for i in range(len(solution)):
            for j in range(len(solution)):
                for split_pos in range(1, len(solution[i].nodes)-1):
                    for insert_pos in range(1, len(solution[j].nodes)):

                        # **浅拷贝 `solution`**
                        new_solution = solution[:]
                        
                        # **仅深拷贝被修改的路径**
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        route_info = (['Relocate', self.route_info[i] + str(split_pos), self.route_info[j] + str(insert_pos)])
                        route_info.sort()

                        # **执行 Relocate**
                        if i == j and insert_pos > split_pos:
                            insert_pos -= 1  # 调整插入位置

                        node = new_solution[i].nodes.pop(split_pos)
                        new_solution[j].nodes.insert(insert_pos, node)

                        relocate_solution.append(new_solution)
                        relocate_tabu_list.append(route_info)

        return relocate_solution, relocate_tabu_list

    def _exchange(self, solution):
        """Exchange 操作：在两条路径之间交换两个客户"""
        exchange_solution = []
        exchange_tabu_list = []

        for i in range(len(solution)):
            for j in range(len(solution)):
                for split_pos1 in range(1, len(solution[i].nodes)-1):
                    if solution[i].nodes[split_pos1].type != 'c':
                        continue
                    for split_pos2 in range(1, len(solution[j].nodes)-1):
                        if solution[j].nodes[split_pos2].type != 'c' or (i == j and split_pos1 == split_pos2):
                            continue

                        # **先浅拷贝 `solution`**
                        new_solution = solution[:]

                        # **仅深拷贝被修改的 `route`**
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        # **执行交换**
                        new_solution[i].nodes[split_pos1], new_solution[j].nodes[split_pos2] = (
                            new_solution[j].nodes[split_pos2],
                            new_solution[i].nodes[split_pos1],
                        )

                        # **记录 `tabu_list` 信息**
                        route_info = (['Exchange', self.route_info[i] + str(split_pos1), self.route_info[j] + str(split_pos2)])
                        route_info.sort()

                        # **添加到解集中**
                        exchange_solution.append(new_solution)
                        exchange_tabu_list.append(route_info)

        return exchange_solution, exchange_tabu_list

    def _station_insertion(self, solution):
        """StationReIn 操作：插入/移除充电站，并使用局部 tabu 机制"""

        # **初始化局部 tabu list**
        if not hasattr(self, 'StationReIn_tabu_list'):
            self.StationReIn_tabu_list = {}

        station_in_re_solution = []
        station_in_re_tabu_list = []

        for i in range(len(solution)):
            for insert_pos in range(1, len(solution[i].nodes)):
                node = solution[i].nodes[insert_pos]

                # **移除充电站**
                if node.type == 'f':
                    μ, ζ = solution[i].nodes[insert_pos-1], solution[i].nodes[insert_pos+1]
                    arc = (μ.id, ζ.id)  # 记录被删除的 arc

                    # **浅拷贝 `solution`**
                    new_solution = solution[:]

                    # **仅深拷贝 `solution[i]`**
                    new_solution[i] = copy.deepcopy(solution[i])

                    # **移除充电站**
                    new_solution[i].nodes = new_solution[i].nodes[:insert_pos] + new_solution[i].nodes[insert_pos+1:]

                    # **记录 `tabu_list` 信息**
                    route_info = ['StationInReRemove', self.route_info[i] + "|" + str(insert_pos)]
                    route_info.sort()

                    # **更新 `tabu_list`**
                    tabu_tenure = random.randint(15, 30)
                    self.StationReIn_tabu_list[arc] = tabu_tenure

                    # **添加到解集中**
                    station_in_re_solution.append(new_solution)
                    station_in_re_tabu_list.append(route_info)

                # **插入充电站**
                else:
                    for station in self.instance.stations:
                        if solution[i].nodes[insert_pos-1].id == station.id:
                            continue  # 避免连续重复插入相同的充电站
                        
                        μ, ζ = solution[i].nodes[insert_pos-1], station
                        arc = (μ.id, ζ.id)

                        # **检查 `arc` 是否在 `tabu_list`**
                        if arc in self.StationReIn_tabu_list and self.StationReIn_tabu_list[arc] > 0:
                            continue  # 该弧仍在禁忌期，跳过

                        # **浅拷贝 `solution`**
                        new_solution = solution[:]

                        # **仅深拷贝 `solution[i]`**
                        new_solution[i] = copy.deepcopy(solution[i])

                        # **插入充电站**
                        new_solution[i].nodes.insert(insert_pos, station)

                        # **记录 `tabu_list` 信息**
                        route_info = ['StationInReInsert', self.route_info[i] + "|" + str(insert_pos)]
                        route_info.sort()

                        # **添加到解集中**
                        station_in_re_solution.append(new_solution)
                        station_in_re_tabu_list.append(route_info)

        # **减少 `tabu_list` 期限**
        for arc in list(self.StationReIn_tabu_list.keys()):
            if self.StationReIn_tabu_list[arc] > 0:
                self.StationReIn_tabu_list[arc] -= 1  # 每轮减少禁忌期
            if self.StationReIn_tabu_list[arc] == 0:
                del self.StationReIn_tabu_list[arc]  # 过期的 tabu 记录删除

        return station_in_re_solution, station_in_re_tabu_list

    def adjacent_check(self, solutions, name=None, checkpoint_mode=True):
        for solution in solutions:
            for route in solution:
                for i in range(1, len(route.nodes) - 1):
                    if route.nodes[i] == route.nodes[i-1]:
                        print("{} Adjacent Check Failed".format(name))
                        if checkpoint_mode:
                            breakpoint()
                        else:
                            return False
        return True

    def is_solution_feasible(self, solution):
        """检查整个解决方案的可行性"""
        # 记录所有已经服务的客户ID
        served_customers = set()
        # 检查每一条路径的可行性
        for route in solution:
            if not self.is_route_feasible(route):
                return False  # 如果任何路径不可行，返回False

            # 确保该路径中的所有客户都被服务，并且每个客户只被访问一次
            for node in route.nodes:
                if node.type == 'c':  # 只检查客户节点
                    if node.id in served_customers:
                        return False  # 如果该客户已经被访问过，返回False
                    served_customers.add(node.id)
        
        # 检查是否所有客户都被服务
        all_customers = {customer.id for customer in self.instance.customers}
        if served_customers != all_customers:
            return False  # 如果没有服务到所有客户，返回False

        # 如果所有客户都被服务并且路径可行，返回True
        return True

    def is_route_feasible(self, route, new_node=None):
        """检查路径可行性"""
        return not (self.load_violation(route) or self.time_violation(route) or self.battery_violation(route))

    def create_new_route(self):
        """创建新路径"""
        return Route([self.instance.depot])

    def solution_fix(self, solution):
        S = []
        for route in solution:
            # 1. 过滤掉连续重复的节点，提高效率
            route.nodes = [route.nodes[i] for i in range(len(route.nodes)) if i == 0 or route.nodes[i].id != route.nodes[i - 1].id]

            # 2. 检查是否包含至少一个客户节点（提高效率）
            if any(node.type == "c" for node in route.nodes):
                S.append(route)

        return S

    def polar_angle(self, customer, depot, random_point):
        """计算客户相对于 Depot 和随机点的极角"""
        # 向量1: Depot → 随机点
        dx1, dy1 = random_point.x - depot.x, random_point.y - depot.y
        # 向量2: Depot → 客户
        dx2, dy2 = customer.x - depot.x, customer.y - depot.y

        # 计算极角（相对于随机点）
        angle1 = math.atan2(dy1, dx1)  # Depot → 随机点
        angle2 = math.atan2(dy2, dx2)  # Depot → 客户

        # 计算相对角度并归一化到 [0, 2π]
        relative_angle = (angle2 - angle1) % (2 * math.pi) 
        return relative_angle

    def copy_solution(self, solution):
        """深拷贝解决方案"""
        return [self.copy_route(r) for r in solution]

    def copy_route(self, route):
        """复制单条路径"""
        new_route = Route()
        new_route.nodes = copy.deepcopy(route.nodes)
        new_route.load = route.load
        new_route.time = route.time
        new_route.fuel = route.fuel
        return new_route

    def print_solution(self, solution):
        res = []
        for routes in solution:
            route = []
            for node in routes.nodes:
                route.append(node.id)
            res.append(' -> '.join(route))
        res.sort()
        print(' | '.join(res))

def set_random_seed(seed: int):
    """设置所有常见库的随机种子"""
    # 1. 设置Python内置的随机模块的种子
    random.seed(seed)
    
    # 2. 设置NumPy的随机种子
    np.random.seed(seed)
    
    # 4. 设置Python环境的种子（确保跨平台可复现）
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"已设置随机种子为 {seed}")

def plot_solution(instance, solution):
    plt.figure(figsize=(10, 8))
    
    # 绘制仓库
    plt.scatter(instance.depot.x, instance.depot.y, c='red', marker='s', s=100, label='Depot')
    
    # 绘制充电站
    stations = instance.stations
    xs = [s.x for s in stations]
    ys = [s.y for s in stations]
    plt.scatter(xs, ys, c='green', marker='^', s=80, label='Stations')
    
    # 绘制客户
    customers = instance.customers
    xs = [c.x for c in customers]
    ys = [c.y for c in customers]
    plt.scatter(xs, ys, c='blue', marker='o', s=50, label='Customers')
    
    # 绘制路径
    colors = plt.cm.tab10.colors
    for i, route in enumerate(solution):
        color = colors[i % len(colors)]
        nodes = route.nodes
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        plt.plot(xs, ys, '--', color=color, linewidth=1)
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=8)
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('EVRPTW Solution Visualization')
    plt.grid(True)
    plt.show()

# # ====================== Files 主程序 ======================
# if __name__ == "__main__":
#     # 加载数据
#     data_dir = "/Users/maojietang/Desktop/Git_Pro/EVRP_TW/evrptw_instances/small_instances/Cplex5er/"
#     instance_file = "rc105C5.txt"

#     instance_path = os.path.join(data_dir, instance_file)
#     instance = load_instance(instance_path)
#     set_random_seed(1234)  # 设定一个固定种子
#     # 运行算法
#     solver = VNSTSolver(instance, predefine_route_number = 1)
#     print(f"样例Index:{instance_file}")
#     start_time = time.time()
#     solution = solver.solve()
#     end_time = time.time()
#     # 输出结果
#     print(f"最优解使用 {len(solution)} 辆车辆")
#     print(f"总行驶距离: {solver.global_value:.2f}")
#     print(f"样例用时: {(end_time-start_time):3f}")
    
#     # # 可视化
#     # plot_solution(instance, solution)

# ====================== Files 主程序 ======================
if __name__ == "__main__":
    # 加载数据 (rc208c5, r203C5, c208C5, r202C5)
    # {rc105C5: 238 -> 232 should be optimal}
    predefine_route_list = {
    "c101C5.txt":3, "c206C5.txt":3, "r104C5.txt":2, "r202C5.txt":1, "rc105C5.txt":3, "rc204C5.txt":1,
    "c103C5.txt":2, "c208C5.txt":1, "r105C5.txt":2, "r203C5.txt":1, "rc108C5.txt":2, "rc208C5.txt":1
    }
    data_dir = "./Cplex5er/"
    for instance_file in tqdm(os.listdir(data_dir)):
        if not instance_file.endswith("txt"):
            continue
        instance_path = os.path.join(data_dir, instance_file)
        instance = load_instance(instance_path)
        set_random_seed(1234)  # 设定一个固定种子
        # 运行算法
        solver = VNSTSolver(instance, predefine_route_number = predefine_route_list[instance_file])
        print(f"样例Index:{instance_file}")
        start_time = time.time()
        solution = solver.solve()
        end_time = time.time()
        # 输出结果
        print(f"最优解使用 {len(solution)} 辆车辆")
        print(f"总行驶距离: {solver.global_value:.2f}")
        print(f"样例用时: {(end_time-start_time):3f}s\n")
    
    # # 可视化
    # plot_solution(instance, solution)
