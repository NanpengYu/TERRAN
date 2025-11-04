import random
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import re
import numpy as np
import torch
import argparse

class Solomon_EVRPTW_Generation:
    def __init__(self, config_path):
        # Load configuration from JSON file
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # General parameters
        general_config = config.get("general", {})
        self.max_time = 0
        self.customer_size = general_config.get("customer_size")
        self.rs_size = general_config.get("rs_size")
        self.types_ratios = general_config.get("types_ratios_R_C_RC")
        self.x_range = general_config.get("x_range", [0, 100])
        self.y_range = general_config.get("y_range", [0, 100])
        self.pos_scale = general_config.get("pos_scale", 100)
        self.demand_range = general_config.get("demand_range", [5, 50])
        self.energy_consumption = general_config.get("energy_consumption", 1)
        self.velocity = general_config.get("velocity", 1)
        self.time_window_ratio_list = general_config.get("time_window_ratio_list", [0.25, 0.5, 0.75, 1.0])
        time_window_dist =  general_config.get("time_window_ratio_dist", [1, 1, 1, 4])
        self.time_window_ratio = self.sample_tw_ratio(time_window_dist)
        self.depot_based_ratio = general_config.get("depot_based_ratio", 0.5)
        self.rs_based_type_ratio = general_config.get("rs_based_type_ratio", 0.5)

        # Type-specific parameters
        self.parameters_by_type = config.get("parameters_by_type", {})

    def sample_tw_ratio(self, distribution):
        return random.choices([i for i in range(len(self.time_window_ratio_list))], weights=distribution, k=1)[0]

    def sample_from_distribution(self, distribution):
        """Sample a type based on provided distribution ratios."""
        return random.choices([0, 1, 2], weights=distribution, k=1)[0]

    def _generate_batch_instances(self, instances_number=1):
        """Generate all instances at once based on the config parameters."""
        data_instance_set = []
        for _ in tqdm(range(instances_number)):
            instance_data = self._generate_instances()
            data_instance_set.append(instance_data)

        return data_instance_set

    def _generate_instances(self, instances_number=1):
        """Generate all instances at once based on the config parameters."""
        # Generate instances
        # Step 1: Determine the problem type (0: R, 1: C, 2: RC)
        self.max_time = 0
        type_number = self.sample_from_distribution(self.types_ratios)
        
        # Step 2: Determine the time window tightness (1: strain, 2: loose)
        time_strain_type = random.choice([1, 2])  # 1 for strain, 2 for loose
        
        # Step 3: Construct type identifier (e.g., "R1", "R2", "C1", "C2", "RC1", "RC2")
        type_map = {0: "R", 1: "C", 2: "RC"}
        instance_type = f"{type_map[type_number]}{time_strain_type}"
        self.type = instance_type
        config = self.parameters_by_type[instance_type]

        demand_capacity = config.get("demand_capacity", 200)
        self.charging_rate = 1/config.get("inverse_charging_rate", 3.47)
        self.service_time = config.get("service_time", 10)

        # Step 4: Generate data for the selected instance type
        if instance_type in self.parameters_by_type:
            config = self.parameters_by_type[instance_type]
            self.battery_capacity = config.get("battery_capacity", 200)

            if instance_type.startswith("RC"):
                self.rs_random_ratio = config.get("rs_random_ratio", 0.5)
            depot_pos, rs_pos, cus_pos, time_windows, demands = self._generate_one_instance(instance_type, time_strain_type)
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        # Step 5: Map instance type to a unique number
        ins_number_dic = {"R1": 101, "R2": 102, "C1": 201, "C2": 202, "RC1": 301, "RC2": 302}
        ins_number = ins_number_dic[instance_type]
        instance_mask = [False] * (self.customer_size + self.rs_size + 1)

        # Pack all data into a dictionary
        instances_dict = {
            "depot_loc": depot_pos / self.pos_scale,  # Shape: (instances_number, 1, 2)
            "cus_loc": cus_pos / self.pos_scale,  # Shape: (instances_number, customer_size, 2)
            "rs_loc": rs_pos / self.pos_scale,  # Shape: (instances_number, rs_size, 2)
            "time_window": time_windows / self.max_time,  # Shape: (instances_number, 1+rs_size+customer_size, 2)
            "demand": demands / demand_capacity,  # Shape: (instances_number, 1+rs_size+customer_size)
            "max_time": self.max_time,  # Shape: (instances_number,)
            "demand_capacity": demand_capacity,  # Shape: (instances_number,)
            "battery_capacity": self.battery_capacity,  # Shape: (instances_number,)
            "types": instance_type,  # Shape: (instances_number,)
            "velocity_base": self.velocity,  # Scalar
            "energy_consumption": self.energy_consumption,  # Scalar
            "service_time": self.service_time / self.max_time,
            "charging_rate": self.charging_rate,
            "instance_mask": np.array(instance_mask)
        }
            
        return instances_dict

    def _generate_one_instance(self, instance_type, time_strain_type):
        """ Generate instance based on its type """
        if instance_type.startswith("RC"):
            depot_pos, rs_pos, customer_pos, time_windows, demands = self._generate_mix_rc_instances(time_strain_type)
        elif instance_type.startswith("R"):
            depot_pos, rs_pos, customer_pos, time_windows, demands = self._generate_random_instances(time_strain_type)
        else:
            depot_pos, rs_pos, customer_pos, time_windows, demands = self._generate_cluster_instances(time_strain_type)
        return depot_pos[None,:], rs_pos, customer_pos, time_windows, demands


    def _generate_random_instances(self, time_strain_type):
        """ Generate a R type instance """
        # Instance: Pos_X, Pos_Y, Demand, Time_Window

        # Pos
        position = self._gen_random_position()

        # Demand
        demand = self._generate_demands()

        # Time Windows
        time_windows = self._generate_random_time_windows(depot_rs = position[:1 + self.rs_size, :], 
                                                        customer_pos = position[1+self.rs_size:, :], 
                                                        time_strain_type = time_strain_type,)

        time_windows[:self.rs_size + 1, 1] = self.max_time * 1.0

        return position[0,:], position[1:1+self.rs_size,:], position[1+self.rs_size:, :], time_windows, demand


    def _generate_cluster_instances(self, time_strain_type):
        """ Generate a C type instance """
        # Instance: Pos_X, Pos_Y, Demand, Time_Window
        # Pos & time windows
        rs_based_data = random.random() > self.rs_based_type_ratio
        if rs_based_data:
            position, time_windows = self._gen_cluster_position_and_time_windows(time_strain_type)
        else:
            position, time_windows = self._gen_cluster_position_and_time_windows_non_rs_based(time_strain_type)
        # Demand
        demand = self._generate_demands()
        return position[0,:], position[1:1+self.rs_size,:], position[1+self.rs_size:, :], np.array(time_windows), demand

    def _generate_mix_rc_instances(self, time_strain_type):
        """ Generate a RC type instance """
        config = self.parameters_by_type.get("RC" + str(time_strain_type), {})
        self.max_time = config.get("max_route_time")
        random_instances = int(self.customer_size * self.rs_random_ratio)

        rs_based_data = random.random() > self.rs_based_type_ratio
        if rs_based_data:
            # Generate cluster positions and time windows
            cluster_instances = (1 + self.rs_size + self.customer_size) - random_instances
            cluster_position, cluster_time_window = self._gen_cluster_position_and_time_windows(time_strain_type, instance_size=cluster_instances)
        else:
            cluster_instances = self.customer_size - random_instances
            cluster_position, cluster_time_window = self._gen_cluster_position_and_time_windows_non_rs_based(time_strain_type, instance_size=cluster_instances)
            self.cluster_pos = cluster_position
        # Generate random positions and time windows
        random_position = self._gen_random_position(instance_size=random_instances, 
                                                    depot_require=False, 
                                                    rs_position_require=False, 
                                                    cus_require=True,
                                                    depot_init=cluster_position[0], 
                                                    RS_init=cluster_position[1:self.rs_size+1,:])
        self.random_pos = random_position
        random_time_window = self._generate_random_time_windows(depot_rs = cluster_position[:1 + self.rs_size, :], 
                                                                customer_pos = random_position,
                                                                time_strain_type = time_strain_type, 
                                                                instance_size=random_instances)

        # Combine positions and time windows
        positions = np.vstack((cluster_position, random_position))
        time_windows = np.vstack((cluster_time_window, random_time_window))

        # Generate demands
        demand = self._generate_demands()
        return positions[0,:], positions[1:1+self.rs_size,:], positions[1+self.rs_size:, :], time_windows, demand
        
    def _generate_demands(self):
        """ Generate demands for one instance """
        demands = np.random.uniform(self.demand_range[0], self.demand_range[1], size=(1 + self.rs_size + self.customer_size)).astype(int)
        demands[:self.rs_size + 1] = 0 # set 0 for depot and RSs
        return demands

    def _gen_random_position(self, instance_size=-1, depot_require=True, rs_position_require=True, cus_require=True, depot_init = None, RS_init = None):
        """
        Generate random positions for an instance:
        1. Generate Depot first.
        2. Generate RSs based on battery range.
        3. Generate Customers within a feasible range.

        Returns:
            positions: numpy array of shape (N, 2), where each row represents (x, y).
        """
        # Max travel distance for RSs and Customers
        max_travel_distance = self.battery_capacity / self.energy_consumption
        res = []

        depot = None
        if depot_require:
            depot = np.array([[np.random.uniform(self.x_range[0], self.x_range[1]),
                            np.random.uniform(self.y_range[0], self.y_range[1])]])  # Shape (1,2)
            res.append(depot)

        rs_positions = None
        if rs_position_require:
            rs_positions = []
            for _ in range(self.rs_size):
                while True:
                    rs_candidate = np.array([[
                        np.random.uniform(self.x_range[0], self.x_range[1]),
                        np.random.uniform(self.y_range[0], self.y_range[1])
                    ]])
                    if depot is not None:
                        distance_to_depot = np.linalg.norm(rs_candidate - depot)
                        if distance_to_depot / self.velocity * self.energy_consumption <= self.battery_capacity:
                            rs_positions.append(rs_candidate)
                            break
                    else:
                        rs_positions.append(rs_candidate)
                        break
            rs_positions = np.vstack(rs_positions) if rs_positions else np.empty((0, 2))  # Ensure valid shape
            res.append(rs_positions)

        customer_positions = None
        if (depot_init is None and not depot_require) or (RS_init is None and not rs_position_require):
            raise ValueError("You need to set initial depot/RS or generate them using the program.")
        if depot_init is not None:
            depot = depot_init
        if RS_init is not None:
            rs_positions = RS_init

        if cus_require:
            customer_positions = []
            instance_size = self.customer_size if instance_size == -1 else instance_size

            for _ in range(instance_size):
                while True:
                    customer_candidate = np.array([[
                        np.random.uniform(self.x_range[0], self.x_range[1]),
                        np.random.uniform(self.y_range[0], self.y_range[1])
                    ]])
                    min_distance = float("inf")  
                    
                    if depot is not None:
                        min_distance = min(min_distance, np.linalg.norm(customer_candidate - depot))
                    
                    if min_distance / self.velocity * self.energy_consumption <= (self.battery_capacity/2 - 0.1):
                        customer_positions.append(customer_candidate)
                        break

            customer_positions = np.vstack(customer_positions) if customer_positions else np.empty((0, 2))  # Ensure valid shape
            res.append(customer_positions)

        positions = np.vstack(res) if len(res) else np.empty((0, 2))

        return positions

    def _gen_cluster_position_and_time_windows_non_rs_based(self, time_strain_type, type_name="C", instance_size=-1, cluster_std=2):
        config = self.parameters_by_type.get(type_name + str(time_strain_type), {})

        # Set scheduling parameters
        if time_strain_type == 1:  # C1 (tight)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 200)
            mu_tw = 0.1 * self.max_time
            sigma_tw = 0.05 * self.max_time
        elif time_strain_type == 2:  # C2 (loose)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 1000)
            mu_tw = 0.2 * self.max_time
            sigma_tw = 0.1 * self.max_time
        else:
            raise ValueError("Invalid time strain type. Must be 1 (C1) or 2 (C2).")
        num_centroids = int(config.get("cluster_number", np.ceil(self.rs_size/2)))
        depot_and_rs = self._gen_random_position(self.rs_size, cus_require=False)
        s_c = self.service_time
        if instance_size == -1:
            instance_size = self.customer_size

        customer_per_cluster = np.array([instance_size // num_centroids for k in range(num_centroids)])
        if np.sum(customer_per_cluster) != instance_size:
            customer_per_cluster[-1] += instance_size - np.sum(customer_per_cluster)

        customer_positions = []
        customer_time_windows = []
        
        # generate centroids positions
        k = 0
        while k < num_centroids:
            count = 0
            while True:
                count += 1
                if count == 10000:
                    depot_and_rs = self._gen_random_position(self.rs_size, cus_require=False)
                    customer_positions = []
                    customer_time_windows = []
                    count = 0
                    k = 0
                    

                # Step1.1: Centroid
                centroid_position = np.array([[np.random.uniform(self.x_range[0], self.x_range[1]),
                                    np.random.uniform(self.y_range[0], self.y_range[1])]])  # Shape (1,2)

                # Position Constraints
                if (np.linalg.norm(centroid_position - depot_and_rs[0]) / self.velocity > self.max_time / 2):
                    continue
                
                # Step 1.2: Centroid TW
                time_all_depot_rs = np.linalg.norm(centroid_position - depot_and_rs, axis=1) / self.velocity 
                nearest_depot_rs_idx = np.argmin(time_all_depot_rs)
                time_nearest_time_depot_rs = time_all_depot_rs[nearest_depot_rs_idx]

                # Battery Constraints
                if time_nearest_time_depot_rs * self.energy_consumption > 0.5 * self.battery_capacity or time_nearest_time_depot_rs * self.energy_consumption < 0.1 * self.battery_capacity:
                    continue

                time_depot_to_nearest_rs = np.linalg.norm(depot_and_rs[nearest_depot_rs_idx] - depot_and_rs[0]) / self.velocity 
                earliest_time = 0 + time_depot_to_nearest_rs + time_nearest_time_depot_rs + (time_depot_to_nearest_rs * self.energy_consumption) / self.charging_rate
                lattest_time = self.max_time - s_c - time_depot_to_nearest_rs - time_nearest_time_depot_rs - (time_nearest_time_depot_rs * self.energy_consumption) / self.charging_rate
                if earliest_time > lattest_time:
                    continue
                    
                cur_customer = 0
                while True:
                    cluster_tw_center = np.random.uniform(earliest_time, lattest_time)
                    cluster_tw_width = 2 * np.abs(np.random.normal(mu_tw, sigma_tw, num_centroids))

                    customer_tw_center = np.random.normal(cluster_tw_center, cluster_tw_width)[0]
                    customer_tw_width = 2 * np.abs(np.random.normal(cluster_tw_width, 0.05 * cluster_tw_width))[0]

                    # Step 2.1: Customer Position
                    customer_pos = np.random.normal(loc=centroid_position, scale=cluster_std, size=(1, 2))

                    if (np.linalg.norm(centroid_position - depot_and_rs[0]) / self.velocity > self.max_time / 2):
                        continue
                    # Compute distances
                    dist_to_depot = np.linalg.norm(customer_pos - depot_and_rs[0]) / self.velocity
                    dist_to_rs = np.linalg.norm(customer_pos - depot_and_rs[1:], axis=1) / self.velocity  # RSs only

                    # Find the nearest RS
                    nearest_rs_idx = np.argmin(dist_to_rs)
                    t_nearest_rs = dist_to_rs[nearest_rs_idx] # Customer -> Nearest RS (Time)
                    depot_to_nearest_rs = np.linalg.norm(depot_and_rs[0] - depot_and_rs[1 + nearest_rs_idx]) / self.velocity

                    # Compute earliest and latest service times
                    e_i = max(0, customer_tw_center - customer_tw_width / 2)
                    l_i = min(customer_tw_center + customer_tw_width / 2, self.max_time)
    
                    # time window check
                    one_hop_head_to_node_time_check = (dist_to_depot < l_i) and (self.energy_consumption * dist_to_depot <= self.battery_capacity/2 - 0.05)
                    one_hop_back_to_depot_time_check = (dist_to_depot * 2 + self.service_time) < self.max_time


                    two_hop_head_to_node_time_check = (t_nearest_rs + depot_to_nearest_rs + (depot_to_nearest_rs * self.energy_consumption) / self.charging_rate <= l_i) and (t_nearest_rs * self.energy_consumption < self.battery_capacity / 2 - 0.01)
                    two_hop_back_to_depot_time_check = (2 * t_nearest_rs + 2 * depot_to_nearest_rs + (depot_to_nearest_rs * self.energy_consumption + t_nearest_rs * self.energy_consumption) / self.charging_rate + self.service_time) < self.max_time
                    if (one_hop_head_to_node_time_check and one_hop_back_to_depot_time_check):

                        if random.uniform(0, 1) > self.time_window_ratio:
                            e_i = 0
                            l_i = self.max_time - self.service_time / self.max_time

                        customer_positions.append(customer_pos)
                        customer_time_windows.append((e_i, l_i))
                        cur_customer+=1

                    elif two_hop_head_to_node_time_check and two_hop_back_to_depot_time_check:
                            if random.uniform(0, 1) > self.time_window_ratio:
                                e_i = 0
                                l_i = self.max_time - self.service_time / self.max_time
                            customer_positions.append(customer_pos)
                            customer_time_windows.append((e_i, l_i))
                            cur_customer+=1
                    
                    if cur_customer == customer_per_cluster[k]:
                        break
                    else:
                        continue
                if cur_customer == customer_per_cluster[k]:
                    break
            k += 1
            
        # Convert lists to numpy arrays
        customer_positions = np.vstack(customer_positions)

        # Combine centroids and customer positions
        positions = np.vstack((depot_and_rs, customer_positions))
        rs_depot_time_window = np.array([[np.array(0), np.array(self.max_time)]] * (self.rs_size + 1))
        time_windows = np.array(customer_time_windows).reshape(-1,2)
        time_windows = np.vstack((rs_depot_time_window, time_windows))
        time_windows[:1 + self.rs_size, 0] = 0
        time_windows[:1 + self.rs_size, 1] = self.max_time * 1.0
        return positions, time_windows

        


    def _gen_cluster_position_and_time_windows(self, time_strain_type, type_name="C", instance_size=-1, cluster_std=6):
        """
        Generate cluster positions and time windows for one instance, ensuring that customers
        can return to an RS or Depot and satisfy the time window constraints.

        :param time_strain_type: 1 for C1 (tight time windows), 2 for C2 (loose time windows)
        :param cluster_std: Standard deviation for customer dispersion around cluster centers.
        :return: Tuple (positions, time_windows) where positions is an array of shape (num_positions, 2),
                and time_windows is a list of (earliest, latest) tuples.
        """
        config = self.parameters_by_type.get(type_name + str(time_strain_type), {})

        # Set scheduling parameters
        if time_strain_type == 1:  # C1 (tight)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 200)
            mu_tw = 0.1 * self.max_time
            sigma_tw = 0.05 * self.max_time
        elif time_strain_type == 2:  # C2 (loose)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 1000)
            mu_tw = 0.2 * self.max_time
            sigma_tw = 0.1 * self.max_time
        else:
            raise ValueError("Invalid time strain type. Must be 1 (C1) or 2 (C2).")

        # Number of cluster centroids = Depot + RSs
        num_centroids = 1 + self.rs_size
        centroids = self._gen_random_position(num_centroids, cus_require=False)  # Generate Depot and RSs positions

        # Number of customers
        num_customers = self.customer_size if instance_size == -1 else (instance_size - 1 - self.rs_size)

        # Compute travel times from depot to each centroid
        t_0i = np.linalg.norm(centroids - centroids[0], axis=1) / self.velocity  # Depot → Cluster center
        t_c0 = np.linalg.norm(centroids[0] - centroids, axis=1) / self.velocity  # Cluster center → Depot
        s_c = self.service_time  # Random service time per cluster

        # Generate cluster center time windows
        cluster_tw_center = np.random.uniform(0 + t_0i, self.max_time - t_c0 - s_c)
        cluster_tw_width = 2 * np.abs(np.random.normal(mu_tw, sigma_tw, num_centroids))

        # Store cluster time windows
        cluster_time_windows = [(max(0, cluster_tw_center[i] - cluster_tw_width[i] / 2),
                                min(self.max_time, cluster_tw_center[i] + cluster_tw_width[i] / 2))
                                for i in range(num_centroids)]

        # Initialize customer positions and time windows
        customer_positions = []
        customer_time_windows = []

        # Compute travel times from customer to nearest RS
        depot_based = random.random() > self.depot_based_ratio
        for _ in range(num_customers):
            while True: 
                # Assign to the nearest centroid
                cluster_idx = np.random.choice(num_centroids) if depot_based else np.random.randint(1, num_centroids)
                cluster_center = centroids[cluster_idx]

                # Generate customer position based on normal distribution
                customer_pos = np.random.normal(loc=cluster_center, scale=cluster_std, size=(1, 2))

                # Compute distances
                dist_to_depot = np.linalg.norm(customer_pos - centroids[0]) / self.velocity
                dist_to_rs = np.linalg.norm(customer_pos - centroids[1:], axis=1) / self.velocity  # RSs only

                # Find the nearest RS
                nearest_rs_idx = np.argmin(dist_to_rs)
                t_nearest_rs = dist_to_rs[nearest_rs_idx] # Customer -> Nearest RS (Time)
                depot_to_nearest_rs = np.linalg.norm(centroids[0] - centroids[1 + nearest_rs_idx]) / self.velocity

                # Compute time window for customer
                customer_tw_center = np.random.normal(cluster_tw_center[cluster_idx], 0.1 * cluster_tw_width[cluster_idx])
                customer_tw_width = 2 * np.abs(np.random.normal(cluster_tw_width[cluster_idx], 0.05 * cluster_tw_width[cluster_idx]))

                # Compute earliest and latest service times
                e_i = max(0, customer_tw_center - customer_tw_width / 2)
                l_i = min(customer_tw_center + customer_tw_width / 2, self.max_time)

                # time window check
                one_hop_head_to_node_time_check = (dist_to_depot <= l_i) and (self.energy_consumption * dist_to_depot <= self.battery_capacity/2 - 0.05)
                one_hop_back_to_depot_time_check = (dist_to_depot * 2 + self.service_time) < self.max_time
                if (one_hop_head_to_node_time_check and one_hop_back_to_depot_time_check):
    
                    if random.uniform(0, 1) > self.time_window_ratio:
                        e_i = 0
                        l_i = self.max_time - self.service_time / self.max_time

                    customer_positions.append(customer_pos)
                    customer_time_windows.append((e_i, l_i))
                    break

                two_hop_head_to_node_time_check = (t_nearest_rs + depot_to_nearest_rs + (depot_to_nearest_rs * self.energy_consumption) / self.charging_rate <= l_i) and (t_nearest_rs * self.energy_consumption < self.battery_capacity / 2 - 0.01)
                two_hop_back_to_depot_time_check = (2 * t_nearest_rs + 2 * depot_to_nearest_rs + (depot_to_nearest_rs * self.energy_consumption + t_nearest_rs * self.energy_consumption) / self.charging_rate + self.service_time) < self.max_time
                if two_hop_head_to_node_time_check and two_hop_back_to_depot_time_check:
                        if random.uniform(0, 1) > self.time_window_ratio:
                            e_i = 0
                            l_i = self.max_time - self.service_time / self.max_time
                        customer_positions.append(customer_pos)
                        customer_time_windows.append((e_i, l_i))
                        break

                continue

        # Convert lists to numpy arrays
        customer_positions = np.vstack(customer_positions)

        # Combine centroids and customer positions
        positions = np.vstack((centroids, customer_positions))
        time_windows = np.array(cluster_time_windows + customer_time_windows)
        time_windows[:1 + self.rs_size, 0] = 0
        time_windows[:1 + self.rs_size, 1] = self.max_time * 1.0

        return positions, time_windows


    def _generate_random_time_windows(self, depot_rs, customer_pos, time_strain_type, instance_size=-1):
        """
        Generate random time windows based on time strain (R1 or R2).
        
        Constraints:
        - Depot -> Customer -> Nearest RS must be feasible.
        - If not, re-generate the time window.
        
        :param depot_rs: Array of depot and recharging station positions (N, 2).
        :param customer_pos: Array of customer positions (M, 2).
        :param time_strain_type: 1 for R1 (strain), 2 for R2 (loose).
        :param instance_size: Number of customers (-1 means use self.customer_size).
        :return: Numpy array of time windows [(earliest, latest), ...].
        """

        config = self.parameters_by_type.get("R" + str(time_strain_type), {})

        # Set max scheduling time based on R1 or R2
        if time_strain_type == 1:  # R1 (Strict)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 200)
            mu_tw = 0.1 * self.max_time
            sigma_tw = 0.05 * self.max_time
        elif time_strain_type == 2:  # R2 (Loose)
            if self.max_time == 0:
                self.max_time = config.get("max_route_time", 1000)
            mu_tw = 0.2 * self.max_time
            sigma_tw = 0.1 * self.max_time
        else:
            raise ValueError("Invalid time strain type. Must be 1 (R1) or 2 (R2).")

        customer_size = self.customer_size if instance_size == -1 else instance_size

        # Compute travel times from depot to each customer
        travel_dis = np.linalg.norm(customer_pos - depot_rs[0], axis=1)
        travel_times = travel_dis / self.velocity

        # Compute travel times from customer to nearest RS
        dist_customer_to_rs = np.linalg.norm(customer_pos[:, np.newaxis] - depot_rs, axis=2)
        nearest_rs_indices = np.argmin(dist_customer_to_rs, axis=1)
        dist_depot_to_nearest_rs = np.linalg.norm(depot_rs[0] - depot_rs[nearest_rs_indices], axis=1)
        
        # travel_times_to_nearest_rs = (dist_depot_to_nearest_rs + dist_customer_to_rs[np.arange(customer_size), nearest_rs_indices]) / self.velocity
        # Ensure valid time windows (Depot -> Customer -> RS)
        valid_time_windows = []
        for i in range(customer_size):
            while True: 
                # Sample tw_width from N(mu, sigma) and set TW_width = 2 * tw_width
                tw_width = np.abs(np.random.normal(mu_tw, sigma_tw))  # Ensure non-negative
                TW_width = 2 * tw_width

                # Generate random TW center
                tw_center = np.random.uniform(travel_times[i], self.max_time - travel_times[i] - self.service_time)

                # Calculate earliest and latest times
                earliest_time = max(0, tw_center - TW_width / 2)
                latest_time = min(self.max_time, tw_center + TW_width / 2)

                # depot -> Customer
                if ((travel_times[i] * self.energy_consumption < self.battery_capacity / 2 -0.05) and ((travel_times[i] < latest_time))):
                    if random.uniform(0, 1) > self.time_window_ratio:
                        earliest_time = 0
                        latest_time = self.max_time - self.service_time / self.max_time

                    valid_time_windows.append((earliest_time, latest_time))
                    break

        # Convert to NumPy array
        time_windows = np.array(valid_time_windows)

        if instance_size == -1:
            # Concatenate with Depot and RSs
            depot_rs_tw = np.array([[0, self.max_time]] * (1 + self.rs_size))
            time_windows = np.vstack((depot_rs_tw, time_windows))

        return time_windows

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def plot_instance(instances, idx, save_path="instance_plot"):
    """
    Plot the EVRPTW instance with different colors for Depot, Recharging Stations, and Customers.

    :param instances: Dictionary containing instance data.
                      Expected keys: "nodes" (array of (x,y) positions), 
                                     "time_window" (list of (earliest, latest) time windows),
                                     "rs_size" (int, number of recharging stations).
    :param save_path: File path to save the plot.
    """
    # Extract data
    ins_number_dic = {"R1": 101, "R2": 102, "C1": 201, "C2": 202, "RC1": 301, "RC2": 302}
    reverse_ins_number_dic = {}
    for key, value in ins_number_dic.items():
        reverse_ins_number_dic[value] = key 
    depot_pos = instances["depot_loc"]
    rs_pos = instances["rs_loc"]
    cus_pos = instances["cus_loc"]
    instance_type = instances["types"]

    data = np.vstack((depot_pos, rs_pos, cus_pos)) # Shape (N, 2)

    time_window = instances["time_window"]
    rs_size = len(rs_pos)

    # Define indices
    depot_idx = 0
    rs_indices = np.arange(1, rs_size + 1)
    customer_indices = np.arange(rs_size + 1, len(data))

    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot Depot
    plt.scatter(data[depot_idx, 0], data[depot_idx, 1], color='red', marker='s', s=150, label="Depot")

    # Plot Recharging Stations
    if len(rs_indices) > 0:
        plt.scatter(data[rs_indices, 0], data[rs_indices, 1], color='blue', marker='^', s=100, label="Recharging Station")

    # Plot Customers
    plt.scatter(data[customer_indices, 0], data[customer_indices, 1], color='green', marker='o', s=25, label="Customer")
    # Labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("EVRPTW Instance Visualization of {}".format(instance_type))
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_path = save_path + "_" + instance_type + ".png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Instance plot saved as {save_path}")


from scipy.spatial.distance import cdist
from tqdm import tqdm

def check_feasibility(instance, scale=120):
    depot = instance["depot_loc"]
    RSs = instance["rs_loc"]
    customer = instance["cus_loc"]
    rs_size = len(RSs)
    time_window_limit = instance["max_time"]
    battery_scale = instance["battery_capacity"]
    rs_speed_base = instance["charging_rate"]
    pos_scale = 100

    if depot.shape[0] + RSs.shape[0] + customer.shape[0] != scale:
        raise ValueError("The data scale does not match!")

    energy_consumption = instance["energy_consumption"] * time_window_limit / battery_scale
    velocity = instance["velocity_base"]  * time_window_limit / pos_scale
    rc_speed = rs_speed_base * time_window_limit / battery_scale 

    battery_capacity = instance["battery_capacity"] 
    # check position
    # (1) RS -> depot
    depot_to_RSs = np.linalg.norm(depot - RSs, axis=-1) / velocity
    depot_to_Cus = np.linalg.norm(depot - customer, axis=-1) / velocity

    if (depot_to_RSs * energy_consumption >= 1.0).any():
        raise ValueError("At least one RS cannot visit the depot")
    depot_with_rs = np.vstack((depot, RSs))
    
    # (2) Cus -> depot/RS
    cus_to_RS = cdist(customer, depot_with_rs)  
    min_cus_to_RS = cus_to_RS.min(axis=1) / velocity 

    if (min_cus_to_RS * energy_consumption >= 0.5).any():
        print(np.where((min_cus_to_RS * energy_consumption >= 0.5)))
        raise ValueError("At least one Customer cannot visit the depot/RSs")
    
    # Time window check
    time_window = instance["time_window"][rs_size + 1:]
    for i in range(len(customer)):
        # Case 1: Depot -> Customer
        if (depot_to_Cus[i] * energy_consumption < 0.5 * battery_capacity) and (depot_to_Cus[i] < time_window[i,1]):
            continue
        RS_to_Cus = np.linalg.norm(customer[i] - RSs, axis=-1) / velocity
        min_rs_id = np.argmin(RS_to_Cus)
        min_rs_dis = min(RS_to_Cus)
        charging_time = (depot_to_RSs[min_rs_id] * energy_consumption) / rc_speed
        # print((RS_to_Cus + depot_to_RSs[min_rs_id] + charging_time < time_window[i, 1]), (depot_to_RSs[min_rs_id] * energy_consumption < 0.5 * battery_capacity))
        if (min_rs_dis + depot_to_RSs[min_rs_id] + charging_time < time_window[i, 1] and  depot_to_RSs[min_rs_id] * energy_consumption < 0.5 * battery_capacity):
            continue
        breakpoint()
        raise ValueError("Time Window cannot match the instance case!")
    # print("Data Generation Success!")

def load_config_maxtime(config_path="./config.json"):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    parameters_by_type = config.get("parameters_by_type", {})
    time_dict = {}
    for key, value in parameters_by_type.items():
        time_dict[key] = value['max_route_time']
    return time_dict



def generate_from_solomon(dir_path="/data/Maojie_Github/EVRP_TW/evrptw_instances/small_instances/Cplex5er", 
                          config_path = "./config.json", 
                          pos_scale=100, 
                          rs_limit=20, 
                          instance_type="all"):
    # hardcode the file_path but can fix in the future
    data_instance_set = []
    max_time_dict = load_config_maxtime(config_path = config_path)
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        general_config = config.get("general", {})

    rs_limit = general_config.get("rs_limit", 20)
    instance_type = instance_type.upper()
    if instance_type != "ALL":
        files = sorted([i for i in (os.listdir(dir_path)) if i.endswith('txt') and i.upper().startswith(instance_type)])
    else:
        files = sorted([i for i in (os.listdir(dir_path)) if i.endswith('txt')])

    for i in tqdm(range(len(files))):
        txt_path = os.path.join(dir_path, files[i])
        with open(txt_path, 'r') as file:
            data = file.read()

        # Extract tabular data
        pattern = r"(\S+)\s+(\S+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
        rows = re.findall(pattern, data)

        rs_nodes = 0
        loc = []
        time_window = []
        demand = []
        types_data = []
        instance_mask = []

        service_time_recorded = False  # record if we've seen a non-zero ServiceTime
        first_service_time = None      

        for row in rows:
            if row[0].startswith("D"):
                continue  # 跳过 Depot
            elif row[0].startswith("S"):
                loc.append((float(row[2]) , float(row[3])))
                demand.append(0.0)
                time_window.append((0, 1))
                rs_nodes += 1
            else:
                while rs_nodes - 1 < rs_limit:
                    loc.append(loc[0])
                    demand.append(0.0)
                    time_window.append((0, 1))
                    rs_nodes += 1
                    instance_mask.append(True)

                loc.append((float(row[2]), float(row[3])))
                demand.append(float(row[4]))
                time_window.append((float(row[5]), float(row[6])))

                # record the first non-zero service time (since all of the share the same service time)
                service_time = float(row[7])
                if service_time != 0.0 and not service_time_recorded:
                    first_service_time = service_time
                    service_time_recorded = True  
            instance_mask.append(False)

        loc = np.array(loc)
        time_window = np.array(time_window)
        demand = np.array(demand)
        rs_nodes = rs_nodes - 1
        if rs_nodes > rs_limit:
            raise ValueError("RS limit is too low for this case!") 
        instance_type = txt_path.split('/')[-1][:3].upper() if txt_path.split('/')[-1][:2].upper().startswith("RC") else txt_path.split('/')[-1][:2].upper()
        max_time = max_time_dict[instance_type]
        time_window[rs_nodes+1:, :] /= max_time

        Q_match = re.search(r"Q\s+Vehicle fuel tank capacity\s+/([\d.]+)/", data)
        C_match = re.search(r"C\s+Vehicle load capacity\s+/([\d.]+)/", data)
        r_match = re.search(r"r\s+fuel consumption rate\s+/([\d.]+)/", data)
        g_match = re.search(r"g\s+inverse refueling rate\s+/([\d.]+)/", data)
        v_match = re.search(r"v\s+average Velocity\s+/([\d.]+)/", data)

        # if match is not found, set to None
        battery_capacity = float(Q_match.group(1)) if Q_match else None
        demand_capacity = float(C_match.group(1)) if C_match else None
        energy_consum_rate = float(r_match.group(1)) if r_match else None
        rs_speed_base = 1/float(g_match.group(1)) if g_match else None # inverse charging_rate
        velocity_base = float(v_match.group(1)) if v_match else None

        instances_dict = {
        "depot_loc": loc[0][None,:] / pos_scale,  # Shape: (instances_number, 1, 2)
        "cus_loc": loc[rs_nodes+1:, :] / pos_scale,  # Shape: (instances_number, customer_size, 2)
        "rs_loc": loc[1:rs_nodes+1, :] / pos_scale,  # Shape: (instances_number, rs_size, 2)
        "time_window": time_window,  # Shape: (instances_number, 1+rs_size+customer_size, 2)
        "demand": demand / demand_capacity,  # Shape: (instances_number, 1+rs_size+customer_size)
        "max_time": max_time,  # Shape: (instances_number,)
        "demand_capacity": demand_capacity,  # Shape: (instances_number,)
        "battery_capacity": battery_capacity,  # Shape: (instances_number,)
        "types": instance_type,  # Shape: (instances_number,)
        "velocity_base": velocity_base,  # Scalar
        "energy_consumption": energy_consum_rate,  # Scalar
        "service_time": service_time / max_time,
        "charging_rate": rs_speed_base,
        "instance_mask": np.array(instance_mask)
        }
        
        data_instance_set.append(instances_dict)
    return data_instance_set

def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_and_save_dataset(seed, save_path="../data/evrptw/", instances_number = 1000, config_path = "./config.json", name="validation"):
    set_seed(seed)  # fix random seed
    dataset = Solomon_EVRPTW_Generation("./config.json")
    instance_list = dataset._generate_batch_instances(instances_number = instances_number)
    scale = instance_list[0]["cus_loc"].shape[0]
    data_path = "{}evrptw{}_{}_seed{}.pkl".format(save_path, scale, name, seed)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_dataset(instance_list, data_path)

def prepare_and_save_solomon_dataset(data_path, save_root, config_path, name="validation", seed=0, instance_type="all"):  
    instance_list = generate_from_solomon(data_path, instance_type=instance_type, config_path=config_path)
    scale = instance_list[0]["cus_loc"].shape[0]
    if instance_type == "all":
        save_path = "{}evrptw{}_{}_seed{}.pkl".format(save_root, scale, name, seed)
    else:
        save_path = "{}evrptw{}_{}_seed{}_type_{}.pkl".format(save_root, scale, name, seed, instance_type)
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    save_dataset(instance_list, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Solomon EVRPTW instances")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input Solomon dataset")
    parser.add_argument("--save_root", type=str, required=True, help="Directory to save generated instances")
    parser.add_argument("--instance_type", type=str, default="all", choices=["all", "C", "R", "RC"], help="Instance type to generate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)
    data = Solomon_EVRPTW_Generation(args.config)
    prepare_and_save_solomon_dataset(
        data_path=args.data_path,
        save_root=args.save_root,
        instance_type=args.instance_type,
        config_path=args.config,
        seed=args.seed
    )

# python instance_generator.py \
#   --config ./configs/config_5c.json \
#   --data_path ./data/solomon_datasets/small_instances/Cplex5er/ \
#   --save_root ./data/solomon_evrptw_input/ \
#   --instance_type all
