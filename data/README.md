# How each field is generated

Below we detail **where each piece of data comes from** and **how we sample it** for the three spatial families (**R/C/RC**) and the two time-window regimes (**1 = tight, 2 = loose**). Unless otherwise stated, **system parameters** (vehicle speed, energy consumption, charging rate, capacities, max route time) are **taken directly from the Solomon Dataset of the same customer scale** (our code reads them from `parameters_by_type` in the config; `charging_rate = 1 / inverse_charging_rate`).

> **Abbreviations**: Depot = `D`, Recharging Station = `RS`, Customer = `C`.  
> **Normalization**: all coordinates are later divided by `pos_scale` (default `100`), time windows by `max_route_time`, demand by `demand_capacity`.

---

## 0) Shared primitives (used by all families)

- **Depot position `depot_loc`**  
  Sampled uniformly in the raw box `x_range × y_range` (e.g., `[0,100] × [0,100]`).  
  After generation, we normalize by `pos_scale`.

- **k Recharging Stations** `rs_loc`  
  We place `rs_size` RS (e.g., 3 by default). Each RS is sampled uniformly in the box but must be **reachable from the Depot** within one-way battery constraints:

$$
time(D \to RS) = \frac{\lVert RS - D \rVert}{velocity}, \quad
energy(D \to RS) = time \times \mathrm{energy\_consumption} \le \mathrm{battery\_capacity}.
$$


  If violated, resample that RS.

- **Customer count**  
  `customer_size` (e.g., `5` by default). Each customer’s **position**, **time window**, and **demand** are sampled per family rules below, with **feasibility checks** (battery + schedule).

- **Demand `demand`**  
  For each customer, draw an **integer** uniformly from `demand_range` (e.g., `[1, 50]`), then **scale** by `demand_capacity` of the current regime (`R1/R2/C1/C2/RC1/RC2`). Depot/RS demand is set to `0`.

- **Service time `service_time`**  
  Taken from the regime in the config. We store it **normalized** by `max_route_time`.

- **Time windows `time_window`**  
  For Depot/RS: `[0, max_route_time]` (normalized to `[0, 1]`).  
  For Customers (per regime):

  1. Compute shortest feasible reference travel time to **arrive** (typically `D → C`, or `D → RS → C` when needed).  
  2. Compute the minimum **return** time (e.g., `C → D`, or `C → RS → D`) including **charging time** when battery would be insufficient.  
  3. Define a **feasible center range**:

     $$
     c \sim \mathcal{U}\!\left(\text{arrive\_time},\ \text{max\_route\_time} - \text{return\_time} - \text{service\_time}\right).
     $$

  4. Draw a **width** `w` from a normal distribution controlled by regime:  
     - **Tight** (`*1`): smaller mean/variance (e.g., ~10% of `max_route_time`)  
     - **Loose** (`*2`): larger mean/variance (e.g., ~20%)

     Then set `TW = [c - w/2, c + w/2]` and **clip** to `[0, max_route_time]`.  
  5. With probability determined by `time_window_ratio_list` & `time_window_ratio_dist`, we **relax** to a **wide** TW (effectively loosening scheduling).

- **Feasibility checks (battery + time)**  
  Every customer must admit at least one feasible visit plan that respects:

  - **Battery:**

    $$
    \text{energy}(\text{leg}) = \text{time}(\text{leg}) \times \text{energy\_consumption} \le \text{battery\_capacity}.
    $$

    We typically ensure `D → C` uses ≤ 50% capacity or that `C` can reach a nearby `RS` and continue after **charging**.

  - **Time:** arrival by `l_i`, plus service and return before `max_route_time`.

---

## 1) Random family (`R1` / `R2`)

**Goal:** uniformly scattered customers, yet always operable.

- **Customer position `cus_loc`**  
  Sample each `C` uniformly in the box. Require that either:
  - **Direct pattern:** `D → C → D` is feasible (both time & battery), or  
  - **Two-hop pattern:** `D → RS → C → RS → D` is feasible, where **RS** is the **nearest** station to that leg.  
  If not, **resample** the customer position.

- **Customer time windows**  
  Use the shared TW procedure with regime-specific tight/loose widths (`R1` tight, `R2` loose).

- **Other parameters** *(copied from Solomon of the same scale)*  
  - `R1`: `service_time = 10`, `battery_capacity = 60.63`, `inverse_charging_rate = 0.49` (→ `charging_rate ≈ 2.0408`), `demand_capacity = 200`, `max_route_time = 230`.  
  - `R2`: same but `demand_capacity = 1000`, `max_route_time = 1000`.

---

## 2) Cluster family (`C1` / `C2`)

We support **two** cluster mechanisms that mirror typical Solomon patterns and a practical variant.

### 2.a Intra-city (Solomon-like)

- **Depot & RS:** as in shared primitives. RS act as **bridges** between spatial regions.  
- **Cluster centroids (`m` of them):** sample `m` **customer centroids** uniformly; split customers across clusters.  
- **Customer positions:** for cluster `j`, draw each customer from a Gaussian `N(centroid_j, \sigma^2 I)` (σ configurable; smaller σ → tighter clusters).  
- **Feasibility:** same feasibility tests as in `R`; resample a customer if it fails.  
- **Customer TW:** use shared TW procedure (`C1` tight, `C2` loose).

### 2.b Inter-city (RS-centered)

- **RS as centers:** treat **each RS** as a **cluster center**; customers around RS are drawn from `N(RS, \sigma^2 I)`.  
- **Extra RS (if any):** placed via the random RS rule.  
- **TW & feasibility:** identical to 2.a.

- **Other parameters** *(copied from Solomon of the same scale)*  
  - `C1`: `service_time = 90`, `battery_capacity = 77.75`, `inverse_charging_rate = 3.47` (→ `charging_rate ≈ 0.2882`), `demand_capacity = 200`, `max_route_time = 1236`.  
  - `C2`: same but `demand_capacity = 700`, `max_route_time = 3390`.

---

## 3) Mixed family (`RC1` / `RC2`)

- **Split customers**  
  Draw a ratio `mix_random_ratio` (e.g., `0.3`) → **30% Random**, **70% Cluster**.

- **RS placement priority**  
  Build RS following the **Cluster** logic first (to anchor clusters). If more RS are needed, fill via the **Random** RS rule.

- **Customer generation**  
  Generate the **Random subset** with the **R** rules and the **Cluster subset** with the **C** rules **independently**, then **merge**.

- **TW & feasibility**  
  Same as corresponding sub-rules; all customers must pass feasibility checks.

- **Other parameters** *(copied from Solomon of the same scale)*  
  - `RC1`: `service_time = 10`, `battery_capacity = 77.75`, `inverse_charging_rate = 0.39` (→ `charging_rate ≈ 2.5641`), `demand_capacity = 200`, `max_route_time = 240`.  
  - `RC2`: same but `demand_capacity = 1000`, `max_route_time = 960`.

---

## 4) Parsing from raw Solomon txt (baseline)

When using the **original Solomon txt files**, we:

- **Read** raw coordinates, time windows, demands, and per-regime constants.  
- **Normalize**: positions / `max_time` / `demand_capacity` as above.  
- **Pad/limit RS** using `rs_limit` if a fixed RS count is required by the model.  
- **Carry over** regime parameters exactly (no resampling).

---

## 5) Summary: what’s sampled vs. what’s copied

- **Sampled**
  - `depot_loc`, `rs_loc` (positions under battery constraints)  
  - `cus_loc` (uniform for **R**; Gaussian around centroids for **C/RC**)  
  - Customer **time windows** (center+width per tight/loose regime with feasibility)  
  - Customer **demand** integers in `demand_range`

- **Copied from Solomon (same scale & regime)**
  - `battery_capacity (Q)`, `demand_capacity (C)`  
  - `energy_consumption (r)`, `inverse_charging_rate (g)` → `charging_rate = 1/g`  
  - `velocity (v)`, `service_time`, `max_route_time`

This separation guarantees **statistical consistency** with Solomon’s EVRPTW while allowing **arbitrary instance counts** for training.

