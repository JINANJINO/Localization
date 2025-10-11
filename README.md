#  Localization

## 1. What is Localization in Robotics?

In robotics, **localization** is the process by which a robot determines its own position and orientation within its environment.  
Essentially, localization enables a robot to answer the fundamental question: **“Where am I?”**

---

### 🔹 Key Components of Localization

To perform localization, a robot typically needs three components:

#### **1. Map**
A representation of the environment that the robot uses for reference. Common types include:
- **Occupancy Grid:** Divides the environment into cells (occupied, free, or unknown).
- **Feature Map:** Stores distinct landmarks such as doors, walls, or furniture.

#### **2. Sensors**
Sensors allow the robot to perceive and measure its surroundings:
- **LiDAR:** Emits laser beams to measure distance, producing 2D/3D point clouds.  
- **Camera:** Provides rich visual information for identifying landmarks or performing visual odometry.  
- **IMU (Inertial Measurement Unit):** Measures acceleration and angular velocity to estimate motion.  
- **Wheel Encoders (Odometry):** Estimate displacement from wheel rotations (subject to drift over time).

#### **3. Algorithms**
Localization algorithms compare current sensor data to the map to estimate the most probable position.  
This process is often probabilistic — instead of a single position, the robot maintains a **belief distribution** over possible locations.  
A popular example is the **Particle Filter**, which represents multiple hypotheses (“particles”) and updates their likelihoods as new data arrives.

---

## 2. Moving Average Filter

> **Goal:** Reduce sensor noise and stabilize localization estimates.

In localization, sensor readings (e.g., IMU, LiDAR, GPS) are often noisy, causing unstable position or velocity estimates.  
A **Moving Average Filter** smooths these fluctuations to produce more reliable localization results.

---

### 2.1 What is a Recursive Moving Average Filter?

The **Moving Average Filter (MAF)** is a simple **low-pass filter** that reduces high-frequency noise by averaging recent data points.  
However, computing a full average for each new sample can be inefficient.  
The **Recursive Moving Average Filter** provides an efficient alternative by updating the mean incrementally.

---

### 2.2 Mathematical Formulation

Let the input signal be \( x_k \) and the filtered output be \( a_k \).  
The recursive moving average is defined as:

$$
\[
a_k = \frac{k - 1}{k} a_{k - 1} + \frac{1}{k} x_k
\]
$$

Where:
- $$\( a_k \)$$: current filtered value (moving average)  
- $$\( a_{k-1} \)$$: previous average  
- $$\( x_k \)$$: current input sample  
- $$\( k \)$$: current time step (sample index)

This equation blends the new sample $$\( x_k \)$$ with the previous mean $$\( a_{k-1} \)$$, progressively refining the average as more data becomes available.

---

### 2.3 Implementation Example
> You can find this in my repository in the **```average_filter.py```** file.

- **Result**

[Screencast from 10-11-2025 03:26:20 PM.webm](https://github.com/user-attachments/assets/194b4635-ca8b-4854-adc5-37455bcf7551)

---

## 3. Bayes Filter
The **Bayes Filter** is a probabilistic approach to estimate the **state** of a system (e.g., robot position) given **uncertain measurements** and **control inputs**.

It follows the principle of **Bayesian inference**, combining **prior belief**, **motion model**, and **sensor model** to compute a **posterior belief**.

### 🔸 Algorithmic Form

At each time step $$\( t \)$$:

1. **Prediction Step**  
   Based on the control input $$\( u_t \)$$:

$$
\overline{bel}(x_t) = \int p(x_t \mid u_t, x_{t-1}) \, bel(x_{t-1}) \, dx_{t-1}
$$

   - $$\( \overline{bel}(x_t) \)$$: Predicted belief before observing new measurement  
   - $$\( p(x_t \mid u_t, x_{t-1}) \)$$: Motion (state transition) model  
   - $$\( bel(x_{t-1}) \)$$: Previous belief distribution  

3. **Correction (Update) Step**  
   Incorporating the measurement $$\( z_t \)$$:

   $$
   bel(x_t) = \eta \, p(z_t \mid x_t) \, \overline{bel}(x_t)
   $$

   - $$\( bel(x_t) \)$$: Updated (posterior) belief  
   - $$\( p(z_t \mid x_t) \)$$: Sensor model  
   - $$\( \eta \)$$: Normalization constant ensuring probabilities sum to 1  

### 🔹 Intuitive Explanation
- The **Prediction Step** moves the belief based on how the system is expected to evolve.
- The **Correction Step** adjusts this belief based on what the sensors observe.
- Over time, this iterative process converges toward a more accurate estimate of the true state — even under uncertainty.

---

## Summary Comparison

| Filter Type | Nature | Handles Uncertainty | Use Case |
|--------------|--------|--------------------|-----------|
| Moving Average Filter | Deterministic | ❌ No | Simple smoothing for noisy data |
| Bayes Filter | Probabilistic | ✅ Yes | Robot localization, tracking, SLAM |

---

