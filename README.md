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
    a_k = ((k - 1) / k) * a_prev + (1 / k) * x_k
    return a_k
