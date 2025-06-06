Creating a simple rocket simulator to model rocket launches involves implementing basic physics principles and allowing for the input of various parameters to analyze how they affect the rocket's flight. Below, I’ll provide a basic Python simulation that models a rocket's launch using fundamental physics equations.

### Python Rocket Simulator

This simulation will include parameters such as initial velocity, mass, thrust, drag, and gravitational acceleration to model the flight trajectory of a rocket.

#### Requirements

Before running the code, ensure you have Python installed along with the `matplotlib` library for plotting the results. You can install it via pip if you haven't already:

```bash
pip install matplotlib
```

#### Simulator Code

Here’s a basic code structure for the rocket simulator:

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravitational acceleration (m/s^2)

class Rocket:
    def __init__(self, mass, thrust, burn_time, drag_coefficient, cross_sectional_area):
        self.mass = mass  # initial mass (kg)
        self.thrust = thrust  # thrust (N)
        self.burn_time = burn_time  # burn time (s)
        self.drag_coefficient = drag_coefficient  # drag coefficient (dimensionless)
        self.cross_sectional_area = cross_sectional_area  # cross-sectional area (m^2)
        self.velocity = 0  # initial velocity (m/s)
        self.altitude = 0  # initial altitude (m)

    def calculate_drag(self):
        # Drag force (F_d = 0.5 * C_d * rho * A * v^2)
        rho = 1.225  # air density at sea level (kg/m^3)
        drag_force = 0.5 * self.drag_coefficient * rho * self.cross_sectional_area * (self.velocity ** 2)
        return drag_force

    def update_state(self, dt):
        if self.burn_time > 0:
            # Calculate forces
            drag = self.calculate_drag()
            net_force = self.thrust - drag - (self.mass * g)
            # Update mass (burning fuel, assuming constant thrust)
            self.mass -= (self.thrust / 9.81) * dt  # Mass decrease over time based on thrust

        else:
            net_force = -self.calculate_drag() - (self.mass * g)

        # Update acceleration, velocity, and altitude
        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.altitude += self.velocity * dt

        # Update burn time
        if self.burn_time > 0:
            self.burn_time -= dt

def simulate_rocket_launch(rocket, total_time, dt):
    times = np.arange(0, total_time, dt)
    altitudes = []
    velocities = []

    for t in times:
        rocket.update_state(dt)
        altitudes.append(rocket.altitude)
        velocities.append(rocket.velocity)

        # Stop simulation if rocket crashes
        if rocket.altitude < 0:
            break

    return times, altitudes, velocities

# Parameters for the rocket
initial_mass = 1000  # kg
thrust = 15000  # N
burn_time = 30  # s
drag_coefficient = 0.75  # dimensionless
cross_sectional_area = 0.2  # m^2

# Create the rocket
rocket = Rocket(initial_mass, thrust, burn_time, drag_coefficient, cross_sectional_area)

# Run the simulation
total_time = 60  # total time to simulate (s)
dt = 0.1  # time step (s)
times, altitudes, velocities = simulate_rocket_launch(rocket, total_time, dt)

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times, altitudes)
plt.title('Rocket Altitude Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(times, velocities)
plt.title('Rocket Velocity Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid()

plt.tight_layout()
plt.show()
```

### Explanation of the Code

1. **Rocket Class**: This class contains the rocket's properties (mass, thrust, etc.) and methods to calculate drag and update the rocket's state.
   
2. **Drag Calculation**: The drag force is computed based on the rocket's velocity, cross-sectional area, and drag coefficient.

3. **State Update**: The rocket's mass decreases over time due to fuel burn. The acceleration, velocity, and altitude are updated based on the net force acting on the rocket.

4. **Simulation Function**: This function simulates the rocket's launch over a specified duration and time step, storing altitude and velocity data.

5. **Visualization**: The results are plotted using `matplotlib` to visualize the altitude and velocity over time.

### Running the Simulation

You can modify parameters such as `initial_mass`, `thrust`, `burn_time`, `drag_coefficient`, and `cross_sectional_area` to observe how they affect the rocket's flight performance.

### Conclusion

This simple rocket simulator provides a foundational understanding of rocket dynamics and the effects of various parameters on flight. You can expand upon this model by adding more complexity, such as multiple stages, varying thrust profiles, or more sophisticated drag models, to create a more advanced simulator.