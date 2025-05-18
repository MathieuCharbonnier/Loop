import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HierarchicalAnkleController:
    def __init__(self, reflex_model, target_amplitude, target_period, 
                 update_interval=200, prediction_horizon=1000):
        """
        Hierarchical controller for ankle movement using EES parameters.
        
        Args:
            reflex_model: Function that simulates reflex loop for specified time steps
            target_amplitude: Desired amplitude of sinusoidal movement (degrees)
            target_period: Desired period of sinusoidal movement (milliseconds)
            update_interval: Controller update interval in ms (should be multiple of model step)
            prediction_horizon: How far ahead to predict for MPC (milliseconds)
        """
        self.reflex_model = reflex_model
        self.target_amplitude = target_amplitude
        self.target_period = target_period
        self.update_interval = update_interval
        self.prediction_horizon = prediction_horizon
        
        # Internal model step (given as 50ms)
        self.model_step = 50
        
        # Check that update interval is multiple of model step
        assert update_interval % self.model_step == 0, "Update interval must be multiple of model step"
        self.steps_per_update = update_interval // self.model_step
        
        # Current parameters
        self.current_state = None  # Will be set when sim starts
        self.ees_frequency = 30.0  # Initial EES frequency
        self.flexor_ratio = 0.5    # Initial flexor ratio (0-1)
        
        # Controller memory
        self.history = {
            'time': [],
            'actual_angle': [],
            'desired_angle': [],
            'ees_frequency': [],
            'flexor_ratio': [],
            'error': []
        }
        
        # Current simulation time
        self.current_time = 0
        
    def desired_trajectory(self, t):
        """Generate desired angle at time t (in ms)"""
        t_seconds = t / 1000.0  # Convert to seconds for readability
        period_seconds = self.target_period / 1000.0
        return self.target_amplitude * np.sin(2 * np.pi * t_seconds / period_seconds)
    
    def initialize_simulation(self, initial_state):
        """Initialize simulation with given state"""
        self.current_state = initial_state
        self.current_time = 0
        
        # Clear history
        for key in self.history:
            self.history[key] = []
            
    def cost_function(self, params):
        """
        Cost function for optimization.
        Predicts future trajectory and compares with desired trajectory.
        
        Args:
            params: [ees_frequency, flexor_ratio]
            
        Returns:
            Cost value (lower is better)
        """
        ees_freq, flex_ratio = params
        
        # Constraint penalties
        if ees_freq < 10 or ees_freq > 100:
            return 1000 + abs(ees_freq - 50)
        if flex_ratio < 0.1 or flex_ratio > 0.9:
            return 1000 + abs(flex_ratio - 0.5)
        
        # Predict future states
        num_steps = self.prediction_horizon // self.model_step
        states = []
        angles = []
        
        # Make a copy of current state to avoid modifying the original
        state = self.current_state.copy()
        
        # Calculate number of model steps to simulate
        steps_to_simulate = min(num_steps, 20)  # Limit to 20 steps for computational efficiency
        
        # Simulate forward with these parameters
        for i in range(steps_to_simulate):
            # Run one step of reflex model
            next_state, angle = self.reflex_model(
                state, 
                ees_frequency=ees_freq, 
                flexor_ratio=flex_ratio
            )
            
            states.append(next_state)
            angles.append(angle)
            state = next_state
        
        # Calculate error over prediction horizon
        cost = 0
        for i, angle in enumerate(angles):
            t = self.current_time + (i+1) * self.model_step
            desired = self.desired_trajectory(t)
            error = angle - desired
            
            # Weighted cost: earlier predictions matter more
            weight = 1.0 / (i + 1)
            cost += weight * (error ** 2)
            
        # Add regularization to discourage large parameter changes
        reg_ees = 0.1 * ((ees_freq - self.ees_frequency) ** 2)
        reg_ratio = 5.0 * ((flex_ratio - self.flexor_ratio) ** 2)
        
        return cost + reg_ees + reg_ratio
    
    def optimize_parameters(self):
        """
        Use model predictive control to optimize EES parameters.
        Returns optimal parameters for next control interval.
        """
        # Initial guess = current parameters
        initial_guess = [self.ees_frequency, self.flexor_ratio]
        
        # Parameter bounds
        bounds = [(10, 100), (0.1, 0.9)]  # EES frequency, flexor ratio
        
        # Optimize
        result = minimize(
            self.cost_function,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Get optimal parameters
        optimal_ees, optimal_ratio = result.x
        
        return optimal_ees, optimal_ratio
    
    def update(self):
        """
        Update controller and run simulation for one controller time step.
        """
        # Get current angle from state
        current_angle = self.current_state['joint_angle']
        desired_angle = self.desired_trajectory(self.current_time)
        
        # Record current state before updating
        self.history['time'].append(self.current_time)
        self.history['actual_angle'].append(current_angle)
        self.history['desired_angle'].append(desired_angle)
        self.history['ees_frequency'].append(self.ees_frequency)
        self.history['flexor_ratio'].append(self.flexor_ratio)
        self.history['error'].append(desired_angle - current_angle)
        
        # Optimize parameters for next interval
        optimal_ees, optimal_ratio = self.optimize_parameters()
        
        # Update control parameters
        self.ees_frequency = optimal_ees
        self.flexor_ratio = optimal_ratio
        
        # Run reflex model for the steps in this control interval
        for _ in range(self.steps_per_update):
            # Apply current parameters
            next_state, _ = self.reflex_model(
                self.current_state,
                ees_frequency=self.ees_frequency,
                flexor_ratio=self.flexor_ratio
            )
            
            # Update state and time
            self.current_state = next_state
            self.current_time += self.model_step
    
    def run_simulation(self, duration_ms):
        """
        Run the complete simulation for specified duration.
        
        Args:
            duration_ms: Duration in milliseconds
        """
        num_updates = duration_ms // self.update_interval
        
        for _ in range(num_updates):
            self.update()
            
        return self.history
    
    def plot_results(self):
        """Plot simulation results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Convert time to seconds for readability
        time_sec = [t/1000 for t in self.history['time']]
        
        # Plot angles
        ax1.plot(time_sec, self.history['desired_angle'], 'b-', label='Desired')
        ax1.plot(time_sec, self.history['actual_angle'], 'r-', label='Actual')
        ax1.set_ylabel('Joint Angle (deg)')
        ax1.legend()
        ax1.set_title('Ankle Joint Angle')
        
        # Plot EES frequency
        ax2.plot(time_sec, self.history['ees_frequency'], 'g-')
        ax2.set_ylabel('EES Frequency (Hz)')
        ax2.set_title('EES Stimulation Parameter')
        
        # Plot flexor ratio
        ax3.plot(time_sec, self.history['flexor_ratio'], 'm-')
        ax3.set_ylabel('Flexor Ratio')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Flexor Recruitment Ratio')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print performance metrics
        errors = np.array(self.history['error'])
        rmse = np.sqrt(np.mean(errors**2))
        print(f"RMSE: {rmse:.2f} degrees")
        print(f"Max error: {np.max(np.abs(errors)):.2f} degrees")


# Example usage with a simplified reflex model
def example_reflex_model(state, ees_frequency, flexor_ratio):
    """
    Simplified reflex model for demonstration.
    In your real implementation, you would replace this with your actual model.
    
    Args:
        state: Current state dictionary
        ees_frequency: EES frequency (Hz)
        flexor_ratio: Ratio of flexor recruitment (0-1)
        
    Returns:
        next_state: Updated state dictionary
        joint_angle: Current joint angle
    """
    # Unpack current state
    current_angle = state['joint_angle']
    current_velocity = state['joint_velocity']
    flexor_activation = state['flexor_activation']
    extensor_activation = state['extensor_activation']
    
    # Simplified muscle dynamics (replace with your model)
    # EES influences motoneuron firing
    flexor_target = ees_frequency * flexor_ratio
    extensor_target = ees_frequency * (1 - flexor_ratio)
    
    # Simple first-order dynamics for muscle activation
    tau = 0.1  # Time constant (s)
    flexor_activation += (flexor_target - flexor_activation) * (0.05 / tau)
    extensor_activation += (extensor_target - extensor_activation) * (0.05 / tau)
    
    # Net torque from muscle activation
    net_torque = 0.2 * (flexor_activation - extensor_activation)
    
    # Simple joint dynamics (replace with your model)
    # Add angle-dependent passive forces
    passive_torque = -0.05 * current_angle - 0.01 * current_velocity
    
    # Total torque
    total_torque = net_torque + passive_torque
    
    # Update velocity and position
    # Simple Euler integration
    dt = 0.05  # 50ms
    inertia = 0.1  # Joint inertia
    new_velocity = current_velocity + total_torque * dt / inertia
    new_angle = current_angle + new_velocity * dt
    
    # Create new state
    next_state = {
        'joint_angle': new_angle,
        'joint_velocity': new_velocity,
        'flexor_activation': flexor_activation,
        'extensor_activation': extensor_activation
    }
    
    return next_state, new_angle

# Run a test simulation
def run_test():
    # Initial state
    initial_state = {
        'joint_angle': 0.0,
        'joint_velocity': 0.0,
        'flexor_activation': 0.0,
        'extensor_activation': 0.0
    }
    
    # Create controller
    controller = HierarchicalAnkleController(
        reflex_model=example_reflex_model,
        target_amplitude=15.0,  # 15 degrees
        target_period=2000.0,   # 2 seconds (2000ms)
        update_interval=200,    # 200ms controller update
        prediction_horizon=1000  # 1000ms prediction
    )
    
    # Initialize and run simulation
    controller.initialize_simulation(initial_state)
    controller.run_simulation(10000)  # 10 seconds
    
    # Plot results
    controller.plot_results()

if __name__ == "__main__":
    run_test()
