import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from brian2  import *

class Controller:
    def __init__(self, biologicalsystem, target_amplitude, target_period, 
                 update_interval=200, prediction_horizon=1000):
        """
        Hierarchical controller for ankle movement using EES parameters.
        
        Args:
            target_amplitude: Desired amplitude of sinusoidal movement (degrees)
            target_period: Desired period of sinusoidal movement (milliseconds)
            update_interval: Controller update interval in ms (should be multiple of model step)
            prediction_horizon: How far ahead to predict for MPC (milliseconds)
        """
        self.biologicalsystem=biologicalsystem
        self.target_amplitude = target_amplitude
        self.target_period = target_period
        self.update_interval = update_interval
        self.prediction_horizon = prediction_horizon
        
        # Internal model step (given as 50ms)
        self.model_step = 50*ms
        
        # Check that update interval is multiple of model step
        assert update_interval % self.model_step == 0, "Update interval must be multiple of model step"
        self.steps_per_update = update_interval // self.model_step
        
        # Current parameters
        self. ees_params={'freq': 30*hertz,
                    'intensity':intensity,#fixed during all the simulation
                    'balance': 0.0} # initial balance paramater between flexor and extensor [-1,1]
        
        # Controller memory
        self.history = {
            'time': [],
            'actual_angle': [],
            'desired_angle': [],
            'ees_frequency': [],
            'balance': [],
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
        ees_freq, balance = params
        
        # Constraint penalties
        if ees_freq < 10 or ees_freq > 100:
            return 1000 + abs(ees_freq - 50)
        if balance < -0.6 or balance > 0.6:
            return 1000 + abs(balance )
        
        # Predict future states
        num_steps = self.prediction_horizon // self.model_step
        states = []
        angles = []
        
        # Make a copy of current state to avoid modifying the original
        state = self.current_state.copy()
        
        # Calculate number of model steps to simulate
        steps_to_simulate = min(num_steps, 10)  # Limit to 10 steps for computational efficiency
        
        # Simulate forward with these parameters
        for i in range(steps_to_simulate):
            # Run one step of reflex model
            next_state, angle = closed_loop(
                state, 
                ees_frequency=ees_freq, 
                balance=balance
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
        initial_guess = [self.ees_frequency, self.balance]
        
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
        self.history['balance'].append(self.flexor_ratio)
        self.history['error'].append(desired_angle - current_angle)
        
        # Optimize parameters for next interval
        optimal_ees, optimal_balance = self.optimize_parameters()
        
        # Update control parameters

        
        # Run reflex model for the steps in this control interval

        # Apply current parameters
        spikes, time_series = self.biologicalsystem.run_simulation(
            n_iteration,
            time_steps,
            ees_stimulation_params=self.ees_stimulation_params,
            torque_profile=None, 
            seed=42,
            base_output_path=None, 
            plot=True)
            
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


