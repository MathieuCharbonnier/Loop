# muscle_sim.py
import argparse
import numpy as np
import opensim as osim

def run_simulation(dt, T, activation_array, output_name):
    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    muscle_name = "iliacus_r"
    time_array=np.arange(0,T, dt)

    class ActivationController(osim.PrescribedController):
        def __init__(self, model, muscle_name, time_array, activation_array):
            super().__init__()
            self.setName("ActivationController")
            muscle = model.getMuscles().get(muscle_name)
            self.addActuator(muscle)

            func = osim.PiecewiseLinearFunction()
            for t, a in zip(time_array, activation_array):
                func.addPoint(t, float(a))

            self.prescribeControlForActuator(muscle_name, func)

    controller = ActivationController(model, muscle_name, time_array, activation_array)
    model.addController(controller)

    reporter = osim.TableReporter()
    reporter.setName("MuscleReporter")
    reporter.set_report_time_interval(dt)
    muscle = model.getMuscles().get(muscle_name)
    reporter.addToReport(muscle.getOutput("fiber_length"), "fiber_length")
    reporter.addToReport(muscle.getOutput("fiber_velocity"), "fiber_velocity")
    model.addComponent(reporter)

    state = model.initSystem()
    manager = osim.Manager(model)
    state.setTime(0)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(T)

    results_table = reporter.getTable()
    osim.STOFileAdapter.write(results_table, output_name)
    print(f"Simulation complete. Data saved to {output_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True, help="Time interval")
    parser.add_argument("--T", type=float, required=True, help="Final time")
    parser.add_argument("--activation", type=str, required=True, help="Path to activation .npy file")
    parser.add_argument("--output", type=str, required=True, help="Name of the output file")
    args = parser.parse_args()

  
    activation_array = np.load(args.activation)
    output_name=args.output
    dt=args.dt
    T=args.T


    run_simulation(dt, T, activation_array, output_name)
