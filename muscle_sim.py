# muscle_sim.py
import argparse
import numpy as np
import opensim as osim

def run_simulation(time_array, activation_array):
    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    muscle_name = "iliacus_r"

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
    state.setTime(time_array[0])
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(time_array[-1])

    results_table = reporter.getTable()
    osim.STOFileAdapter.write(results_table, f"muscle_output.sto")
    print("Simulation complete. Data saved to 'muscle_output.sto'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True, help="Time interval")
    parser.add_argument("--T", type=float, required=True, help="")
    parser.add_argument("--activation", type=str, required=True, help="Path to activation .npy file")
    args = parser.parse_args()

    time_array = np.load(args.time)
    activation_array = np.load(args.activation)
    dt=args.dt

    run_simulation(time_array, activation_array, dt)
