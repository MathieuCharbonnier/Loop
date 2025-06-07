import opensim as osim

def run_moco_with_ankle_motion():
    model_file = "data/gait2392_millard2012muscle.osim"
    motion_file = "data/BothLegsWalk.mot"

    # Create ModelProcessor and modify the model
    model_processor = osim.ModelProcessor(model_file)
    model_processor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_processor.append(osim.ModOpIgnoreTendonCompliance())
    model_processor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_processor.append(osim.ModOpAddReserves(1.0))

    # Setup MocoTrack
    track = osim.MocoTrack()
    track.setName("track_ankle_motion")
    track.setModel(model_processor)

    # Set motion data as the states reference
    track.setStatesReference(osim.TableProcessor(motion_file))
    track.set_states_global_tracking_weight(1.0)
    track.set_allow_unused_references(True)


    # Solve
    print("Solving MocoTrack...")
    solution = track.solve()
    solution.write("moco_solution.sto")
    print("Solution written to 'moco_solution.sto'")

    # Extract muscle activations
    activations = solution.exportToStatesTable()
    activations_file = "muscle_activations.sto"
    osim.STOFileAdapter.write(activations, activations_file)
    print(f"Muscle activations written to '{activations_file}'")

    return solution

if __name__ == "__main__":
    run_moco_with_ankle_motion()
