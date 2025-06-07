import opensim as osim
import numpy as np

def get_muscle_activations():
    # File paths
    model_file = 'data/gait2392_millard2012_pelvislocked.osim'
    motion_file = 'data/BothLegsWalk.mot'
    
    # Load the model
    model = osim.Model(model_file)
    
    # Get time range from motion file
    motion_data = osim.TimeSeriesTable(motion_file)
    start_time = motion_data.getIndependentColumn()[0]
    end_time = motion_data.getIndependentColumn()[-1]
    
    print(f"Time range: {start_time} to {end_time} seconds")
    
    # Step 1: Inverse Dynamics
    print("Running Inverse Dynamics...")
    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(model_file)
    id_tool.setCoordinatesFileName(motion_file)
    id_tool.setOutputGenForceFileName('inverse_dynamics.sto')
    id_tool.setStartTime(start_time)
    id_tool.setEndTime(end_time)
    id_tool.setLowpassCutoffFrequency(6.0)
    
    # Run inverse dynamics
    id_tool.run()
    print("Inverse Dynamics completed.")
    
    # Step 2: Static Optimization for muscle activations
    print("Running Static Optimization...")
    
    # Load model for static optimization
    model = osim.Model(model_file)
    
    # Disable all muscles except the ones we want
    muscle_set = model.getMuscles()
    target_muscles = ["tib_ant_r", "med_gas_r"]
    """
    for i in range(muscle_set.getSize()):
        muscle = muscle_set.get(i)
        muscle_name = muscle.getName()
        
        if muscle_name not in target_muscles:
            # Set other muscles to minimum activation
            muscle.setMinControl(0.001)
            muscle.setMaxControl(0.001)
        else:
            print(f"Keeping muscle active: {muscle_name}")
    """
    # Create Static Optimization analysis
    static_opt = osim.StaticOptimization()
    static_opt.setName('StaticOptimization')
    static_opt.setStartTime(start_time)
    static_opt.setEndTime(end_time)
    static_opt.setUseModelForceSet(True)
    static_opt.setActivationExponent(2)
    static_opt.setConvergenceCriterion(1e-4)
    static_opt.setMaxIterations(1000)
    
    # Add analysis to model
    model.addAnalysis(static_opt)
    
    # Initialize the model
    state = model.initSystem()
    
    # Create AnalyzeTool to run the static optimization
    analyze_tool = osim.AnalyzeTool()
    analyze_tool.setModelFilename(model_file)
    analyze_tool.setCoordinatesFileName(motion_file)
    analyze_tool.setExternalLoadsFileName('inverse_dynamics.sto')
    analyze_tool.setStartTime(start_time)
    analyze_tool.setEndTime(end_time)
    analyze_tool.setLowpassCutoffFrequency(6.0)
    analyze_tool.setResultsDir('.')
    analyze_tool.setName('muscle_activation_analysis')
    
    # Run the analysis
    analyze_tool.run()
    print("Static Optimization completed.")
    
    # Step 3: Extract and display results
    print("Extracting muscle activations...")
    
    # Read the activation results
    activation_file = 'muscle_activation_analysis_StaticOptimization_activation.sto'
    activation_data = osim.TimeSeriesTable(activation_file)
        
    # Get time vector
    time_vec = np.array(activation_data.getIndependentColumn())
        
    # Extract muscle activations
    tib_ant_activations = []
    med_gas_activations = []
        
    # Get column labels to find the right indices
    labels = activation_data.getColumnLabels()
    print(f"Available muscle columns: {labels}")
        
    # Find the correct column indices
    tib_ant_idx = -1
    med_gas_idx = -1
        
    for i, label in enumerate(labels):
        if 'tib_ant_r' in label:
            tib_ant_idx = i
        elif 'med_gas_r' in label:
            med_gas_idx = i
        
    if tib_ant_idx >= 0 and med_gas_idx >= 0:
        # Extract data
        for i in range(activation_data.getNumRows()):
            row = activation_data.getRowAtIndex(i)
            tib_ant_activations.append(row.getElt(tib_ant_idx))
            med_gas_activations.append(row.getElt(med_gas_idx))
            
        # Convert to numpy arrays
        tib_ant_a = np.array(tib_ant_activations)
        med_gas_a = np.array(med_gas_activations)
            
        # Display results
        print(f"\nResults Summary:")
        print(f"Time points: {len(time_vec)}")
        print(f"Tibialis Anterior - Mean: {np.mean(tib_ant_a):.3f}, Max: {np.max(tib_ant_a):.3f}, Min: {np.min(tib_ant_a):.3f}")
        print(f"Medial Gastrocnemius - Mean: {np.mean(med_gas_a):.3f}, Max: {np.max(med_gas_a):.3f}, Min: {np.min(med_gas_a):.3f}")
            
        # Save results to file
        results = np.column_stack((time_vec, tib_ant_a, med_gas_a))
        np.savetxt('muscle_activations.txt', results, 
                      header='time tib_ant_r_activation med_gas_r_activation', 
                      fmt='%.6f')
        print(f"\nResults saved to 'muscle_activations.txt'")
            
        return time_vec, tib_ant_a, med_gas_a
            
    else:
        print("Error: Could not find the specified muscles in the results")
        return None, None, None
            

# Run the analysis
if __name__ == "__main__":
    time, tib_ant_activations, med_gas_activations = get_muscle_activations()
    
    if time is not None:
        print("\nMuscle activation analysis completed successfully!")
        print("The activation values (between 0 and 1) represent the neural drive needed")
        print("for each muscle to achieve the desired ankle motion.")
    else:
        print("Analysis failed. Please check the error messages above.")
