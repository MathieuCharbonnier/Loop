import opensim as osim
import json
import os

def extract_muscle_resting_lengths(model_path):
    """
    Extract resting fiber lengths from an OpenSim model and return a dictionary.
    
    Parameters:
    -----------
    model_path : str
        Path to the OpenSim model file (.osim)
        
    Returns:
    --------
    dict
        Dictionary mapping muscle names to their resting fiber lengths
    """
    try:
        # Load the model
        model = osim.Model(model_path)
        print(f"Model loaded successfully: {model.getName()}")
        state = model.initSystem()
        model.equilibrateMuscles(state)

        muscles = model.getMuscles()
        print(f"Found {muscles.getSize()} muscles in the model")

        # Extract and store as dictionary
        muscle_data = {}

        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            name = muscle.getName()
            fiber_length = round(muscle.getFiberLength(state), 4)
            muscle_data[name] = fiber_length

        return muscle_data

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def save_results_to_json(data, output_json="muscle_resting_lengths.json"):
    """
    Save muscle data to a JSON file.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing muscle names and resting fiber lengths
    output_json : str
        Output JSON filename
    """
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"\nResults saved to: {output_json}")
    print(f"Total muscles saved: {len(data)}")

def main():
    model_path = "data/gait2392_millard2012_pelvislocked.osim"
    
    muscle_lengths = extract_muscle_resting_lengths(model_path)
    save_results_to_json(muscle_lengths)

if __name__ == "__main__":
    main()
