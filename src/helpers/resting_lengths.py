

import opensim as osim
import pandas as pd
import os

def extract_muscle_resting_lengths(model_path):
    """
    Extract optimal fiber lengths from an OpenSim model.
    
    Parameters:
    -----------
    model_path : str
        Path to the OpenSim model file (.osim)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing muscle names and their optimal fiber lengths
    """
    
    try:
        # Load the model
        model = osim.Model(model_path)
        print(f"Model loaded successfully: {model.getName()}")
        
        # Get all muscles from the model
        muscles = model.getMuscles()
        print(f"Found {muscles.getSize()} muscles in the model")
        
        # Extract muscle data
        muscle_data = []
        
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            muscle_name = muscle.getName()
            optimal_fiber_length = muscle.getOptimalFiberLength()
            
            muscle_data.append({
                'Muscle Name': muscle_name,
                'Optimal Fiber Length (m)': optimal_fiber_length
            })
        
        # Create DataFrame
        df = pd.DataFrame(muscle_data)
        
        # Sort by muscle name for better readability
        df = df.sort_values('Muscle Name').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def save_results(df, output_csv="muscle_resting_lengths.csv"):
    """
    Save results to CSV file and display formatted table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing muscle data
    output_csv : str
        Output CSV filename
    """
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Display formatted table
    print("\n" + "="*80)
    print("MUSCLE RESTING LENGTHS (OPTIMAL FIBER LENGTHS)")
    print("="*80)
    
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total number of muscles: {len(df)}")
    print(f"Mean optimal fiber length: {df['Optimal Fiber Length (m)'].mean():.4f} m")
    print(f"Median optimal fiber length: {df['Optimal Fiber Length (m)'].median():.4f} m")
    print(f"Min optimal fiber length: {df['Optimal Fiber Length (m)'].min():.4f} m")
    print(f"Max optimal fiber length: {df['Optimal Fiber Length (m)'].max():.4f} m")
    print(f"Standard deviation: {df['Optimal Fiber Length (m)'].std():.4f} m")

def main():
    """Main function to execute the muscle length extraction."""
    
    # Model path
    model_path = "data/gait2392_millard2012_pelvislocked.osim"
    

    # Extract muscle resting lengths
    df = extract_muscle_resting_lengths(model_path)
        
    # Save and display results
    save_results(df)
        
       

if __name__ == "__main__":
    main()
