    def copy_brian_dict(d):
        """
        Safely copy dictionaries that may contain Brian2 quantities.
        
        Parameters:
        -----------
        d : dict or other
            Dictionary or value to copy
            
        Returns:
        --------
        dict or other
            Copied dictionary or value
        """
        if isinstance(d, dict):
            # If d is a dictionary, apply the function to each value
            return {k: BiologicalSystem.copy_brian_dict(v) for k, v in d.items()}
        elif hasattr(d, 'copy') and callable(getattr(d, 'copy', None)):
            # If the object has a callable `.copy()` method (e.g., Brian2 Quantity), use it
            return d.copy()
        else:
            # Otherwise, return the value as-is (int, float, string, bool, etc.)
            return d
