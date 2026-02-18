
import polars as pl
from datetime import datetime

from afml import DollarBarsProcessor
import os

def reproduce():
    # Create a dummy parquet file with non-standard column names
    filename = "dummy_ticks.parquet"
    
    # Create data with 'timestamp', 'price', 'qty' (instead of datetime, close, volume)
    data = {
        "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5)],
        "price": [100.0, 101.0, 102.0, 103.0, 104.0],
        "quantity": [10, 10, 10, 10, 10], 
        "is_buyer_maker": [True, False, True, False, True] # Extra column mentioned in user issue
    }
    
    df = pl.DataFrame(data)
    df.write_parquet(filename)
    print(f"Created {filename} with columns: {df.columns}")
    
    # Try to process it using transform_chunked
    # We need to fit it first (which we can do via fit_dynamic or manually setting threshold)
    processor = DollarBarsProcessor(daily_target=10)
    processor.threshold_ = 50.0 # Manually set threshold to skip fit
    
    print("Attempting transform_chunked via run_step_bars (Pipeline Integration)...")
    try:
        import sys
        sys.path.append(os.getcwd())
        from afml_polars_pipeline import run_step_bars
        
        # run_step_bars handles loading, renaming, and calling transform_chunked
        # We need to ensure output_path is set to avoid overwriting real data, or just check return
        
        # Create a dummy visual_analysis dir if needed
        os.makedirs("visual_analysis_test", exist_ok=True)
        
        # Test 1: run_step_bars with our dummy file
        # It should succeed now because run_step_bars renames cols and passes LazyFrame to transform_chunked
        output_file = "dummy_dollar_bars.parquet"
        result = run_step_bars(
            input_path=filename, 
            daily_target=10, 
            visualize=False,
            output_path=output_file
            # Force verify we are hitting the chunked path by ensuring it treats parquet as lazy
            # logic in run_step_bars: if endswith .parquet -> is_lazy=True -> transform_chunked
        )
        
        print("Success! Pipeline integration verified.")
        print(f"Generated {result['bars_count']} bars.")

        if os.path.exists(output_file):
            print(f"Output file created: {output_file}")
            os.remove(output_file)

        
    except Exception as e:
        print(f"Caught unexpected error in pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists("visual_analysis_test"):
            import shutil
            shutil.rmtree("visual_analysis_test")
        # cleanup default output if generated
        if os.path.exists("data/dollar_bars_polars.parquet"):
             # Don't delete real data if we accidentaly wrote to it? 
             # The script defaults to data/dollar_bars_polars.parquet. 
             # I should probably backup it or use output override.
             pass

if __name__ == "__main__":
    reproduce()
