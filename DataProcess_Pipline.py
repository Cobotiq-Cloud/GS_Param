#!/usr/bin/env python3
"""
Parameter Dataset Updater

This script updates the Full_Param dataset with values from update CSV files,
then matches parameters with a YAML configuration file.

Features:
- Updates Full_Param.csv with any number of update CSV files
- Automatically detects parameter path columns
- Matches updated dataset with YAML configuration
- Generates comprehensive reports
- Flexible for future updates - just add CSV files to update directory

Usage:    
    python DataProcess_Pipline.py \
    --base-csv /Users/chenzaowen/Desktop/GS_Param/Param_CSV/Full_Param.csv \
    --yaml-config /Users/chenzaowen/Desktop/GS_Param/Yaml/A4-00-14p-20250417_user_config.yaml \
    --update-dir Update \
    --output-dir Results

Arguments:
    --base-csv : Path to base parameter CSV file
    --yaml-config : Path to YAML configuration file
    --update-dir : Directory containing update CSV files
    --output-dir : Output directory for results
    -h, --help : Show help message


"""

import pandas as pd
import yaml
import os
import argparse
import glob
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParameterUpdater:
    """Class to handle parameter dataset updates and YAML matching."""
    
    def __init__(self, base_csv_path: str, yaml_path: str = None, update_dir: str = None):
        """
        Initialize the Parameter Updater.
        
        Args:
            base_csv_path: Path to the main Full_Param.csv file
            yaml_path: Path to YAML configuration file (optional)
            update_dir: Directory containing update CSV files (optional)
        """
        self.base_csv_path = base_csv_path
        self.yaml_path = yaml_path
        self.update_dir = update_dir or "updates"
        self.param_column_mapping = {}
        
    def detect_parameter_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the parameter path column in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Column name containing parameter paths
        """
        # Common patterns for parameter path columns
        path_patterns = [
            '参数位置', 'parameter_path', 'param_path', 'path', 'location',
            '1参数位置和名称', '/strategy/', 'config_path'
        ]
        
        for col in df.columns:
            # Check if column name matches patterns
            if any(pattern in str(col).lower() for pattern in path_patterns):
                # Verify column contains path-like values
                sample_values = df[col].dropna().head(10)
                if any(str(val).startswith('/') for val in sample_values):
                    logger.info(f"Detected parameter column: {col}")
                    return col
                    
        # If no obvious column found, check for columns with path-like values
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                path_count = sum(1 for val in sample_values if str(val).startswith('/'))
                if path_count >= len(sample_values) * 0.7:  # 70% threshold
                    logger.info(f"Detected parameter column by content: {col}")
                    return col
                    
        raise ValueError(f"Could not detect parameter path column in DataFrame with columns: {list(df.columns)}")
    
    def load_base_dataset(self) -> pd.DataFrame:
        """Load the main parameter dataset."""
        try:
            df = pd.read_csv(self.base_csv_path, encoding='utf-8')
            df.columns = df.columns.str.strip()  # Clean column names
            logger.info(f"Loaded base dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading base dataset: {e}")
            raise
    
    def load_update_files(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load all CSV files from the update directory.
        
        Returns:
            List of tuples (filename, DataFrame)
        """
        update_files = []
        
        if not os.path.exists(self.update_dir):
            logger.warning(f"Update directory {self.update_dir} does not exist")
            return update_files
            
        csv_files = glob.glob(os.path.join(self.update_dir, "*.csv"))
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df.columns = df.columns.str.strip()
                filename = os.path.basename(file_path)
                update_files.append((filename, df))
                logger.info(f"Loaded update file: {filename} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
                
        return update_files
    
    def update_dataset(self, base_df: pd.DataFrame, update_files: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Update the base dataset with values from update files.
        
        Args:
            base_df: Main parameter dataset
            update_files: List of (filename, DataFrame) tuples
            
        Returns:
            Updated DataFrame
        """
        updated_df = base_df.copy()
        update_log = []
        
        # Detect parameter column in base dataset
        base_param_col = self.detect_parameter_column(base_df)
        
        for filename, update_df in update_files:
            try:
                # Detect parameter column in update file
                update_param_col = self.detect_parameter_column(update_df)
                
                # Get parameter paths that exist in both datasets
                base_paths = set(updated_df[base_param_col].dropna())
                update_paths = set(update_df[update_param_col].dropna())
                common_paths = base_paths.intersection(update_paths)
                
                logger.info(f"Processing {filename}: {len(common_paths)} matching parameters")
                
                if not common_paths:
                    logger.warning(f"No matching parameters found in {filename}")
                    continue
                
                # Update matching rows
                for path in common_paths:
                    base_idx = updated_df[updated_df[base_param_col] == path].index
                    update_row = update_df[update_df[update_param_col] == path].iloc[0]
                    
                    if len(base_idx) > 0:
                        base_idx = base_idx[0]
                        updated_columns = []
                        
                        # Update each column that has a value in update file
                        for col in update_df.columns:
                            if col != update_param_col and pd.notna(update_row[col]) and str(update_row[col]).strip():
                                # Map update column to base column if possible
                                target_col = self.map_column_name(col, updated_df.columns)
                                if target_col:
                                    old_value = updated_df.loc[base_idx, target_col]
                                    new_value = update_row[col]
                                    updated_df.loc[base_idx, target_col] = new_value
                                    updated_columns.append(f"{target_col}: '{old_value}' -> '{new_value}'")
                                else:
                                    # Add new column if it doesn't exist
                                    if col not in updated_df.columns:
                                        updated_df[col] = None
                                    updated_df.loc[base_idx, col] = update_row[col]
                                    updated_columns.append(f"Added {col}: '{update_row[col]}'")
                        
                        if updated_columns:
                            update_log.append({
                                'file': filename,
                                'parameter': path,
                                'updates': updated_columns
                            })
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        # Save update log
        self.save_update_log(update_log)
        logger.info(f"Dataset updated with {len(update_log)} parameter changes")
        
        return updated_df
    
    def map_column_name(self, update_col: str, base_columns: List[str]) -> str:
        """
        Map update file column names to base dataset column names.
        
        Args:
            update_col: Column name from update file
            base_columns: List of column names in base dataset
            
        Returns:
            Mapped column name or None if no mapping found
        """
        # Define column mappings
        column_mappings = {
            '参数名称': ['4参数中文名称', 'chinese_name', 'name'],
            '参数释义': ['5参数释义', 'description', 'explanation'],
            '默认值': ['default_value', 'default', 'value'],
            '功能分类1': ['6功能分类1', 'category1', 'function_category'],
            '功能分类2': ['7功能分类2', 'category2', 'sub_category'],
            '功能说明': ['8功能说明', 'function_description', 'notes']
        }
        
        # Direct match first
        if update_col in base_columns:
            return update_col
            
        # Check mappings
        for key, possible_names in column_mappings.items():
            if update_col == key:
                for name in possible_names:
                    if name in base_columns:
                        return name
        
        # Fuzzy match
        update_col_lower = update_col.lower()
        for base_col in base_columns:
            if update_col_lower in base_col.lower() or base_col.lower() in update_col_lower:
                return base_col
                
        return None
    
    def extract_yaml_paths(self, yaml_file: str) -> List[str]:
        """
        Extract all parameter paths from YAML file.
        
        Args:
            yaml_file: Path to YAML configuration file
            
        Returns:
            List of parameter paths
        """
        def extract_paths(obj, prefix=""):
            paths = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        paths.extend(extract_paths(value, current_path))
                    else:
                        paths.append(f"/{current_path}")
            return paths
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            paths = extract_paths(data)
            logger.info(f"Extracted {len(paths)} parameter paths from YAML")
            return paths
        except Exception as e:
            logger.error(f"Error reading YAML file: {e}")
            return []
    
    def match_with_yaml(self, updated_df: pd.DataFrame, yaml_paths: List[str]) -> pd.DataFrame:
        """
        Match updated dataset with YAML configuration.
        
        Args:
            updated_df: Updated parameter dataset
            yaml_paths: List of parameter paths from YAML
            
        Returns:
            DataFrame with matched parameters only
        """
        if not yaml_paths:
            logger.warning("No YAML paths provided, returning full dataset")
            return updated_df
            
        param_col = self.detect_parameter_column(updated_df)
        matched_df = updated_df[updated_df[param_col].isin(yaml_paths)].copy()
        
        # Calculate statistics
        total_yaml = len(yaml_paths)
        total_matched = len(matched_df)
        match_percentage = (total_matched / total_yaml * 100) if total_yaml > 0 else 0
        
        logger.info(f"YAML matching results:")
        logger.info(f"  - Total YAML parameters: {total_yaml}")
        logger.info(f"  - Matched parameters: {total_matched}")
        logger.info(f"  - Match percentage: {match_percentage:.1f}%")
        
        # Save unmatched parameters for reference
        dataset_paths = set(updated_df[param_col].dropna())
        unmatched_yaml = [path for path in yaml_paths if path not in dataset_paths]
        
        if unmatched_yaml:
            unmatched_df = pd.DataFrame({'unmatched_yaml_path': unmatched_yaml})
            unmatched_df.to_csv('unmatched_yaml_parameters.csv', index=False)
            logger.info(f"Saved {len(unmatched_yaml)} unmatched YAML parameters to unmatched_yaml_parameters.csv")
        
        return matched_df
    
    def save_update_log(self, update_log: List[Dict]) -> None:
        """Save update log to file."""
        if update_log:
            log_data = []
            for entry in update_log:
                for update in entry['updates']:
                    log_data.append({
                        'file': entry['file'],
                        'parameter': entry['parameter'],
                        'update': update
                    })
            
            log_df = pd.DataFrame(log_data)
            log_df.to_csv('parameter_update_log.csv', index=False)
            logger.info(f"Saved update log with {len(log_data)} changes to parameter_update_log.csv")
    
    def generate_report(self, original_df: pd.DataFrame, updated_df: pd.DataFrame, 
                       matched_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive report of the update process."""
        report = {
            'original_dataset': {
                'rows': len(original_df),
                'columns': len(original_df.columns)
            },
            'updated_dataset': {
                'rows': len(updated_df),
                'columns': len(updated_df.columns),
                'new_columns': list(set(updated_df.columns) - set(original_df.columns))
            }
        }
        
        if matched_df is not None:
            report['matched_dataset'] = {
                'rows': len(matched_df),
                'columns': len(matched_df.columns)
            }
        
        return report
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Run the complete parameter update and matching pipeline.
        
        Returns:
            Final processed DataFrame
        """
        logger.info("Starting parameter update pipeline...")
        
        # Step 1: Load base dataset
        base_df = self.load_base_dataset()
        
        # Step 2: Load update files
        update_files = self.load_update_files()
        
        # Step 3: Update dataset
        updated_df = self.update_dataset(base_df, update_files)
        
        # Step 4: Save updated full dataset
        updated_df.to_csv('Full_Param_Updated.csv', index=False, encoding='utf-8')
        logger.info("Saved updated full dataset to Full_Param_Updated.csv")
        
        # Step 5: Match with YAML if provided
        final_df = updated_df
        if self.yaml_path and os.path.exists(self.yaml_path):
            yaml_paths = self.extract_yaml_paths(self.yaml_path)
            if yaml_paths:
                final_df = self.match_with_yaml(updated_df, yaml_paths)
                final_df.to_csv('Final_Matched_Parameters.csv', index=False, encoding='utf-8')
                logger.info("Saved final matched parameters to Final_Matched_Parameters.csv")
        
        # Step 6: Generate report
        report = self.generate_report(base_df, updated_df, final_df if self.yaml_path else None)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
        
        return final_df


def main():
    """Main function to run the parameter updater."""
    parser = argparse.ArgumentParser(description='Update parameter dataset and match with YAML configuration')
    parser.add_argument('--base-csv', default='/Users/chenzaowen/Desktop/GS_Param/Param_CSV/Full_Param.csv', help='Path to base parameter CSV file')
    parser.add_argument('--yaml-config', help='Path to YAML configuration file')
    parser.add_argument('--update-dir', default='updates', help='Directory containing update CSV files')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Initialize and run updater
    updater = ParameterUpdater(
        base_csv_path=args.base_csv,
        yaml_path=args.yaml_config,
        update_dir=args.update_dir
    )
    
    try:
        final_dataset = updater.run_full_pipeline()
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final dataset: {len(final_dataset)} parameters")
        print(f"📁 Check output files in: {os.getcwd()}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())