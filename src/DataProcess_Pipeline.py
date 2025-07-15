#!/usr/bin/env python3
"""
Enhanced Parameter Dataset Updater

This script updates the Full_Param dataset with values from translated CSV files,
then matches parameters with a YAML configuration file.

Features:
- Works with translated CSV files from the Translated directory
- Updates Full_Param.csv with any number of translated CSV files
- Automatically detects parameter path columns
- Matches updated dataset with YAML configuration
- Generates comprehensive reports in Results directory
- Handles both English and Chinese parameter files

Usage:    
    python DataProcess_Pipeline.py \
    --base-csv /Users/chenzaowen/Desktop/GS_Param/Translated/Full_Param_en.csv \
    --yaml-config /Users/chenzaowen/Desktop/GS_Param/Yaml/A4-00-14p-20250417_user_config.yaml \
    --translated-dir Translated \
    --output-dir Results

Arguments:
    --base-csv : Path to base parameter CSV file
    --yaml-config : Path to YAML configuration file
    --translated-dir : Directory containing translated CSV files (default: Translated)
    --output-dir : Output directory for results (default: Results)
    --file-pattern : File pattern to match (default: *_en.csv)
    --preserve-originals : Keep original language columns alongside translations
    -h, --help : Show help message
"""

import pandas as pd
import yaml
import os
import argparse
import glob
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedParameterUpdater:
    """Enhanced class to handle parameter dataset updates with translated files."""
    
    def __init__(self, base_csv_path: str, yaml_path: str = None, 
                 translated_dir: str = "Translated", output_dir: str = "Results"):
        """
        Initialize the Enhanced Parameter Updater.
        
        Args:
            base_csv_path: Path to the main Full_Param.csv file
            yaml_path: Path to YAML configuration file (optional)
            translated_dir: Directory containing translated CSV files
            output_dir: Output directory for results
        """
        self.base_csv_path = base_csv_path
        self.yaml_path = yaml_path
        self.translated_dir = translated_dir
        self.output_dir = output_dir
        self.param_column_mapping = {}
        self.update_statistics = {
            'files_processed': 0,
            'parameters_updated': 0,
            'new_parameters_added': 0,
            'translation_matches': 0
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def detect_parameter_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the parameter path column in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Column name containing parameter paths
        """
        # Enhanced patterns for parameter path columns (including English translations)
        path_patterns = [
            '参数位置', 'parameter_path', 'param_path', 'path', 'location',
            '1参数位置和名称', '/strategy/', 'config_path', 'parameter_location',
            'param_location', 'configuration_path', 'setting_path'
        ]
        
        for col in df.columns:
            # Check if column name matches patterns
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in path_patterns):
                # Verify column contains path-like values
                sample_values = df[col].dropna().head(10)
                path_like_count = sum(1 for val in sample_values 
                                    if str(val).strip().startswith('/') or 
                                    '/strategy/' in str(val) or
                                    str(val).count('/') >= 2)
                if path_like_count >= len(sample_values) * 0.5:  # 50% threshold
                    logger.info(f"Detected parameter column: {col}")
                    return col
                    
        # If no obvious column found, check for columns with path-like values
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(20)
                if len(sample_values) == 0:
                    continue
                    
                path_count = sum(1 for val in sample_values 
                               if str(val).strip().startswith('/') or 
                               '/strategy/' in str(val) or
                               str(val).count('/') >= 2)
                               
                if path_count >= len(sample_values) * 0.6:  # 60% threshold
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
    
    def find_translated_files(self, file_pattern: str = "*_en.csv") -> List[str]:
        """
        Find all translated CSV files in the translated directory.
        
        Args:
            file_pattern: Pattern to match translated files
            
        Returns:
            List of file paths
        """
        if not os.path.exists(self.translated_dir):
            logger.warning(f"Translated directory {self.translated_dir} does not exist")
            return []
            
        # Search recursively for translated files
        search_pattern = os.path.join(self.translated_dir, "**", file_pattern)
        csv_files = glob.glob(search_pattern, recursive=True)
        
        logger.info(f"Found {len(csv_files)} translated files matching pattern '{file_pattern}'")
        return sorted(csv_files)
    
    def load_translated_files(self, file_pattern: str = "*_en.csv") -> List[Tuple[str, pd.DataFrame]]:
        """
        Load all translated CSV files from the translated directory.
        
        Args:
            file_pattern: Pattern to match translated files
            
        Returns:
            List of tuples (filename, DataFrame)
        """
        translated_files = []
        csv_files = self.find_translated_files(file_pattern)
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df.columns = df.columns.str.strip()
                filename = os.path.relpath(file_path, self.translated_dir)
                translated_files.append((filename, df))
                logger.info(f"Loaded translated file: {filename} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
                
        return translated_files
    
    def enhanced_column_mapping(self, update_col: str, base_columns: List[str]) -> Optional[str]:
        """
        Enhanced column mapping for translated files.
        
        Args:
            update_col: Column name from translated file
            base_columns: List of column names in base dataset
            
        Returns:
            Mapped column name or None if no mapping found
        """
        # Enhanced column mappings including English translations
        column_mappings = {
            # Chinese to standard mappings
            '参数名称': ['4参数中文名称', 'chinese_name', 'name', 'parameter_name'],
            '参数释义': ['5参数释义', 'description', 'explanation', 'parameter_description'],
            '默认值': ['default_value', 'default', 'value', 'default_val'],
            '功能分类1': ['6功能分类1', 'category1', 'function_category', 'category_1'],
            '功能分类2': ['7功能分类2', 'category2', 'sub_category', 'category_2'],
            '功能说明': ['8功能说明', 'function_description', 'notes', 'function_notes'],
            
            # English translations to standard mappings
            'parameter_name': ['4参数中文名称', 'chinese_name', 'name'],
            'description': ['5参数释义', 'parameter_description', 'explanation'],
            'parameter_description': ['5参数释义', 'description', 'explanation'],
            'function_category': ['6功能分类1', 'category1', 'category_1'],
            'sub_category': ['7功能分类2', 'category2', 'category_2'],
            'function_description': ['8功能说明', 'function_notes', 'notes'],
            'category1': ['6功能分类1', 'function_category', 'category_1'],
            'category2': ['7功能分类2', 'sub_category', 'category_2'],
        }
        
        # Direct match first
        if update_col in base_columns:
            return update_col
            
        # Check explicit mappings
        update_col_clean = update_col.strip()
        if update_col_clean in column_mappings:
            for candidate in column_mappings[update_col_clean]:
                if candidate in base_columns:
                    return candidate
        
        # Reverse mapping check
        for base_key, candidates in column_mappings.items():
            if update_col_clean in candidates:
                if base_key in base_columns:
                    return base_key
        
        # Fuzzy match with enhanced logic
        update_col_lower = update_col_clean.lower()
        
        # Remove common prefixes/suffixes for better matching
        clean_update = re.sub(r'^(column_|col_|param_|parameter_)', '', update_col_lower)
        clean_update = re.sub(r'(_en|_english|_translated)$', '', clean_update)
        
        best_match = None
        best_score = 0
        
        for base_col in base_columns:
            base_col_lower = base_col.lower()
            clean_base = re.sub(r'^(column_|col_|param_|parameter_|\d+)', '', base_col_lower)
            
            # Calculate similarity score
            score = 0
            if clean_update == clean_base:
                score = 100
            elif clean_update in clean_base or clean_base in clean_update:
                score = 80
            elif any(word in clean_base for word in clean_update.split('_') if len(word) > 2):
                score = 60
            
            if score > best_score:
                best_score = score
                best_match = base_col
        
        if best_score >= 60:  # Minimum similarity threshold
            logger.info(f"Fuzzy matched '{update_col}' -> '{best_match}' (score: {best_score})")
            return best_match
                
        return None
    
    def update_dataset_with_translations(self, base_df: pd.DataFrame, 
                                       translated_files: List[Tuple[str, pd.DataFrame]], 
                                       preserve_originals: bool = False) -> pd.DataFrame:
        """
        Update the base dataset with values from translated files.
        
        Args:
            base_df: Main parameter dataset
            translated_files: List of (filename, DataFrame) tuples
            preserve_originals: Whether to keep original language columns
            
        Returns:
            Updated DataFrame
        """
        updated_df = base_df.copy()
        update_log = []
        
        # Detect parameter column in base dataset
        base_param_col = self.detect_parameter_column(base_df)
        
        for filename, trans_df in translated_files:
            try:
                logger.info(f"Processing translated file: {filename}")
                
                # Detect parameter column in translated file
                trans_param_col = self.detect_parameter_column(trans_df)
                
                # Get parameter paths that exist in both datasets
                base_paths = set(updated_df[base_param_col].dropna().astype(str))
                trans_paths = set(trans_df[trans_param_col].dropna().astype(str))
                common_paths = base_paths.intersection(trans_paths)
                
                logger.info(f"  - Found {len(common_paths)} matching parameters")
                self.update_statistics['translation_matches'] += len(common_paths)
                
                if not common_paths:
                    logger.warning(f"  - No matching parameters found in {filename}")
                    continue
                
                # Update matching rows
                for path in common_paths:
                    base_mask = updated_df[base_param_col].astype(str) == path
                    base_indices = updated_df[base_mask].index
                    
                    trans_mask = trans_df[trans_param_col].astype(str) == path
                    trans_rows = trans_df[trans_mask]
                    
                    if len(base_indices) > 0 and len(trans_rows) > 0:
                        base_idx = base_indices[0]
                        trans_row = trans_rows.iloc[0]
                        updated_columns = []
                        
                        # Update each column that has a value in translated file
                        for col in trans_df.columns:
                            if col == trans_param_col:
                                continue
                                
                            if pd.notna(trans_row[col]) and str(trans_row[col]).strip():
                                # Map translated column to base column
                                target_col = self.enhanced_column_mapping(col, updated_df.columns)
                                
                                if target_col:
                                    old_value = updated_df.loc[base_idx, target_col]
                                    new_value = str(trans_row[col]).strip()
                                    
                                    # Only update if the new value is different and not empty
                                    if str(old_value).strip() != new_value and new_value:
                                        if preserve_originals and not pd.isna(old_value) and str(old_value).strip():
                                            # Create backup column for original
                                            backup_col = f"{target_col}_original"
                                            if backup_col not in updated_df.columns:
                                                updated_df[backup_col] = updated_df[target_col]
                                        
                                        updated_df.loc[base_idx, target_col] = new_value
                                        updated_columns.append(f"{target_col}: '{old_value}' -> '{new_value}'")
                                        self.update_statistics['parameters_updated'] += 1
                                        
                                else:
                                    # Add new column with English suffix
                                    new_col_name = f"{col}_en" if not col.endswith('_en') else col
                                    if new_col_name not in updated_df.columns:
                                        updated_df[new_col_name] = None
                                    updated_df.loc[base_idx, new_col_name] = str(trans_row[col]).strip()
                                    updated_columns.append(f"Added {new_col_name}: '{trans_row[col]}'")
                                    self.update_statistics['new_parameters_added'] += 1
                        
                        if updated_columns:
                            update_log.append({
                                'file': filename,
                                'parameter': path,
                                'updates': updated_columns
                            })
                
                self.update_statistics['files_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        # Save update log
        self.save_update_log(update_log)
        logger.info(f"Dataset updated with changes from {len(translated_files)} files")
        
        return updated_df
    
    def extract_yaml_paths(self, yaml_file: str) -> List[str]:
        """
        Extract all parameter paths from YAML file with enhanced path detection.
        """
        def extract_paths(obj, prefix=""):
            paths = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        paths.extend(extract_paths(value, current_path))
                    else:
                        # Add the path with leading slash
                        full_path = f"/{current_path}" if not current_path.startswith('/') else current_path
                        paths.append(full_path)
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
        """Enhanced YAML matching with better statistics."""
        if not yaml_paths:
            logger.warning("No YAML paths provided, returning full dataset")
            return updated_df
            
        param_col = self.detect_parameter_column(updated_df)
        dataset_paths = set(updated_df[param_col].dropna().astype(str))
        yaml_paths_set = set(yaml_paths)
        
        # Find matches
        matched_paths = dataset_paths.intersection(yaml_paths_set)
        matched_df = updated_df[updated_df[param_col].isin(matched_paths)].copy()
        
        # Calculate detailed statistics
        total_yaml = len(yaml_paths)
        total_dataset = len(dataset_paths)
        total_matched = len(matched_paths)
        
        match_percentage = (total_matched / total_yaml * 100) if total_yaml > 0 else 0
        coverage_percentage = (total_matched / total_dataset * 100) if total_dataset > 0 else 0
        
        logger.info(f"YAML matching results:")
        logger.info(f"  - Total YAML parameters: {total_yaml}")
        logger.info(f"  - Total dataset parameters: {total_dataset}")
        logger.info(f"  - Matched parameters: {total_matched}")
        logger.info(f"  - YAML match percentage: {match_percentage:.1f}%")
        logger.info(f"  - Dataset coverage: {coverage_percentage:.1f}%")
        
        # Save unmatched parameters for analysis
        unmatched_yaml = yaml_paths_set - matched_paths
        unmatched_dataset = dataset_paths - matched_paths
        
        if unmatched_yaml:
            unmatched_yaml_df = pd.DataFrame({'unmatched_yaml_path': sorted(unmatched_yaml)})
            yaml_output_path = os.path.join(self.output_dir, 'unmatched_yaml_parameters.csv')
            unmatched_yaml_df.to_csv(yaml_output_path, index=False, encoding='utf-8')
            logger.info(f"Saved {len(unmatched_yaml)} unmatched YAML parameters")
        
        if unmatched_dataset:
            unmatched_dataset_df = pd.DataFrame({'unmatched_dataset_path': sorted(unmatched_dataset)})
            dataset_output_path = os.path.join(self.output_dir, 'unmatched_dataset_parameters.csv')
            unmatched_dataset_df.to_csv(dataset_output_path, index=False, encoding='utf-8')
            logger.info(f"Saved {len(unmatched_dataset)} unmatched dataset parameters")
        
        return matched_df
    
    def save_update_log(self, update_log: List[Dict]) -> None:
        """Save detailed update log to file."""
        if update_log:
            log_data = []
            for entry in update_log:
                for update in entry['updates']:
                    log_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'file': entry['file'],
                        'parameter': entry['parameter'],
                        'update': update
                    })
            
            log_df = pd.DataFrame(log_data)
            log_output_path = os.path.join(self.output_dir, 'parameter_update_log.csv')
            log_df.to_csv(log_output_path, index=False, encoding='utf-8')
            logger.info(f"Saved update log with {len(log_data)} changes")
    
    def generate_comprehensive_report(self, original_df: pd.DataFrame, updated_df: pd.DataFrame, 
                                    matched_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive report with translation statistics."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_statistics': self.update_statistics.copy(),
            'original_dataset': {
                'rows': len(original_df),
                'columns': len(original_df.columns),
                'column_names': list(original_df.columns)
            },
            'updated_dataset': {
                'rows': len(updated_df),
                'columns': len(updated_df.columns),
                'new_columns': list(set(updated_df.columns) - set(original_df.columns)),
                'column_names': list(updated_df.columns)
            }
        }
        
        if matched_df is not None:
            yaml_match_rate = (len(matched_df) / len(updated_df) * 100) if len(updated_df) > 0 else 0
            report['matched_dataset'] = {
                'rows': len(matched_df),
                'columns': len(matched_df.columns),
                'yaml_match_rate': f"{yaml_match_rate:.1f}%"
            }
        
        # Save report as JSON
        import json
        report_path = os.path.join(self.output_dir, 'processing_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def run_enhanced_pipeline(self, file_pattern: str = "*_en.csv", 
                            preserve_originals: bool = False) -> pd.DataFrame:
        """
        Run the complete enhanced parameter update and matching pipeline.
        
        Args:
            file_pattern: Pattern to match translated files
            preserve_originals: Whether to preserve original language columns
            
        Returns:
            Final processed DataFrame
        """
        logger.info("Starting enhanced parameter update pipeline...")
        logger.info(f"Input: {self.base_csv_path}")
        logger.info(f"Translated directory: {self.translated_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"File pattern: {file_pattern}")
        
        # Step 1: Load base dataset
        base_df = self.load_base_dataset()
        
        # Step 2: Load translated files
        translated_files = self.load_translated_files(file_pattern)
        
        if not translated_files:
            logger.warning("No translated files found!")
            return base_df
        
        # Step 3: Update dataset with translations
        updated_df = self.update_dataset_with_translations(base_df, translated_files, preserve_originals)
        
        # Step 4: Save updated full dataset
        updated_output_path = os.path.join(self.output_dir, 'Full_Param_Updated_with_Translations.csv')
        updated_df.to_csv(updated_output_path, index=False, encoding='utf-8')
        logger.info(f"Saved updated full dataset: {updated_output_path}")
        
        # Step 5: Match with YAML if provided
        final_df = updated_df
        if self.yaml_path and os.path.exists(self.yaml_path):
            logger.info(f"Processing YAML configuration: {self.yaml_path}")
            yaml_paths = self.extract_yaml_paths(self.yaml_path)
            if yaml_paths:
                final_df = self.match_with_yaml(updated_df, yaml_paths)
                final_output_path = os.path.join(self.output_dir, 'Final_Matched_Parameters.csv')
                final_df.to_csv(final_output_path, index=False, encoding='utf-8')
                logger.info(f"Saved final matched parameters: {final_output_path}")
        
        # Step 6: Generate comprehensive report
        report = self.generate_comprehensive_report(base_df, updated_df, final_df if self.yaml_path else None)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Files processed: {self.update_statistics['files_processed']}")
        logger.info(f"📊 Parameters updated: {self.update_statistics['parameters_updated']}")
        logger.info(f"📊 New parameters added: {self.update_statistics['new_parameters_added']}")
        logger.info(f"📊 Translation matches: {self.update_statistics['translation_matches']}")
        logger.info(f"📊 Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
        logger.info(f"📁 Results saved to: {os.path.abspath(self.output_dir)}")
        
        return final_df


def main():
    """Main function to run the enhanced parameter updater."""
    parser = argparse.ArgumentParser(
        description='Enhanced Parameter Dataset Updater with Translation Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default directories
  python %(prog)s --base-csv Full_Param.csv --yaml-config config.yaml
  
  # Custom directories
  python %(prog)s --base-csv data/Full_Param.csv --translated-dir MyTranslated --output-dir MyResults
  
  # With original preservation
  python %(prog)s --base-csv Full_Param.csv --preserve-originals
        """
    )
    
    parser.add_argument('--base-csv', required=True,
                       help='Path to base parameter CSV file')
    parser.add_argument('--yaml-config', 
                       help='Path to YAML configuration file (optional)')
    parser.add_argument('--translated-dir', default='Translated',
                       help='Directory containing translated CSV files (default: Translated)')
    parser.add_argument('--output-dir', default='Results',
                       help='Output directory for results (default: Results)')
    parser.add_argument('--file-pattern', default='*_en.csv',
                       help='File pattern to match translated files (default: *_en.csv)')
    parser.add_argument('--preserve-originals', action='store_true',
                       help='Keep original language columns alongside translations')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_csv):
        print(f"❌ Error: Base CSV file not found: {args.base_csv}")
        return 1
    
    if args.yaml_config and not os.path.exists(args.yaml_config):
        print(f"❌ Error: YAML config file not found: {args.yaml_config}")
        return 1
    
    if not os.path.exists(args.translated_dir):
        print(f"❌ Error: Translated directory not found: {args.translated_dir}")
        return 1
    
    # Initialize and run enhanced updater
    updater = EnhancedParameterUpdater(
        base_csv_path=args.base_csv,
        yaml_path=args.yaml_config,
        translated_dir=args.translated_dir,
        output_dir=args.output_dir
    )
    
    try:
        final_dataset = updater.run_enhanced_pipeline(
            file_pattern=args.file_pattern,
            preserve_originals=args.preserve_originals
        )
        
        print(f"\n✅ Enhanced pipeline completed successfully!")
        print(f"📊 Final dataset: {len(final_dataset)} parameters")
        print(f"📁 Check output files in: {os.path.abspath(args.output_dir)}")
        print(f"📈 Updated {updater.update_statistics['parameters_updated']} parameters")
        print(f"📈 Added {updater.update_statistics['new_parameters_added']} new parameter columns")
        
        return 0
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Enhanced pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())