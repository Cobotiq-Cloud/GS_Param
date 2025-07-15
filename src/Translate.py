#!/usr/bin/env python3
import pandas as pd
from googletrans import Translator
import json, os, time, sys, argparse
from tqdm import tqdm
import re
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Translate cleaned CSV files from Chinese to English')
    parser.add_argument('input_path', help='Input CSV file path or directory')
    parser.add_argument('-o', '--output', help='Output translated CSV file path (only for single-file mode)', default=None)
    parser.add_argument('-g', '--glossary', help='Glossary file path', default='glossary_zh2en.txt')
    parser.add_argument('-c', '--cache', help='Translation cache file path', default='translation_cache.json')
    parser.add_argument('-b', '--batch-size', help='Translation batch size', type=int, default=20)
    parser.add_argument('--max-retries', help='Maximum retries for translation', type=int, default=3)
    parser.add_argument('--retry-delay', help='Delay between retries (seconds)', type=int, default=5)
    parser.add_argument('--min-text-length', help='Minimum text length to translate', type=int, default=2)
    parser.add_argument('--skip-columns', help='Comma-separated list of column names/numbers to skip', default='')
    parser.add_argument('--file-pattern', help='File pattern to match (e.g., *_cleaned.csv)', default='*.csv')
    parser.add_argument('--skip-existing', help='Skip files that already have translated versions', action='store_true')
    
    return parser.parse_args()

def generate_output_filename(input_file, custom_output=None):
    """Generate output filename based on input filename"""
    if custom_output:
        return custom_output
    
    input_path = Path(input_file)
    
    # If input ends with _cleaned, replace with _en
    if input_path.stem.endswith('_cleaned'):
        stem = input_path.stem[:-8]  # Remove '_cleaned'
        output_name = stem + '_en' + input_path.suffix
    else:
        output_name = input_path.stem + '_en' + input_path.suffix
    
    output_path = input_path.parent / output_name
    return str(output_path)

def contains_chinese(text):
    """Check if text contains Chinese characters"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def should_translate(text, glossary, cache, min_length=2):
    """Determine if text should be translated"""
    if not text or str(text).strip() == '':
        return False
    
    text_str = str(text).strip()
    
    # Skip if already translated
    if text_str in glossary or text_str in cache:
        return False
    
    # Skip if no Chinese characters
    if not contains_chinese(text_str):
        return False
    
    # Skip very short strings
    if len(text_str) < min_length:
        return False
    
    # Skip pure numbers
    if text_str.replace('.', '').replace('-', '').isdigit():
        return False
    
    # Skip URLs and paths (but allow partial URLs in descriptions)
    if (text_str.startswith(('http://', 'https://', 'www.', 'ftp://')) or
        (text_str.startswith('/') and '/' in text_str[1:] and not contains_chinese(text_str))):
        return False
    
    # Skip very short alphanumeric strings
    if text_str.isalnum() and len(text_str) <= 3:
        return False
    
    return True

def load_glossary(glossary_file):
    """Load glossary from file"""
    glossary = {}
    if os.path.exists(glossary_file):
        print(f"📚 Loading glossary from: {glossary_file}")
        try:
            with open(glossary_file, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ',' in line:
                        try:
                            src, tgt = [p.strip() for p in line.split(',', 1)]
                            if src and tgt:
                                glossary[src] = tgt
                        except Exception as e:
                            print(f"   ⚠️  Warning: Error parsing glossary line {line_num}: {e}")
        except Exception as e:
            print(f"   ⚠️  Warning: Error reading glossary file: {e}")
    else:
        print(f"📚 Glossary file not found: {glossary_file}")
    
    return glossary

def load_cache(cache_file):
    """Load translation cache from file"""
    if os.path.exists(cache_file):
        print(f"💾 Loading translation cache from: {cache_file}")
        try:
            with open(cache_file, encoding='utf-8') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"   ⚠️  Warning: Error reading cache file: {e}")
            cache = {}
    else:
        print(f"💾 Cache file not found, starting with empty cache: {cache_file}")
        cache = {}
    
    return cache

def save_cache(cache, cache_file):
    """Save translation cache to file"""
    print(f"💾 Saving translation cache to: {cache_file}")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"   ⚠️  Warning: Error saving cache: {e}")

def get_columns_to_skip(skip_columns_str, df_columns):
    """Parse skip columns argument and return set of column names"""
    skip_set = set()
    if not skip_columns_str:
        return skip_set
    
    for item in skip_columns_str.split(','):
        item = item.strip()
        if not item:
            continue
        
        # Check if it's a number (column index)
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(df_columns):
                skip_set.add(df_columns[idx])
        else:
            # It's a column name
            if item in df_columns:
                skip_set.add(item)
            else:
                print(f"   ⚠️  Warning: Column '{item}' not found in CSV")
    
    return skip_set

def find_csv_files(directory, pattern='*.csv'):
    """Find all CSV files in directory recursively"""
    csv_files = []
    directory_path = Path(directory)
    
    # Use glob to find files matching the pattern
    for csv_file in directory_path.rglob(pattern):
        if csv_file.is_file():
            csv_files.append(str(csv_file))
    
    return sorted(csv_files)

def has_chinese_content(csv_file):
    """Quickly check if CSV file has Chinese content"""
    try:
        # Read first few rows to check for Chinese content
        df_sample = pd.read_csv(csv_file, nrows=10, dtype=str, na_filter=False, encoding='utf-8')
        for col in df_sample.columns:
            for val in df_sample[col].values:
                if contains_chinese(str(val)):
                    return True
        return False
    except Exception as e:
        print(f"   ⚠️  Warning: Could not check Chinese content in {csv_file}: {e}")
        return True  # Assume it has Chinese content if we can't check

def translate_csv_file(input_csv, output_csv, glossary, cache, args, translator=None):
    """Translate a single CSV file"""
    print(f"\n🔄 Processing: {input_csv}")
    
    # Check if output already exists and skip if requested
    if args.skip_existing and os.path.exists(output_csv):
        print(f"   ⏭️  Skipping (output exists): {output_csv}")
        return True, cache
    
    # Load the cleaned CSV
    try:
        df = pd.read_csv(input_csv, 
                        dtype=str,
                        na_filter=False,
                        keep_default_na=False,
                        encoding='utf-8')
        
        # Ensure all data is string type
        df = df.astype(str)
        df = df.replace('nan', '')
        df = df.replace('None', '')
        
        print(f"   ✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return False, cache
    
    # Quick check for Chinese content
    if not has_chinese_content(input_csv):
        print(f"   ⏭️  No Chinese content found, copying file...")
        try:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"   ✅ File copied successfully")
            return True, cache
        except Exception as e:
            print(f"   ❌ Error copying file: {e}")
            return False, cache
    
    # Determine columns to skip
    skip_columns = get_columns_to_skip(args.skip_columns, df.columns)
    
    # Collect texts for translation
    all_values = []
    for col in df.columns:
        if col in skip_columns:
            continue
        try:
            col_values = df[col].values.tolist()
            all_values.extend(col_values)
        except Exception as e:
            print(f"   ⚠️  Warning: Error processing column '{col}': {e}")
            continue
    
    # Get unique texts that need translation
    texts_to_translate = []
    for text in set(all_values):
        if should_translate(text, glossary, cache, args.min_text_length):
            clean_text = str(text).strip()
            # Remove problematic quotes
            clean_text = re.sub(r'^"+|"+$', '', clean_text)
            clean_text = clean_text.replace('""', '"')
            if clean_text and clean_text not in texts_to_translate:
                texts_to_translate.append(clean_text)
    
    print(f"   🔤 Texts to translate: {len(texts_to_translate)}")
    
    # Translate new texts
    if texts_to_translate and translator:
        translation_errors = 0
        
        for i in tqdm(range(0, len(texts_to_translate), args.batch_size), 
                     desc=f"Translating {Path(input_csv).name}", 
                     leave=False):
            batch = texts_to_translate[i:i+args.batch_size]
            
            for attempt in range(1, args.max_retries + 1):
                try:
                    # Clean batch
                    clean_batch = []
                    for text in batch:
                        if text and str(text).strip():
                            clean_text = str(text).strip()
                            # Additional cleaning
                            clean_text = re.sub(r'["""]', '"', clean_text)
                            if clean_text:
                                clean_batch.append(clean_text)
                    
                    if not clean_batch:
                        break
                    
                    # Translate
                    results = translator.translate(clean_batch, src='zh-cn', dest='en')
                    
                    if not isinstance(results, list):
                        results = [results]
                    
                    # Store results
                    for original, result in zip(clean_batch, results):
                        try:
                            if hasattr(result, 'text') and result.text:
                                cache[original] = result.text
                            else:
                                cache[original] = str(result) if result else original
                        except Exception as e:
                            cache[original] = original
                            translation_errors += 1
                    
                    break
                    
                except Exception as e:
                    if attempt < args.max_retries:
                        time.sleep(args.retry_delay)
                    else:
                        # Keep original texts
                        for text in clean_batch:
                            if text not in cache:
                                cache[text] = text
                        translation_errors += len(clean_batch)
    
    # Apply translations
    def get_translation(text, column_name):
        """Get translation for a text"""
        if column_name in skip_columns:
            return text
        
        if not text or str(text).strip() == '':
            return ''
        
        text_str = str(text).strip()
        clean_text = re.sub(r'^"+|"+$', '', text_str).replace('""', '"')
        
        # Check glossary first
        if text_str in glossary:
            return glossary[text_str]
        if clean_text in glossary:
            return glossary[clean_text]
        
        # Check cache
        if text_str in cache:
            return cache[text_str]
        if clean_text in cache:
            return cache[clean_text]
        
        # Return original if no translation
        return text_str
    
    # Apply translations
    df_translated = df.copy()
    translation_applied_count = 0
    
    for col in df.columns:
        try:
            if col not in skip_columns:
                original_values = df[col].values
                translated_values = [get_translation(val, col) for val in original_values]
                df_translated[col] = translated_values
                
                # Count actual translations applied
                for orig, trans in zip(original_values, translated_values):
                    if orig != trans:
                        translation_applied_count += 1
            else:
                df_translated[col] = df[col]
                
        except Exception as e:
            print(f"   ⚠️  Warning: Error translating column '{col}': {e}")
            df_translated[col] = df[col]
    
    # Save result
    try:
        df_translated.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {output_csv} ({translation_applied_count} translations applied)")
        return True, cache
    except Exception as e:
        print(f"   ❌ Error saving translated CSV: {e}")
        return False, cache

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("="*80)
    print("CSV Translation Script - Enhanced with Directory Support")
    print("="*80)
    print(f"Input path:       {args.input_path}")
    print(f"Glossary:         {args.glossary}")
    print(f"Cache:            {args.cache}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Min length:       {args.min_text_length}")
    print(f"File pattern:     {args.file_pattern}")
    print(f"Skip existing:    {args.skip_existing}")
    if args.skip_columns:
        print(f"Skip columns:     {args.skip_columns}")
    print("="*80)
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path not found: {args.input_path}")
        sys.exit(1)
    
    # Load glossary & cache
    glossary = load_glossary(args.glossary)
    cache = load_cache(args.cache)
    
    print(f"\n📊 Translation resources:")
    print(f"   Glossary entries: {len(glossary)}")
    print(f"   Cached translations: {len(cache)}")
    
    # Initialize translator
    translator = None
    try:
        translator = Translator(service_urls=['translate.googleapis.com'])
        print(f"   ✅ Translator initialized successfully")
    except Exception as e:
        print(f"   ❌ Error initializing translator: {e}")
        sys.exit(1)
    
    # Determine if processing single file or directory
    if os.path.isfile(args.input_path):
        # Single file mode
        output_csv = generate_output_filename(args.input_path, args.output)
        print(f"\n🔄 Single file mode")
        print(f"   Output: {output_csv}")
        
        success, cache = translate_csv_file(args.input_path, output_csv, glossary, cache, args, translator)
        
        if success:
            print(f"\n✅ Translation completed successfully!")
        else:
            print(f"\n❌ Translation failed!")
            sys.exit(1)
    
    else:
        # Directory mode
        print(f"\n🔄 Directory mode - searching for CSV files...")
        csv_files = find_csv_files(args.input_path, args.file_pattern)
        
        if not csv_files:
            print(f"❌ No CSV files found matching pattern '{args.file_pattern}' in {args.input_path}")
            sys.exit(1)
        
        print(f"📁 Found {len(csv_files)} CSV files to process")
        
        # Process each file
        successful_translations = 0
        failed_translations = 0
        skipped_translations = 0
        
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            output_csv = generate_output_filename(csv_file)
            
            # Check if should skip
            if args.skip_existing and os.path.exists(output_csv):
                skipped_translations += 1
                continue
            
            success, cache = translate_csv_file(csv_file, output_csv, glossary, cache, args, translator)
            
            if success:
                successful_translations += 1
            else:
                failed_translations += 1
            
            # Save cache periodically
            if (successful_translations + failed_translations) % 5 == 0:
                save_cache(cache, args.cache)
        
        print(f"\n" + "="*80)
        print("📊 Directory Processing Summary")
        print("="*80)
        print(f"✅ Successfully processed: {successful_translations}")
        print(f"❌ Failed to process:      {failed_translations}")
        print(f"⏭️  Skipped (existing):     {skipped_translations}")
        print(f"📁 Total files found:      {len(csv_files)}")
    
    # Final cache save
    save_cache(cache, args.cache)
    
    print(f"\n🎉 Processing complete!")
    print(f"💾 Final cache entries: {len(cache)}")

if __name__ == "__main__":
    main()