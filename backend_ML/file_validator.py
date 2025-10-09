# -*- coding: utf-8 -*-
"""
File Validation Module
Validates uploaded files for type, size, and malicious content
"""

import os
import pandas as pd

# Optional import for file type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available. File type detection will be limited.")
import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib
import mimetypes
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class FileValidator:
    """
    Comprehensive file validation for uploaded transaction data
    """
    
    def __init__(self):
        """Initialize file validator with security settings"""
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
        self.allowed_mime_types = {
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json',
            'application/octet-stream'  # For parquet files
        }
        
        # Required columns for transaction data
        self.required_columns = [
            'timestamp', 'from_account', 'to_account', 'amount_paid', 'amount_received'
        ]
        
        # Optional but recommended columns
        self.recommended_columns = [
            'from_bank', 'to_bank', 'payment_currency', 'receiving_currency', 
            'payment_format', 'is_laundering'
        ]
        
        # Suspicious patterns to detect
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:text/html',  # Data URLs
            r'vbscript:',  # VBScript
            r'on\w+\s*=',  # Event handlers
        ]
    
    def validate_file(self, file_path: str) -> Dict[str, any]:
        """
        Comprehensive file validation
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'data_quality': {}
        }
        
        try:
            # Basic file existence check
            if not os.path.exists(file_path):
                result['errors'].append(f"File not found: {file_path}")
                return result
            
            # Get file info
            file_info = self._get_file_info(file_path)
            result['file_info'] = file_info
            
            # Size validation
            if file_info['size'] > self.max_file_size:
                result['errors'].append(f"File too large: {file_info['size_mb']:.2f}MB (max: {self.max_file_size/1024/1024:.0f}MB)")
            
            # Extension validation
            if file_info['extension'] not in self.allowed_extensions:
                result['errors'].append(f"Invalid file extension: {file_info['extension']}")
            
            # MIME type validation
            if file_info['mime_type'] not in self.allowed_mime_types:
                result['warnings'].append(f"Unexpected MIME type: {file_info['mime_type']}")
            
            # Content validation
            content_validation = self._validate_file_content(file_path)
            result['errors'].extend(content_validation['errors'])
            result['warnings'].extend(content_validation['warnings'])
            
            # Data structure validation (for supported formats)
            if file_info['extension'] in ['.csv', '.xlsx', '.xls', '.json', '.parquet']:
                data_validation = self._validate_data_structure(file_path)
                result['data_quality'] = data_validation
                result['errors'].extend(data_validation['errors'])
                result['warnings'].extend(data_validation['warnings'])
            
            # Security checks
            security_validation = self._validate_security(file_path)
            result['errors'].extend(security_validation['errors'])
            result['warnings'].extend(security_validation['warnings'])
            
            # Overall validation result
            result['is_valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _get_file_info(self, file_path: str) -> Dict[str, any]:
        """Extract basic file information"""
        file_path = Path(file_path)
        stat = file_path.stat()
        
        # Get MIME type using python-magic or fallback
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or 'unknown'
        else:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or 'unknown'
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(file_path)
        
        return {
            'name': file_path.name,
            'extension': file_path.suffix.lower(),
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'mime_type': mime_type,
            'hash': file_hash,
            'modified_time': stat.st_mtime
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return "unknown"
    
    def _validate_file_content(self, file_path: str) -> Dict[str, List[str]]:
        """Validate file content for basic integrity"""
        errors = []
        warnings = []
        
        try:
            # Check if file is readable
            with open(file_path, 'rb') as f:
                # Read first 1KB to check for binary content in text files
                first_chunk = f.read(1024)
                
                # Check for null bytes (indicates binary file)
                if b'\x00' in first_chunk:
                    if Path(file_path).suffix.lower() in ['.csv', '.json']:
                        errors.append("File contains binary data but has text extension")
                
                # Check for suspicious patterns
                try:
                    text_content = first_chunk.decode('utf-8', errors='ignore')
                    for pattern in self.suspicious_patterns:
                        import re
                        if re.search(pattern, text_content, re.IGNORECASE):
                            errors.append(f"Suspicious content detected: {pattern}")
                except:
                    pass
                    
        except Exception as e:
            errors.append(f"Cannot read file content: {str(e)}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_data_structure(self, file_path: str) -> Dict[str, any]:
        """Validate the structure and content of data files"""
        errors = []
        warnings = []
        data_info = {}
        
        try:
            # Load data based on file type
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == '.csv':
                df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=1000)
            elif extension == '.json':
                df = pd.read_json(file_path)
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                return {'errors': ['Unsupported file format'], 'warnings': [], 'data_info': {}}
            
            # Basic data info
            data_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Check for required columns
            missing_required = set(self.required_columns) - set(df.columns)
            if missing_required:
                errors.append(f"Missing required columns: {list(missing_required)}")
            
            # Check for recommended columns
            missing_recommended = set(self.recommended_columns) - set(df.columns)
            if missing_recommended:
                warnings.append(f"Missing recommended columns: {list(missing_recommended)}")
            
            # Data quality checks
            quality_issues = self._check_data_quality(df)
            errors.extend(quality_issues['errors'])
            warnings.extend(quality_issues['warnings'])
            
            # Add quality metrics to data_info
            data_info.update(quality_issues['metrics'])
            
        except Exception as e:
            errors.append(f"Error reading data file: {str(e)}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'data_info': data_info
        }
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check data quality and consistency"""
        errors = []
        warnings = []
        metrics = {}
        
        # Check for empty dataframe
        if df.empty:
            errors.append("File contains no data")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            warnings.append(f"Completely empty columns: {empty_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
        
        metrics['duplicate_rows'] = duplicate_count
        metrics['empty_columns'] = len(empty_cols)
        
        # Check for suspicious data patterns
        if 'amount_paid' in df.columns:
            amount_col = df['amount_paid']
            if amount_col.dtype in ['object', 'string']:
                warnings.append("Amount column contains non-numeric data")
            
            # Check for negative amounts (might be valid for some transaction types)
            numeric_amounts = pd.to_numeric(amount_col, errors='coerce')
            negative_count = (numeric_amounts < 0).sum()
            if negative_count > 0:
                warnings.append(f"Found {negative_count} negative amounts")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            timestamp_col = df['timestamp']
            try:
                pd.to_datetime(timestamp_col, errors='raise')
            except:
                warnings.append("Timestamp column contains invalid date formats")
        
        # Check for suspiciously large datasets
        if len(df) > 1000000:  # 1M rows
            warnings.append(f"Very large dataset: {len(df):,} rows")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_security(self, file_path: str) -> Dict[str, List[str]]:
        """Perform security-related validations"""
        errors = []
        warnings = []
        
        try:
            # Check file path for directory traversal attempts
            if '..' in str(file_path) or '/' in str(file_path) or '\\' in str(file_path):
                errors.append("Suspicious file path detected")
            
            # Check for executable file extensions
            executable_extensions = {'.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js'}
            if Path(file_path).suffix.lower() in executable_extensions:
                errors.append("File has executable extension")
            
            # Check file permissions
            file_path = Path(file_path)
            if file_path.exists():
                # Check if file is world-writable (security risk)
                stat = file_path.stat()
                if stat.st_mode & 0o002:  # World writable
                    warnings.append("File is world-writable")
            
        except Exception as e:
            warnings.append(f"Security check error: {str(e)}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def get_validation_report(self, validation_result: Dict[str, any]) -> str:
        """Generate a human-readable validation report"""
        report = []
        report.append("=" * 60)
        report.append("FILE VALIDATION REPORT")
        report.append("=" * 60)
        
        # File info
        file_info = validation_result.get('file_info', {})
        report.append(f"File: {file_info.get('name', 'Unknown')}")
        report.append(f"Size: {file_info.get('size_mb', 0):.2f} MB")
        report.append(f"Type: {file_info.get('mime_type', 'Unknown')}")
        report.append(f"Hash: {file_info.get('hash', 'Unknown')[:16]}...")
        
        # Validation status
        status = "✅ VALID" if validation_result['is_valid'] else "❌ INVALID"
        report.append(f"\nStatus: {status}")
        
        # Errors
        if validation_result['errors']:
            report.append(f"\n❌ ERRORS ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                report.append(f"  • {error}")
        
        # Warnings
        if validation_result['warnings']:
            report.append(f"\n⚠️  WARNINGS ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings']:
                report.append(f"  • {warning}")
        
        # Data quality info
        data_quality = validation_result.get('data_quality', {})
        if data_quality:
            data_info = data_quality.get('data_info', {})
            if data_info:
                report.append(f"\n📊 DATA INFO:")
                report.append(f"  • Rows: {data_info.get('rows', 0):,}")
                report.append(f"  • Columns: {data_info.get('columns', 0)}")
                report.append(f"  • Memory usage: {data_info.get('memory_usage_mb', 0):.2f} MB")
                report.append(f"  • Duplicate rows: {data_info.get('duplicate_rows', 0)}")
                report.append(f"  • Empty columns: {data_info.get('empty_columns', 0)}")
        
        report.append("=" * 60)
        return "\n".join(report)


def validate_uploaded_file(file_path: str) -> Tuple[bool, str, Dict[str, any]]:
    """
    Convenience function to validate an uploaded file
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, report_text, validation_result)
    """
    validator = FileValidator()
    result = validator.validate_file(file_path)
    report = validator.get_validation_report(result)
    
    return result['is_valid'], report, result


if __name__ == "__main__":
    # Test the file validator
    print("Testing File Validator...")
    
    # Create a test CSV file
    test_data = {
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'from_account': ['ACC001', 'ACC002', 'ACC003'],
        'to_account': ['ACC002', 'ACC003', 'ACC001'],
        'amount_paid': [1000, 2000, 1500],
        'amount_received': [1000, 2000, 1500],
        'from_bank': ['Bank A', 'Bank B', 'Bank A'],
        'to_bank': ['Bank B', 'Bank C', 'Bank A'],
        'payment_currency': ['USD', 'EUR', 'GBP'],
        'is_laundering': [0, 1, 0]
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = "test_transactions.csv"
    test_df.to_csv(test_file, index=False)
    
    # Validate the test file
    is_valid, report, result = validate_uploaded_file(test_file)
    
    print(report)
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
    
    # Clean up
    os.remove(test_file)
