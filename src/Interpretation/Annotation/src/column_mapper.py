"""
@description
Flexible column name matching utility for handling diverse input file formats.
Maps common column name variants to standardized names used throughout the pipeline.

Key features:
- Case-insensitive matching
- Support for common naming conventions (snake_case, camelCase, etc.)
- Clear error messages when required columns are missing

@dependencies
- pandas

@examples
- mapper = ColumnMapper(df)
- gene_col = mapper.get_column('gene')  # Finds 'Gene', 'gene', 'gene_name', etc.
"""

import numbers
import re
from typing import Dict, List, Optional, Set
import pandas as pd


class ColumnMapper:
    """
    Maps flexible column names to standardized names.
    
    Supports case-insensitive matching and common variants for:
    - Gene names: Name, Gene, gene_name, GeneName, etc.
    - Scores: Score, score, Loading, loading, etc.
    - Program IDs: program_id, RowID, topic, Topic, etc.
    - Cell types: cell_type, celltype, cluster, Cluster, etc.
    - Log2 fold change: log2_fc, log2FC, log2_fold_change, etc.
    - FDR/p-values: fdr, FDR, p_value, pval, etc.
    """
    
    # Define column aliases (all lowercase for matching)
    ALIASES: Dict[str, List[str]] = {
        'gene': [
            'gene', 'name', 'gene_name', 'genename', 'gene_symbol', 
            'genesymbol', 'symbol', 'gene_id', 'geneid'
        ],
        'score': [
            'score', 'loading', 'weight', 'value', 'loading_score',
            'loadingscore', 'gene_score', 'genescore'
        ],
        'program_id': [
            'program_id', 'programid', 'rowid', 'row_id',
            'topic', 'topic_id', 'topicid', 'program', 'program_number',
            'topic_number', 'k', 'component', 'factor', 'response_id',
            'response', 'program_name', 'program_label', 'topic_name'
        ],
        'cell_type': [
            'cell_type', 'celltype', 'cluster', 'cell_cluster',
            'cellcluster', 'annotation', 'cell_annotation',
            'cellannotation', 'type', 'celltype_id', 'cluster_id'
        ],
        'log2_fc': [
            'log2_fc', 'log2fc', 'log2_fold_change', 'log2foldchange',
            'log_2_fold_change', 'log2_fc_in_vs_out', 'log2fc_in_vs_out', 'log2_fold',
            'log2fold', 'lfc', 'l2fc', 'fold_change', 'foldchange',
            'fc', 'log2ratio', 'log2_ratio'
        ],
        'fdr': [
            'fdr', 'fdr_corrected', 'q_value', 'qvalue', 'qval', 'q',
            'padj', 'p_adj', 'p_adjusted', 'adjusted_pvalue', 
            'p_value', 'pvalue', 'pval', 'p'
        ],
        'regulator_gene': [
            'grna_target', 'target_gene', 'target_gene_name',
            'target_gene_names', 'perturbation_target', 'gene_target',
            'regulator', 'regulator_gene', 'gene'
        ],
        'raw_p_value': [
            'p_value', 'pvalue', 'p_val', 'pval', 'raw_p_value',
            'raw_pvalue', 'raw_pval', 'nominal_p_value',
            'nominal_pvalue', 'nominal_pval', 'p'
        ],
        'adj_p_value': [
            'adj_pval', 'adj_pvalue', 'adjusted_pval', 'adjusted_pvalue',
            'adjusted_p_value', 'p_adj', 'padj', 'p_adjusted',
            'fdr', 'fdr_corrected', 'q_value', 'qvalue', 'qval', 'q'
        ],
        'significant': [
            'significant', 'is_significant', 'significant_hit',
            'is_sig', 'sig', 'passed_significance'
        ]
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize mapper with a DataFrame.
        
        Args:
            df: DataFrame to map columns for
        """
        self.df = df
        self.columns_lower = {col.lower(): col for col in df.columns}
        self._mapped: Dict[str, str] = {}
    
    def get_column(self, standard_name: str, required: bool = True) -> Optional[str]:
        """
        Find the actual column name that matches the standard name.
        
        Args:
            standard_name: Standard column name (e.g., 'gene', 'score', 'program_id')
            required: If True, raise ValueError when column not found
            
        Returns:
            Actual column name in the DataFrame, or None if not found and not required
            
        Raises:
            ValueError: If required=True and column not found
        """
        # Return cached mapping if available
        if standard_name in self._mapped:
            return self._mapped[standard_name]
        
        # Get aliases for this standard name
        if standard_name not in self.ALIASES:
            if required:
                raise ValueError(f"Unknown standard column name: {standard_name}")
            return None
        
        aliases = self.ALIASES[standard_name]
        
        # Try to find a matching column (case-insensitive)
        for alias in aliases:
            if alias in self.columns_lower:
                actual_col = self.columns_lower[alias]
                self._mapped[standard_name] = actual_col
                return actual_col
        
        if required:
            raise ValueError(
                f"Could not find column for '{standard_name}'. "
                f"Expected one of: {aliases[:5]}... "
                f"Found columns: {sorted(self.df.columns)}"
            )
        
        return None
    
    def get_columns(self, standard_names: List[str], required: bool = True) -> Dict[str, Optional[str]]:
        """
        Find multiple columns at once.
        
        Args:
            standard_names: List of standard column names
            required: If True, raise ValueError for any missing required columns
            
        Returns:
            Dict mapping standard names to actual column names
        """
        result = {}
        missing = []
        
        for name in standard_names:
            try:
                result[name] = self.get_column(name, required=required)
            except ValueError:
                missing.append(name)
                result[name] = None
        
        if missing and required:
            aliases_str = "\n".join([
                f"  - {name}: {', '.join(self.ALIASES.get(name, [])[:5])}"
                for name in missing
            ])
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Expected aliases:\n{aliases_str}\n"
                f"Found columns: {sorted(self.df.columns)}"
            )
        
        return result
    
    def rename_columns(self, standard_names: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Rename DataFrame columns to standard names.
        
        Args:
            standard_names: List of standard column names to rename
            inplace: If True, modify the DataFrame in place
            
        Returns:
            DataFrame with renamed columns
        """
        column_mapping = {}
        for std_name in standard_names:
            actual_col = self.get_column(std_name, required=False)
            if actual_col:
                column_mapping[actual_col] = std_name
        
        if inplace:
            self.df.rename(columns=column_mapping, inplace=True)
            return self.df
        else:
            return self.df.rename(columns=column_mapping)


def standardize_gene_loading(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a gene loading DataFrame to have columns: Name, Score, program_id.
    
    Args:
        df: Gene loading DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    mapper = ColumnMapper(df)
    
    # Get the required columns
    cols = mapper.get_columns(['gene', 'score', 'program_id'], required=True)
    
    # Rename to standard names
    rename_map = {
        cols['gene']: 'Name',
        cols['score']: 'Score',
        cols['program_id']: 'program_id'
    }
    
    return df.rename(columns=rename_map)


def standardize_celltype_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a cell-type enrichment DataFrame.
    Expected output columns: cell_type, program, log2_fc_in_vs_out, fdr
    
    Args:
        df: Cell-type enrichment DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    mapper = ColumnMapper(df)
    
    # Get the required columns
    cols = mapper.get_columns(['cell_type', 'program_id', 'log2_fc', 'fdr'], required=True)
    
    # Rename to standard names
    rename_map = {
        cols['cell_type']: 'cell_type',
        cols['program_id']: 'program',
        cols['log2_fc']: 'log2_fc_in_vs_out',
        cols['fdr']: 'fdr'
    }
    
    return df.rename(columns=rename_map)


def extract_program_id(value: object) -> Optional[int]:
    """Extract numeric program IDs from common naming formats."""
    if value is None:
        return None

    if isinstance(value, numbers.Integral):
        return int(value)

    val_str = str(value).strip()
    try:
        return int(val_str)
    except (TypeError, ValueError):
        pass

    patterns = [
        r'^(?:program|topic|p|x)_(\d+)$',
        r'^(?:program|topic|p|x)(\d+)$',
        r'^(\d+)$',
    ]
    for pattern in patterns:
        match = re.match(pattern, val_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _coerce_boolean_series(series: pd.Series) -> pd.Series:
    """Convert common truthy/falsy representations to booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    text = series.astype(str).str.strip().str.lower()
    truthy = {'true', 't', '1', 'yes', 'y'}
    falsy = {'false', 'f', '0', 'no', 'n', 'nan', 'none', ''}
    mapped = pd.Series(pd.NA, index=series.index, dtype="boolean")
    mapped[text.isin(truthy)] = True
    mapped[text.isin(falsy)] = False

    numeric = pd.to_numeric(series, errors='coerce')
    numeric_bool = numeric.notna() & numeric.ne(0)
    mapped = mapped.where(mapped.notna(), numeric_bool)
    return mapped.fillna(False).astype(bool)


def standardize_regulator_results(
    df: pd.DataFrame,
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Standardize regulator-result columns and derive significance when needed.

    Output columns always include:
    - ``program_id``
    - ``grna_target``
    - ``log_2_fold_change``
    - ``p_value``
    - ``significant``

    When an adjusted p-value column is available, the standardized DataFrame also
    includes ``adj_p_value`` and uses it for derived significance if no explicit
    ``significant`` column is present.
    """
    mapper = ColumnMapper(df)
    cols = mapper.get_columns(
        ['program_id', 'regulator_gene', 'log2_fc'],
        required=True,
    )
    raw_p_col = mapper.get_column('raw_p_value', required=False)
    adj_p_col = mapper.get_column('adj_p_value', required=False)
    significant_col = mapper.get_column('significant', required=False)

    if raw_p_col is None and adj_p_col is None:
        raise ValueError(
            "Regulator results must include a raw or adjusted p-value column "
            "(for example p_value, p_val, adj_pval, or p_adj)."
        )

    standardized = df.copy()

    program_ids = standardized[cols['program_id']].apply(extract_program_id)
    if program_ids.isna().any():
        bad_values = (
            standardized.loc[program_ids.isna(), cols['program_id']]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Could not parse regulator program IDs from column "
            f"'{cols['program_id']}'. Example values: {bad_values}"
        )

    standardized['program_id'] = program_ids.astype(int)
    standardized['grna_target'] = standardized[cols['regulator_gene']].astype(str)
    standardized['log_2_fold_change'] = pd.to_numeric(
        standardized[cols['log2_fc']], errors='coerce'
    )

    if standardized['log_2_fold_change'].isna().any():
        raise ValueError(
            f"Could not parse numeric log2 fold changes from '{cols['log2_fc']}'."
        )

    if raw_p_col is not None:
        standardized['p_value'] = pd.to_numeric(standardized[raw_p_col], errors='coerce')
    else:
        standardized['p_value'] = pd.to_numeric(standardized[adj_p_col], errors='coerce')  # type: ignore[index]

    if standardized['p_value'].isna().all():
        raise ValueError("Regulator p-values could not be parsed as numeric values.")

    if adj_p_col is not None:
        standardized['adj_p_value'] = pd.to_numeric(
            standardized[adj_p_col], errors='coerce'
        )

    if significant_col is not None:
        standardized['significant'] = _coerce_boolean_series(standardized[significant_col])
    elif adj_p_col is not None:
        standardized['significant'] = standardized['adj_p_value'].le(significance_threshold)
    else:
        standardized['significant'] = standardized['p_value'].le(significance_threshold)

    return standardized
