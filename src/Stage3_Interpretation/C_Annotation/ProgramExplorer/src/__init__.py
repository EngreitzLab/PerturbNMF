# Annotation module - LLM-driven gene program annotation pipeline
#
# Library-style imports for programmatic usage:
from .gene_extraction import (
    extract_top_genes_by_program,
    build_uniqueness_table,
    generate_celltype_summary,
    extract_program_id,
)
from .string_enrichment import (
    call_string_enrichment,
    filter_process_kegg,
    download_string_enrichment_figure,
)
from .literature_mining import (
    load_program_genes,
    resolve_gene_summaries,
    validate_program_regulators,
    load_regulator_data,
)
from .prompt_builder import (
    generate_prompt,
    PROMPT_TEMPLATE,
    load_gene_table,
    load_celltype_annotations,
    prepare_enrichment_mapping,
    load_ncbi_context,
)
from .result_parser import (
    parse_final_results,
    generate_unique_topic_names,
)
from .html_report import (
    generate_report,
    extract_program_stats,
)
from .column_mapper import ColumnMapper
from .ncbi_api import NcbiClient
from .harmonizome_api import HarmonizomeClient
from .compile_regulators import compile_regulator_days
