from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_text_splitters import MarkdownTextSplitter
from .constants import CHUNK_SIZE, CHUNK_OVERLAP, ARTIFACTS_PATH

def process_pdf_with_docling(file_path, enable_ocr=False):
    """
    Processes a PDF using docling to extract high-quality markdown text.
    Table extraction is disabled for simplicity and speed.
    """
    print(f"Starting PDF processing with docling (OCR: {enable_ocr})...")
    pipeline_options = PdfPipelineOptions(artifacts_path=ARTIFACTS_PATH)
    pipeline_options.do_ocr = enable_ocr
    
    pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.AUTO)

    converter = DocumentConverter(
        format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })
    
    result = converter.convert(file_path)
    
    print(f"PDF processing complete. Extracted {len(result.document.tables)} tables.")

    markdown_content = result.document.export_to_markdown()
    
    print(f"Markdown content extracted with length: {len(markdown_content)} characters.")
    
    return markdown_content

def split_markdown_text(markdown_content):
    """
    Splits the markdown text into smaller chunks for the vector store.
    """
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = markdown_splitter.create_documents([markdown_content])
    print(f"Markdown content split into {len(docs)} chunks.")
    return docs
