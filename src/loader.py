from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .constants import CHUNK_SIZE, CHUNK_OVERLAP, ARTIFACTS_PATH

import os
import time

import torch

def check_gpu():
    """Checks if a CUDA-enabled GPU is available."""
    try:
        return torch.cuda.is_available()
    except Exception as e:
        print(f"Could not check for GPU, falling back to CPU. Error: {e}")
        return False

def process_pdf_with_docling(file_path, enable_ocr = False):
    """
    Processes a PDF using docling to extract high-quality markdown text.
    This method is preferred when a GPU is available.
    """

    print(f"Starting PDF processing with docling (OCR: {enable_ocr})...")

    pipeline_options = PdfPipelineOptions(artifacts_path=ARTIFACTS_PATH)
    pipeline_options.do_table_structure = True # Recognize the structure of tables and represent them as Markdown tables
    pipeline_options.do_ocr = enable_ocr # If document may contain scanned pages or text embedded within images

    num_cpu_threads = os.cpu_count() or 8 

    print(f"Using {num_cpu_threads} CPU threads for processing.")
    
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_cpu_threads, device=AcceleratorDevice.AUTO
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    start_time = time.time()
    result = converter.convert(file_path)
    tot_time = time.time() - start_time
    
    print(f"PDF processing took {tot_time:.2f} seconds.")
    print(f"PDF processing complete. Extracted {len(result.document.tables)} tables.")

    markdown_content = result.document.export_to_markdown()
    
    print(f"Markdown content extracted with length: {len(markdown_content)} characters.")
    
    return markdown_content

def split_markdown_text(markdown_content):
    """
    Splits the markdown text from docling into smaller chunks for the vector store.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Split by headers to group content under its respective headings
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_content)

    # For any chunks that are too large, split them recursively
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(md_header_splits)
    return docs

def load_and_split_with_pypdf(file_path):
    """
    Loads a PDF using PyPDFLoader and splits it into chunks using RecursiveCharacterTextSplitter.
    This is a CPU-based fallback method.
    """
    print("Using CPU fallback: PyPDFLoader and RecursiveCharacterTextSplitter.")
    
    start_time = time.time()
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    tot_time = time.time() - start_time
    
    print(f"PDF processing took {tot_time:.2f} seconds.")

    # For any chunks that are too large, split them recursively
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    docs = text_splitter.split_documents(pages)
    print(f"PDF split into {len(docs)} chunks using PyPDF.")
    return docs
