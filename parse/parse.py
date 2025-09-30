import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import pdfplumber
from docling.document_converter import DocumentConverter


class HybridDocxExtractor:
    def __init__(self, docx_path: str):
        self.docx_path = docx_path

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison - remove extra whitespace, lowercase"""
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()
        # Remove common punctuation and special chars
        text = re.sub(r"[•\-–—\*\_\#]", "", text)
        return text

    def extract_docling_markdown(self) -> str:
        """Extract structured markdown from DOCX using docling"""
        converter = DocumentConverter()
        doc = converter.convert(source=self.docx_path).document
        return doc.export_to_markdown()

    def parse_markdown_into_items(self, md: str) -> List[Dict[str, Any]]:
        """Parse markdown into a flat list of content items with hierarchy info"""
        items = []
        lines = md.split("\n")

        current_section = 0
        current_section_name = ""
        current_subsection = 0
        current_subsection_name = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Heading detection
            if line.startswith("## "):
                current_section += 1
                current_section_name = line[3:].strip()
                current_subsection = 0
                current_subsection_name = ""
                items.append(
                    {
                        "type": "heading",
                        "level": 1,
                        "content": line,
                        "text": current_section_name,
                        "section": current_section,
                        "section_name": current_section_name,
                        "subsection": current_subsection,
                        "subsection_name": current_subsection_name,
                    }
                )
            elif line.startswith("### "):
                current_subsection += 1
                current_subsection_name = line[4:].strip()
                items.append(
                    {
                        "type": "heading",
                        "level": 2,
                        "content": line,
                        "text": current_subsection_name,
                        "section": current_section,
                        "section_name": current_section_name,
                        "subsection": current_subsection,
                        "subsection_name": current_subsection_name,
                    }
                )
            else:
                # Regular content
                items.append(
                    {
                        "type": "text",
                        "content": line,
                        "text": line,
                        "section": current_section,
                        "section_name": current_section_name,
                        "subsection": current_subsection,
                        "subsection_name": current_subsection_name,
                    }
                )

        return items

    def convert_to_pdf(self, output_path: str = None) -> str:
        """Convert DOCX to PDF using LibreOffice"""
        if output_path is None:
            base_name = os.path.splitext(self.docx_path)[0]
            output_path = base_name + ".pdf"

        docx_abs = os.path.abspath(self.docx_path)
        pdf_abs = os.path.abspath(output_path)

        for cmd in ["soffice", "libreoffice"]:
            try:
                result = subprocess.run(
                    [
                        cmd,
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        os.path.dirname(pdf_abs),
                        docx_abs,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and os.path.exists(pdf_abs):
                    print(f"  ✓ Converted using {cmd}")
                    return pdf_abs
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        raise FileNotFoundError("PDF conversion failed. Please install LibreOffice.")

    def extract_pdf_pages_plain_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract plain text from each PDF page (string alpha)"""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # Clean up but keep natural structure
                    text = re.sub(r"\n\s*\n+", "\n\n", text)
                    text = text.strip()

                    pages.append(
                        {
                            "page": page_num,
                            "text": text,
                            "normalized": self.normalize_text(text),
                        }
                    )

        return pages

    def find_text_in_page(
        self, item_text: str, page_text: str, threshold: float = 0.6
    ) -> bool:
        """Check if item text appears in page text with fuzzy matching"""
        item_norm = self.normalize_text(item_text)
        page_norm = self.normalize_text(page_text)

        # Direct substring match (best case)
        if item_norm in page_norm:
            return True

        # For short text, require higher similarity
        if len(item_norm) < 20:
            threshold = 0.8

        # Check if significant portion of item appears in page
        # Split into words and check overlap
        item_words = set(item_norm.split())
        page_words = set(page_norm.split())

        if not item_words:
            return False

        overlap = len(item_words & page_words) / len(item_words)
        return overlap >= threshold

    def align_content_to_pages(
        self, items: List[Dict[str, Any]], pdf_pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sequential alignment: Walk through PDF pages in order,
        assign each structured item to the first page where we find its text.
        """
        print("\n  Walking through pages sequentially...")

        items_with_pages = []
        current_page_idx = 0
        total_pages = len(pdf_pages)

        for i, item in enumerate(items):
            item_text = item["text"]
            assigned = False

            # Search from current page forward (content should be in order)
            for offset in range(total_pages):
                page_idx = (current_page_idx + offset) % total_pages
                page_data = pdf_pages[page_idx]
                page_num = page_data["page"]
                page_text = page_data["text"]

                if self.find_text_in_page(item_text, page_text):
                    new_item = item.copy()
                    new_item["page"] = page_num
                    items_with_pages.append(new_item)

                    # Move current page forward if we found it on a later page
                    if offset > 0:
                        current_page_idx = page_idx

                    assigned = True
                    break

            # Fallback: if not found, assign to current page
            if not assigned:
                new_item = item.copy()
                new_item["page"] = pdf_pages[current_page_idx]["page"]
                items_with_pages.append(new_item)
                print(
                    f"  ⚠ Item {i} not found in any page, assigned to page {pdf_pages[current_page_idx]['page']}"
                )

        # Post-process: ensure pages don't jump backwards
        for i in range(1, len(items_with_pages)):
            if items_with_pages[i]["page"] < items_with_pages[i - 1]["page"]:
                items_with_pages[i]["page"] = items_with_pages[i - 1]["page"]

        return items_with_pages

    def group_by_page_and_hierarchy(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Group items by page, section, and subsection"""
        result = []
        current_page = None
        current_section = None
        current_section_name = ""
        current_subsection = None
        current_subsection_name = ""
        current_content = []

        for item in items:
            # Check if we need to start a new block
            new_block = (
                current_page is None
                or item["page"] != current_page
                or item["section"] != current_section
                or item["subsection"] != current_subsection
            )

            if new_block and current_content:
                # Save the current block
                result.append(
                    {
                        "page": current_page,
                        "section": current_section,
                        "section_name": current_section_name,
                        "subsection": current_subsection,
                        "subsection_name": current_subsection_name,
                        "content": "\n\n".join(current_content),
                    }
                )
                current_content = []

            # Update current state
            current_page = item["page"]
            current_section = item["section"]
            current_section_name = item["section_name"]
            current_subsection = item["subsection"]
            current_subsection_name = item["subsection_name"]

            # Add content
            if item["type"] == "heading":
                content = re.sub(r"^#+\s*", "", item["content"]).strip()
                if item["level"] == 1:
                    content = f"# {content}"
                elif item["level"] == 2:
                    content = f"## {content}"
                else:
                    content = f"{'#' * item['level']} {content}"
                current_content.append(content)
            else:
                current_content.append(item["content"])

        # Add the last block
        if current_content:
            result.append(
                {
                    "page": current_page,
                    "section": current_section,
                    "section_name": current_section_name,
                    "subsection": current_subsection,
                    "subsection_name": current_subsection_name,
                    "content": "\n\n".join(current_content),
                }
            )

        return result

    def extract(self) -> List[Dict[str, Any]]:
        """Main extraction method combining both approaches"""
        print("=" * 60)
        print("HYBRID EXTRACTION: Sequential Page Alignment")
        print("=" * 60)

        # BETA: Get beautifully formatted structure from docling
        print("\n[BETA] Step 1: Extracting structured markdown using docling...")
        md = self.extract_docling_markdown()

        print("[BETA] Step 2: Parsing markdown into items...")
        items = self.parse_markdown_into_items(md)
        print(f"  ✓ Found {len(items)} structured items")

        # ALPHA: Get accurate page breaks from PDF
        print("\n[ALPHA] Step 3: Converting DOCX to PDF...")
        pdf_path = self.convert_to_pdf()
        print(f"  ✓ PDF saved at: {pdf_path}")

        print("[ALPHA] Step 4: Extracting plain text from PDF pages...")
        pdf_pages = self.extract_pdf_pages_plain_text(pdf_path)
        print(f"  ✓ Found {len(pdf_pages)} pages")

        # Debug: Show first page snippet
        if pdf_pages:
            print(f"\n  📄 First page preview (first 200 chars):")
            print(f"     {pdf_pages[0]['text'][:200]}...")

        # ALIGNMENT: Match beta to alpha
        print("\n[ALIGNMENT] Step 5: Aligning structured content with PDF pages...")
        items_with_pages = self.align_content_to_pages(items, pdf_pages)
        print(f"  ✓ Assigned page numbers to all items")

        # GROUPING: Create final structure
        print("\n[GROUPING] Step 6: Creating final structured blocks...")
        result = self.group_by_page_and_hierarchy(items_with_pages)
        print(f"  ✓ Created {len(result)} blocks")

        # Print page distribution
        print("\n" + "=" * 60)
        print("📄 PAGE DISTRIBUTION")
        print("=" * 60)
        page_counts = {}
        for block in result:
            page = block["page"]
            page_counts[page] = page_counts.get(page, 0) + 1

        for page in sorted(page_counts.keys()):
            print(f"  Page {page}: {page_counts[page]} blocks")

        return result


# Usage Example
if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else ""
    extractor = HybridDocxExtractor(str(name) + ".docx")
    result = extractor.extract()

    # Save to JSON
    output_file = str(name) + ".json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✅ EXTRACTION COMPLETE!")
    print("=" * 60)
