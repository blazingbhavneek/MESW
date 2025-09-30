"""
DOCX to Structured Markdown Extractor with Page Numbers
NEW APPROACH: Use PDF for content order, DOCX for hierarchy
"""

import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import pdfplumber
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


class DocxExtractor:
    def __init__(self, docx_path: str):
        self.docx_path = docx_path
        self.doc = Document(docx_path)

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()
        return text

    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings"""
        return SequenceMatcher(
            None, self.normalize_text(text1), self.normalize_text(text2)
        ).ratio()

    def is_heading(self, paragraph: Paragraph) -> tuple:
        """Check if paragraph is a heading and return its level"""
        if paragraph.style.name.startswith("Heading"):
            try:
                level = int(paragraph.style.name.split()[-1])
                return True, level
            except (ValueError, IndexError):
                return False, 0
        return False, 0

    def extract_docx_headings_map(self) -> Dict[str, Tuple[int, str, int]]:
        """
        Extract heading text and their hierarchy from DOCX
        Returns: {normalized_text: (section, section_name, level)}
        """
        headings_map = {}
        current_section = 0
        current_section_name = ""

        for element in self.doc.element.body:
            if isinstance(element, CT_P):
                paragraph = Paragraph(element, self.doc)
                is_head, level = self.is_heading(paragraph)

                if is_head:
                    text = paragraph.text.strip()
                    if not text:
                        continue

                    if level == 1:
                        current_section += 1
                        current_section_name = text

                    # Store normalized text -> hierarchy info
                    normalized = self.normalize_text(text)
                    headings_map[normalized] = (
                        current_section,
                        current_section_name,
                        level,
                    )

                    print(f"  Found heading: H{level} -> {text[:60]}")

        return headings_map

    def convert_to_pdf(self, output_path: str = None) -> str:
        """Convert DOCX to PDF"""
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

    def extract_pdf_pages(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF page by page, preserving order
        Returns list of pages with their content
        """
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if text:
                    # Clean up
                    text = re.sub(r"\n\s*\n+", "\n\n", text)
                    text = text.strip()

                    # Split into lines for processing
                    lines = [l.strip() for l in text.split("\n") if l.strip()]

                    pages.append({"page": page_num, "text": text, "lines": lines})

        return pages

    def identify_heading_in_text(
        self, text: str, headings_map: Dict
    ) -> Tuple[bool, int, int, str]:
        """
        Check if text is a heading and return its hierarchy
        Returns: (is_heading, section, level, section_name)
        """
        normalized = self.normalize_text(text)

        # Exact match
        if normalized in headings_map:
            section, section_name, level = headings_map[normalized]
            return True, section, level, section_name

        # Fuzzy match for headings (sometimes formatting adds/removes spaces)
        for heading_text, (section, section_name, level) in headings_map.items():
            similarity = SequenceMatcher(None, normalized, heading_text).ratio()
            if similarity > 0.85:  # High threshold for headings
                return True, section, level, section_name

        return False, 0, 0, ""

    def process_pages_with_hierarchy(
        self, pages: List[Dict], headings_map: Dict
    ) -> List[Dict]:
        """
        Process PDF pages and add hierarchy information from DOCX
        """
        result = []

        current_section = 0
        current_section_name = ""
        current_subsection = 0
        current_subsection_name = ""

        for page_data in pages:
            page_num = page_data["page"]
            lines = page_data["lines"]

            # Group lines into paragraphs (join consecutive non-heading lines)
            paragraphs = []
            current_para = []

            for line in lines:
                # Check if line is a heading
                is_head, section, level, section_name = self.identify_heading_in_text(
                    line, headings_map
                )

                if is_head:
                    # Save accumulated paragraph
                    if current_para:
                        paragraphs.append(
                            {
                                "type": "text",
                                "content": " ".join(current_para),
                                "section": current_section,
                                "section_name": current_section_name,
                                "subsection": current_subsection,
                                "subsection_name": current_subsection_name,
                            }
                        )
                        current_para = []

                    # Update hierarchy
                    if level == 1:
                        current_section = section
                        current_section_name = section_name
                        current_subsection = 0
                        current_subsection_name = ""
                        heading_text = f"# {line}"
                    elif level == 2:
                        current_subsection += 1
                        current_subsection_name = line
                        heading_text = f"## {line}"
                    else:
                        heading_text = f"{'#' * level} {line}"

                    # Add heading as separate item
                    paragraphs.append(
                        {
                            "type": "heading",
                            "content": heading_text,
                            "level": level,
                            "section": current_section,
                            "section_name": current_section_name,
                            "subsection": current_subsection,
                            "subsection_name": current_subsection_name,
                        }
                    )
                else:
                    # Regular text - accumulate
                    current_para.append(line)

            # Save final paragraph
            if current_para:
                paragraphs.append(
                    {
                        "type": "text",
                        "content": " ".join(current_para),
                        "section": current_section,
                        "section_name": current_section_name,
                        "subsection": current_subsection,
                        "subsection_name": current_subsection_name,
                    }
                )

            # Group paragraphs by section/subsection for this page
            if paragraphs:
                # Combine consecutive items with same section/subsection
                grouped = []
                current_group = []
                current_key = None

                for para in paragraphs:
                    key = (para["section"], para["subsection"])

                    if key != current_key and current_group:
                        # Save previous group
                        grouped.append(
                            {
                                "page": page_num,
                                "section": current_group[0]["section"],
                                "section_name": current_group[0]["section_name"],
                                "subsection": current_group[0]["subsection"],
                                "subsection_name": current_group[0]["subsection_name"],
                                "content": "\n\n".join(
                                    [p["content"] for p in current_group]
                                ),
                            }
                        )
                        current_group = []

                    current_key = key
                    current_group.append(para)

                # Save final group
                if current_group:
                    grouped.append(
                        {
                            "page": page_num,
                            "section": current_group[0]["section"],
                            "section_name": current_group[0]["section_name"],
                            "subsection": current_group[0]["subsection"],
                            "subsection_name": current_group[0]["subsection_name"],
                            "content": "\n\n".join(
                                [p["content"] for p in current_group]
                            ),
                        }
                    )

                result.extend(grouped)

        return result

    def extract(self) -> List[Dict[str, Any]]:
        """
        Main extraction method
        Returns structured data with page numbers, sections, and content
        """
        print("Step 1: Extracting headings hierarchy from DOCX...")
        headings_map = self.extract_docx_headings_map()
        print(f"  Found {len(headings_map)} headings")

        print("\nStep 2: Converting to PDF...")
        pdf_path = self.convert_to_pdf()
        print(f"  PDF saved at: {pdf_path}")

        print("\nStep 3: Extracting content from PDF (in visual order)...")
        pages = self.extract_pdf_pages(pdf_path)
        print(f"  Found {len(pages)} pages")

        print("\nStep 4: Matching PDF content with DOCX hierarchy...")
        result = self.process_pages_with_hierarchy(pages, headings_map)
        print(f"  Created {len(result)} structured blocks")

        # Print page distribution
        print("\n📄 Page Distribution:")
        page_counts = {}
        for block in result:
            page = block["page"]
            page_counts[page] = page_counts.get(page, 0) + 1

        for page in sorted(page_counts.keys()):
            print(f"  Page {page}: {page_counts[page]} blocks")

        return result


# Usage Example
if __name__ == "__main__":
    num_suffix = ""
    extractor = DocxExtractor("test_doc" + str(num_suffix) + ".docx")

    result = extractor.extract()

    # Save to JSON
    output_file = "extracted_data" + str(num_suffix) + ".json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extraction complete! Results saved to: {output_file}")
    print(f"📊 Total blocks: {len(result)}")
