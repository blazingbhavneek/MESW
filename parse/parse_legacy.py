# #########################################
# # Docling based
# #########################################

# import json
# import os
# import re
# import subprocess
# from collections import Counter
# from typing import Any, Dict, List

# import pdfplumber
# from docling.document_converter import DocumentConverter


# class DocxExtractor:
#     def __init__(self, docx_path: str):
#         self.docx_path = docx_path

#     def extract_docling_markdown(self) -> str:
#         """Extract structured markdown from DOCX using docling"""
#         converter = DocumentConverter()
#         doc = converter.convert(source=self.docx_path).document
#         return doc.export_to_markdown()

#     def parse_markdown_into_items(self, md: str) -> List[Dict[str, Any]]:
#         """
#         Parse markdown into a flat list of content items with hierarchy info.
#         Each item has:
#           - type: "heading" or "text"
#           - level: 1 for h1, 2 for h2, etc.
#           - content: the text
#           - section: current section number
#           - section_name: current section name
#           - subsection: current subsection number
#           - subsection_name: current subsection name
#         """
#         items = []
#         lines = md.split("\n")

#         current_section = 0
#         current_section_name = ""
#         current_subsection = 0
#         current_subsection_name = ""

#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue

#             # Heading detection
#             if line.startswith("## "):
#                 current_section += 1
#                 current_section_name = line[3:].strip()
#                 current_subsection = 0
#                 current_subsection_name = ""
#                 items.append(
#                     {
#                         "type": "heading",
#                         "level": 1,
#                         "content": line,
#                         "section": current_section,
#                         "section_name": current_section_name,
#                         "subsection": current_subsection,
#                         "subsection_name": current_subsection_name,
#                     }
#                 )
#             elif line.startswith("### "):
#                 current_subsection += 1
#                 current_subsection_name = line[4:].strip()
#                 items.append(
#                     {
#                         "type": "heading",
#                         "level": 2,
#                         "content": line,
#                         "section": current_section,
#                         "section_name": current_section_name,
#                         "subsection": current_subsection,
#                         "subsection_name": current_subsection_name,
#                     }
#                 )
#             else:
#                 # Regular content
#                 items.append(
#                     {
#                         "type": "text",
#                         "content": line,
#                         "section": current_section,
#                         "section_name": current_section_name,
#                         "subsection": current_subsection,
#                         "subsection_name": current_subsection_name,
#                     }
#                 )

#         return items

#     def convert_to_pdf(self, output_path: str = None) -> str:
#         """Convert DOCX to PDF using LibreOffice"""
#         if output_path is None:
#             base_name = os.path.splitext(self.docx_path)[0]
#             output_path = base_name + ".pdf"

#         docx_abs = os.path.abspath(self.docx_path)
#         pdf_abs = os.path.abspath(output_path)

#         for cmd in ["soffice", "libreoffice"]:
#             try:
#                 result = subprocess.run(
#                     [
#                         cmd,
#                         "--headless",
#                         "--convert-to",
#                         "pdf",
#                         "--outdir",
#                         os.path.dirname(pdf_abs),
#                         docx_abs,
#                     ],
#                     capture_output=True,
#                     text=True,
#                     timeout=30,
#                 )
#                 if result.returncode == 0 and os.path.exists(pdf_abs):
#                     print(f"Converted using {cmd}")
#                     return pdf_abs
#             except (FileNotFoundError, subprocess.TimeoutExpired):
#                 continue

#         raise FileNotFoundError("PDF conversion failed. Please install LibreOffice.")

#     def extract_footer_pattern(self, pdf_path: str) -> str:
#         """Extract the most common footer pattern from the PDF"""
#         footers = []

#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 # Try to extract footer from bottom of page
#                 footer_height = 60  # points
#                 footer_crop = page.crop(
#                     (0, page.height - footer_height, page.width, page.height)
#                 )
#                 footer_text = footer_crop.extract_text()
#                 if footer_text:
#                     footer_text = footer_text.strip()
#                     if footer_text:
#                         footers.append(footer_text)

#         # Find the most common footer pattern (appearing in at least 50% of pages)
#         if footers:
#             footer_counter = Counter(footers)
#             most_common = footer_counter.most_common(1)
#             if most_common and most_common[0][1] >= len(footers) * 0.5:
#                 footer_pattern = most_common[0][0]
#                 print(f"Found consistent footer pattern: '{footer_pattern}'")
#                 return footer_pattern

#         print("No consistent footer pattern found.")
#         return ""

#     def assign_page_numbers(
#         self, items: List[Dict[str, Any]], footer_pattern: str
#     ) -> List[Dict[str, Any]]:
#         """
#         Assign page numbers to items based on footer occurrences.
#         Also marks items that contain the footer pattern.
#         """
#         result = []
#         current_page = 1

#         for item in items:
#             new_item = item.copy()

#             # Check if this item contains the footer pattern
#             if footer_pattern and footer_pattern in new_item["content"]:
#                 # This item contains the footer – mark it
#                 new_item["is_footer"] = True
#                 # Increment page number for the next item
#                 current_page += 1
#             else:
#                 new_item["is_footer"] = False

#             # Assign current page number
#             new_item["page"] = current_page
#             result.append(new_item)

#         return result

#     def group_by_page_and_hierarchy(
#         self, items: List[Dict[str, Any]]
#     ) -> List[Dict[str, Any]]:
#         """
#         Group items by page, section, and subsection.
#         Each result block contains all content for a specific section/subsection on a specific page.
#         """
#         result = []
#         current_page = None
#         current_section = None
#         current_section_name = ""
#         current_subsection = None
#         current_subsection_name = ""
#         current_content = []

#         for item in items:
#             # Skip footer items as they're just delimiters
#             if item.get("is_footer", False):
#                 continue

#             # Check if we need to start a new block
#             new_block = (
#                 current_page is None
#                 or item["page"] != current_page
#                 or item["section"] != current_section
#                 or item["subsection"] != current_subsection
#             )

#             if new_block and current_content:
#                 # Save the current block
#                 result.append(
#                     {
#                         "page": current_page,
#                         "section": current_section,
#                         "section_name": current_section_name,
#                         "subsection": current_subsection,
#                         "subsection_name": current_subsection_name,
#                         "content": "\n\n".join(current_content),
#                     }
#                 )
#                 current_content = []

#             # Update current state
#             current_page = item["page"]
#             current_section = item["section"]
#             current_section_name = item["section_name"]
#             current_subsection = item["subsection"]
#             current_subsection_name = item["subsection_name"]

#             # Add content (skip the heading marker like # or ##)
#             if item["type"] == "heading":
#                 # For headings, just take the text part without the heading markers
#                 content = re.sub(r"^#+\s*", "", item["content"]).strip()
#                 # Add appropriate heading level marker for output
#                 if item["level"] == 1:
#                     content = f"# {content}"
#                 elif item["level"] == 2:
#                     content = f"## {content}"
#                 else:
#                     content = f"{'#' * item['level']} {content}"
#                 current_content.append(content)
#             else:
#                 current_content.append(item["content"])

#         # Add the last block
#         if current_content:
#             result.append(
#                 {
#                     "page": current_page,
#                     "section": current_section,
#                     "section_name": current_section_name,
#                     "subsection": current_subsection,
#                     "subsection_name": current_subsection_name,
#                     "content": "\n\n".join(current_content),
#                 }
#             )

#         return result

#     def extract(self) -> List[Dict[str, Any]]:
#         """Main extraction method"""
#         print("Step 1: Extracting markdown from DOCX using docling...")
#         md = self.extract_docling_markdown()

#         print("Step 2: Parsing markdown into flat list of content items...")
#         items = self.parse_markdown_into_items(md)
#         print(f"Found {len(items)} content items")

#         print("Step 3: Converting to PDF...")
#         pdf_path = self.convert_to_pdf()
#         print(f"PDF saved at: {pdf_path}")

#         print("Step 4: Extracting footer pattern from PDF...")
#         footer_pattern = self.extract_footer_pattern(pdf_path)

#         print("Step 5: Assigning page numbers to content items...")
#         items_with_pages = self.assign_page_numbers(items, footer_pattern)

#         print("Step 6: Grouping content by page, section, and subsection...")
#         result = self.group_by_page_and_hierarchy(items_with_pages)
#         print(f"Created {len(result)} structured blocks")

#         # Print page distribution
#         print("\nPage Distribution:")
#         page_counts = {}
#         for block in result:
#             page = block["page"]
#             page_counts[page] = page_counts.get(page, 0) + 1

#         for page in sorted(page_counts.keys()):
#             print(f"Page {page}: {page_counts[page]} blocks")

#         return result


# # Usage Example
# if __name__ == "__main__":
#     num_suffix = ""
#     extractor = DocxExtractor("test_doc" + str(num_suffix) + ".docx")
#     result = extractor.extract()

#     # Save to JSON
#     output_file = "extracted_data2" + str(num_suffix) + ".json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     print(f"\nExtraction complete! Results saved to: {output_file}")
#     print(f"Total blocks: {len(result)}")


# #########################################
# # PDF Plumber + Markitdown based
# #########################################

# """
# DOCX to Structured Markdown Extractor with Page Numbers
# NEW APPROACH: Use PDF for content order, DOCX for hierarchy
# """

# import json
# import os
# import re
# import subprocess
# from difflib import SequenceMatcher
# from typing import Any, Dict, List, Tuple

# import pdfplumber
# from docx import Document
# from docx.oxml.table import CT_Tbl
# from docx.oxml.text.paragraph import CT_P
# from docx.table import Table
# from docx.text.paragraph import Paragraph


# class DocxExtractor:
#     def __init__(self, docx_path: str):
#         self.docx_path = docx_path
#         self.doc = Document(docx_path)

#     def normalize_text(self, text: str) -> str:
#         """Normalize text for comparison"""
#         text = re.sub(r"\s+", " ", text)
#         text = text.strip().lower()
#         return text

#     def text_similarity(self, text1: str, text2: str) -> float:
#         """Calculate similarity ratio between two text strings"""
#         return SequenceMatcher(
#             None, self.normalize_text(text1), self.normalize_text(text2)
#         ).ratio()

#     def is_heading(self, paragraph: Paragraph) -> tuple:
#         """Check if paragraph is a heading and return its level"""
#         if paragraph.style.name.startswith("Heading"):
#             try:
#                 level = int(paragraph.style.name.split()[-1])
#                 return True, level
#             except (ValueError, IndexError):
#                 return False, 0
#         return False, 0

#     def extract_docx_headings_map(self) -> Dict[str, Tuple[int, str, int]]:
#         """
#         Extract heading text and their hierarchy from DOCX
#         Returns: {normalized_text: (section, section_name, level)}
#         """
#         headings_map = {}
#         current_section = 0
#         current_section_name = ""

#         for element in self.doc.element.body:
#             if isinstance(element, CT_P):
#                 paragraph = Paragraph(element, self.doc)
#                 is_head, level = self.is_heading(paragraph)

#                 if is_head:
#                     text = paragraph.text.strip()
#                     if not text:
#                         continue

#                     if level == 1:
#                         current_section += 1
#                         current_section_name = text

#                     # Store normalized text -> hierarchy info
#                     normalized = self.normalize_text(text)
#                     headings_map[normalized] = (
#                         current_section,
#                         current_section_name,
#                         level,
#                     )

#                     print(f"  Found heading: H{level} -> {text[:60]}")

#         return headings_map

#     def convert_to_pdf(self, output_path: str = None) -> str:
#         """Convert DOCX to PDF"""
#         if output_path is None:
#             base_name = os.path.splitext(self.docx_path)[0]
#             output_path = base_name + ".pdf"

#         docx_abs = os.path.abspath(self.docx_path)
#         pdf_abs = os.path.abspath(output_path)

#         for cmd in ["soffice", "libreoffice"]:
#             try:
#                 result = subprocess.run(
#                     [
#                         cmd,
#                         "--headless",
#                         "--convert-to",
#                         "pdf",
#                         "--outdir",
#                         os.path.dirname(pdf_abs),
#                         docx_abs,
#                     ],
#                     capture_output=True,
#                     text=True,
#                     timeout=30,
#                 )

#                 if result.returncode == 0 and os.path.exists(pdf_abs):
#                     print(f"  ✓ Converted using {cmd}")
#                     return pdf_abs
#             except (FileNotFoundError, subprocess.TimeoutExpired):
#                 continue

#         raise FileNotFoundError("PDF conversion failed. Please install LibreOffice.")

#     def extract_pdf_pages(self, pdf_path: str) -> List[Dict]:
#         """
#         Extract text from PDF page by page, preserving order
#         Returns list of pages with their content
#         """
#         pages = []

#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 text = page.extract_text()

#                 if text:
#                     # Clean up
#                     text = re.sub(r"\n\s*\n+", "\n\n", text)
#                     text = text.strip()

#                     # Split into lines for processing
#                     lines = [l.strip() for l in text.split("\n") if l.strip()]

#                     pages.append({"page": page_num, "text": text, "lines": lines})

#         return pages

#     def identify_heading_in_text(
#         self, text: str, headings_map: Dict
#     ) -> Tuple[bool, int, int, str]:
#         """
#         Check if text is a heading and return its hierarchy
#         Returns: (is_heading, section, level, section_name)
#         """
#         normalized = self.normalize_text(text)

#         # Exact match
#         if normalized in headings_map:
#             section, section_name, level = headings_map[normalized]
#             return True, section, level, section_name

#         # Fuzzy match for headings (sometimes formatting adds/removes spaces)
#         for heading_text, (section, section_name, level) in headings_map.items():
#             similarity = SequenceMatcher(None, normalized, heading_text).ratio()
#             if similarity > 0.85:  # High threshold for headings
#                 return True, section, level, section_name

#         return False, 0, 0, ""

#     def process_pages_with_hierarchy(
#         self, pages: List[Dict], headings_map: Dict
#     ) -> List[Dict]:
#         """
#         Process PDF pages and add hierarchy information from DOCX
#         """
#         result = []

#         current_section = 0
#         current_section_name = ""
#         current_subsection = 0
#         current_subsection_name = ""

#         for page_data in pages:
#             page_num = page_data["page"]
#             lines = page_data["lines"]

#             # Group lines into paragraphs (join consecutive non-heading lines)
#             paragraphs = []
#             current_para = []

#             for line in lines:
#                 # Check if line is a heading
#                 is_head, section, level, section_name = self.identify_heading_in_text(
#                     line, headings_map
#                 )

#                 if is_head:
#                     # Save accumulated paragraph
#                     if current_para:
#                         paragraphs.append(
#                             {
#                                 "type": "text",
#                                 "content": " ".join(current_para),
#                                 "section": current_section,
#                                 "section_name": current_section_name,
#                                 "subsection": current_subsection,
#                                 "subsection_name": current_subsection_name,
#                             }
#                         )
#                         current_para = []

#                     # Update hierarchy
#                     if level == 1:
#                         current_section = section
#                         current_section_name = section_name
#                         current_subsection = 0
#                         current_subsection_name = ""
#                         heading_text = f"# {line}"
#                     elif level == 2:
#                         current_subsection += 1
#                         current_subsection_name = line
#                         heading_text = f"## {line}"
#                     else:
#                         heading_text = f"{'#' * level} {line}"

#                     # Add heading as separate item
#                     paragraphs.append(
#                         {
#                             "type": "heading",
#                             "content": heading_text,
#                             "level": level,
#                             "section": current_section,
#                             "section_name": current_section_name,
#                             "subsection": current_subsection,
#                             "subsection_name": current_subsection_name,
#                         }
#                     )
#                 else:
#                     # Regular text - accumulate
#                     current_para.append(line)

#             # Save final paragraph
#             if current_para:
#                 paragraphs.append(
#                     {
#                         "type": "text",
#                         "content": " ".join(current_para),
#                         "section": current_section,
#                         "section_name": current_section_name,
#                         "subsection": current_subsection,
#                         "subsection_name": current_subsection_name,
#                     }
#                 )

#             # Group paragraphs by section/subsection for this page
#             if paragraphs:
#                 # Combine consecutive items with same section/subsection
#                 grouped = []
#                 current_group = []
#                 current_key = None

#                 for para in paragraphs:
#                     key = (para["section"], para["subsection"])

#                     if key != current_key and current_group:
#                         # Save previous group
#                         grouped.append(
#                             {
#                                 "page": page_num,
#                                 "section": current_group[0]["section"],
#                                 "section_name": current_group[0]["section_name"],
#                                 "subsection": current_group[0]["subsection"],
#                                 "subsection_name": current_group[0]["subsection_name"],
#                                 "content": "\n\n".join(
#                                     [p["content"] for p in current_group]
#                                 ),
#                             }
#                         )
#                         current_group = []

#                     current_key = key
#                     current_group.append(para)

#                 # Save final group
#                 if current_group:
#                     grouped.append(
#                         {
#                             "page": page_num,
#                             "section": current_group[0]["section"],
#                             "section_name": current_group[0]["section_name"],
#                             "subsection": current_group[0]["subsection"],
#                             "subsection_name": current_group[0]["subsection_name"],
#                             "content": "\n\n".join(
#                                 [p["content"] for p in current_group]
#                             ),
#                         }
#                     )

#                 result.extend(grouped)

#         return result

#     def extract(self) -> List[Dict[str, Any]]:
#         """
#         Main extraction method
#         Returns structured data with page numbers, sections, and content
#         """
#         print("Step 1: Extracting headings hierarchy from DOCX...")
#         headings_map = self.extract_docx_headings_map()
#         print(f"  Found {len(headings_map)} headings")

#         print("\nStep 2: Converting to PDF...")
#         pdf_path = self.convert_to_pdf()
#         print(f"  PDF saved at: {pdf_path}")

#         print("\nStep 3: Extracting content from PDF (in visual order)...")
#         pages = self.extract_pdf_pages(pdf_path)
#         print(f"  Found {len(pages)} pages")

#         print("\nStep 4: Matching PDF content with DOCX hierarchy...")
#         result = self.process_pages_with_hierarchy(pages, headings_map)
#         print(f"  Created {len(result)} structured blocks")

#         # Print page distribution
#         print("\n📄 Page Distribution:")
#         page_counts = {}
#         for block in result:
#             page = block["page"]
#             page_counts[page] = page_counts.get(page, 0) + 1

#         for page in sorted(page_counts.keys()):
#             print(f"  Page {page}: {page_counts[page]} blocks")

#         return result


# # Usage Example
# if __name__ == "__main__":
#     num_suffix = ""
#     extractor = DocxExtractor("test_doc" + str(num_suffix) + ".docx")

#     result = extractor.extract()

#     # Save to JSON
#     output_file = "extracted_data" + str(num_suffix) + ".json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     print(f"\n✅ Extraction complete! Results saved to: {output_file}")
#     print(f"📊 Total blocks: {len(result)}")
