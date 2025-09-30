"""
JSON to Markdown Converter
Converts extracted JSON data back to a readable markdown document
"""

import json
import os
from typing import Any, Dict, List


class JsonToMarkdown:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load JSON data from file and sort by page, section, subsection"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Sort by page (primary), section (secondary), subsection (tertiary)
        sorted_data = sorted(
            data, key=lambda x: (x["page"], x["section"], x["subsection"])
        )

        return sorted_data

    def _format_section_header(self, block: Dict[str, Any]) -> str:
        """Format section and subsection information"""
        lines = []

        # Add page number
        lines.append(f"**Page {block['page']}**")

        # Add section info if available
        if block["section"] > 0 and block["section_name"]:
            lines.append(f"**Section {block['section']}: {block['section_name']}**")

        # Add subsection info if available
        if block["subsection"] > 0 and block["subsection_name"]:
            lines.append(
                f"**Subsection {block['subsection']}: {block['subsection_name']}**"
            )

        return "\n".join(lines)

    def convert_simple(self, output_path: str = None) -> str:
        """
        Convert JSON to simple markdown format
        Shows page breaks and content sequentially
        """
        if output_path is None:
            base_name = os.path.splitext(self.json_path)[0]
            output_path = base_name + "_simple.md"

        markdown_lines = []
        markdown_lines.append("# Document Content\n")

        current_page = None
        current_section = None
        current_subsection = None

        for block in self.data:
            # Add page header when page changes
            if block["page"] != current_page:
                if current_page is not None:  # Don't add separator before first page
                    markdown_lines.append("---")
                    markdown_lines.append("")
                markdown_lines.append(f"Page {block['page']}")
                markdown_lines.append("")
                markdown_lines.append("---")
                current_page = block["page"]
                current_section = None  # Reset section tracking when page changes
                current_subsection = None  # Reset subsection tracking when page changes

            # Add section header if it's new
            if (
                block["section"] != current_section
                and block["section"] > 0
                and block["section_name"]
            ):
                current_section = block["section"]
                current_subsection = None  # Reset subsection when section changes

            # Add subsection header if it's new
            if (
                block["subsection"] != current_subsection
                and block["subsection"] > 0
                and block["subsection_name"]
            ):
                current_subsection = block["subsection"]

            # Add content
            markdown_lines.append(block["content"])
            markdown_lines.append("\n")

        markdown_content = "\n".join(markdown_lines)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✓ Simple markdown saved to: {output_path}")
        return markdown_content

    def convert_detailed(self, output_path: str = None) -> str:
        """
        Convert JSON to detailed markdown format
        Includes metadata boxes for each content block
        """
        if output_path is None:
            base_name = os.path.splitext(self.json_path)[0]
            output_path = base_name + "_detailed.md"

        markdown_lines = []
        markdown_lines.append("# Document Content (Detailed View)\n")
        markdown_lines.append(
            "*This view shows the complete structure with metadata*\n"
        )

        current_page = None

        for i, block in enumerate(self.data, 1):
            # Add page header when page changes
            if block["page"] != current_page:
                if current_page is not None:  # Don't add separator before first page
                    markdown_lines.append("---")
                markdown_lines.append(f"Page {block['page']}")
                markdown_lines.append("---")
                current_page = block["page"]

            # Metadata box
            markdown_lines.append(f"\n## Block {i}")
            markdown_lines.append("```")
            markdown_lines.append(f"Page: {block['page']}")
            markdown_lines.append(
                f"Section: {block['section']} - {block['section_name']}"
            )
            markdown_lines.append(
                f"Subsection: {block['subsection']} - {block['subsection_name']}"
            )
            markdown_lines.append("```\n")

            # Content
            markdown_lines.append(block["content"])
            markdown_lines.append("\n---\n")

        markdown_content = "\n".join(markdown_lines)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✓ Detailed markdown saved to: {output_path}")
        return markdown_content

    def convert_structured(self, output_path: str = None) -> str:
        """
        Convert JSON to structured markdown format
        Organizes by sections and subsections with page references
        """
        if output_path is None:
            base_name = os.path.splitext(self.json_path)[0]
            output_path = base_name + "_structured.md"

        markdown_lines = []
        markdown_lines.append("# Document Content (Structured View)\n")

        # Group by section and subsection
        current_page = None
        current_section = None
        current_subsection = None

        for block in self.data:
            # Add page header when page changes
            if block["page"] != current_page:
                if current_page is not None:  # Don't add separator before first page
                    markdown_lines.append("---")
                markdown_lines.append(f"Page {block['page']}")
                markdown_lines.append("---")
                current_page = block["page"]
                current_section = None  # Reset section tracking when page changes
                current_subsection = None  # Reset subsection tracking when page changes

            # New section
            if block["section"] != current_section:
                current_section = block["section"]
                current_subsection = None
                if block["section_name"]:
                    markdown_lines.append(f"\n# {block['section_name']}")
                    markdown_lines.append(f"*Section {block['section']}*\n")

            # New subsection
            if block["subsection"] != current_subsection:
                current_subsection = block["subsection"]
                if block["subsection_name"]:
                    markdown_lines.append(f"\n## {block['subsection_name']}")
                    markdown_lines.append(
                        f"*Subsection {block['subsection']} | Page {block['page']}*\n"
                    )

            # Content with page reference if not already shown
            if not block["subsection_name"]:
                markdown_lines.append(f"\n*Page {block['page']}*\n")

            markdown_lines.append(block["content"])
            markdown_lines.append("\n")

        markdown_content = "\n".join(markdown_lines)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✓ Structured markdown saved to: {output_path}")
        return markdown_content

    def convert_toc(self, output_path: str = None) -> str:
        """
        Convert JSON to markdown with Table of Contents
        Includes navigation links to sections
        """
        if output_path is None:
            base_name = os.path.splitext(self.json_path)[0]
            output_path = base_name + "_toc.md"

        markdown_lines = []
        markdown_lines.append("# Document Content\n")

        # Build Table of Contents
        markdown_lines.append("## Table of Contents\n")

        seen_sections = set()
        seen_subsections = set()

        for block in self.data:
            if block["section"] > 0 and block["section_name"]:
                section_key = (block["section"], block["section_name"])
                if section_key not in seen_sections:
                    seen_sections.add(section_key)
                    anchor = block["section_name"].lower().replace(" ", "-")
                    markdown_lines.append(
                        f"- [{block['section']}. {block['section_name']}](#{anchor}) (Page {block['page']})"
                    )

                    # Add subsections under this section
                    for sub_block in self.data:
                        if (
                            sub_block["section"] == block["section"]
                            and sub_block["subsection"] > 0
                            and sub_block["subsection_name"]
                        ):
                            subsection_key = (
                                sub_block["section"],
                                sub_block["subsection"],
                                sub_block["subsection_name"],
                            )
                            if subsection_key not in seen_subsections:
                                seen_subsections.add(subsection_key)
                                sub_anchor = (
                                    sub_block["subsection_name"]
                                    .lower()
                                    .replace(" ", "-")
                                )
                                markdown_lines.append(
                                    f"  - [{block['section']}.{sub_block['subsection']}. {sub_block['subsection_name']}](#{sub_anchor}) (Page {sub_block['page']})"
                                )

        markdown_lines.append("\n---\n")

        # Add content
        current_page = None
        current_section = None
        current_subsection = None

        for block in self.data:
            # Add page header when page changes
            if block["page"] != current_page:
                if current_page is not None:  # Don't add separator before first page
                    markdown_lines.append("---")
                markdown_lines.append(f"Page {block['page']}")
                markdown_lines.append("---")
                current_page = block["page"]
                current_section = None  # Reset section tracking when page changes
                current_subsection = None  # Reset subsection tracking when page changes

            # New section
            if block["section"] != current_section:
                current_section = block["section"]
                current_subsection = None
                if block["section_name"]:
                    markdown_lines.append(f"\n# {block['section_name']}")
                    markdown_lines.append(
                        f"*Section {block['section']} | Page {block['page']}*\n"
                    )

            # New subsection
            if block["subsection"] != current_subsection:
                current_subsection = block["subsection"]
                if block["subsection_name"]:
                    markdown_lines.append(f"\n## {block['subsection_name']}")
                    markdown_lines.append(
                        f"*Subsection {block['subsection']} | Page {block['page']}*\n"
                    )

            # Content
            markdown_lines.append(block["content"])
            markdown_lines.append("\n")

        markdown_content = "\n".join(markdown_lines)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✓ Markdown with TOC saved to: {output_path}")
        return markdown_content

    def convert_all(self, base_output_path: str = None):
        """Generate all markdown formats"""
        print("\nGenerating all markdown formats...\n")

        self.convert_simple(base_output_path)
        self.convert_detailed(base_output_path)
        self.convert_structured(base_output_path)
        self.convert_toc(base_output_path)

        print("\n✓ All formats generated successfully!")


# Usage Example
if __name__ == "__main__":
    # Initialize converter

    num_suffix = ""

    converter = JsonToMarkdown("extracted_data" + str(num_suffix) + ".json")

    # Option 1: Generate a specific format
    converter.convert_simple("output_simple" + str(num_suffix) + ".md")
