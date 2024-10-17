import ast
import astor
import textwrap
from typing import List

class PythonChunkExtractor:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def extract_chunks(self, file_name: str, content: str) -> List[str]:
        try:
            tree = ast.parse(content)
            chunks = self._process_node(tree, file_name)
            return chunks
        except SyntaxError:
            # Fallback to line-based chunking if parsing fails
            return self._chunk_by_lines(file_name, content)

    def _process_node(self, node: ast.AST, file_name: str, parent_info: str = "") -> List[str]:
        chunks = []

        if isinstance(node, ast.Module):
            for child in ast.iter_child_nodes(node):
                chunks.extend(self._process_node(child, file_name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunk = self._create_chunk(node, file_name, parent_info)
            chunks.append(chunk)

            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = f"class {node.name}"
                        chunks.extend(self._process_node(child, file_name, method_info))
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            # Check for module-level if statements (e.g., if __name__ == "__main__":)
            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                chunk = self._create_chunk(node, file_name, "module-level if statement")
                chunks.append(chunk)

        return chunks

    def _create_chunk(self, node: ast.AST, file_name: str, parent_info: str) -> str:
        node_type = type(node).__name__.lower()
        if isinstance(node, ast.ClassDef):
            name = f"class {node.name}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}function {node.name}"
        else:
            name = "code block"

        source = astor.to_source(node)

        # Only retrieve docstring if the node is a ClassDef or FunctionDef
        docstring = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
        
        if docstring:
            # Remove docstring from source to avoid duplication
            source_lines = source.split('\n')
            docstring_start = source_lines.index('"""' * 2) + 1
            source = '\n'.join(source_lines[docstring_start:]) if docstring_start < len(source_lines) else source

        header = f"File: {file_name}\nType: {node_type}\nName: {name}\n"
        if parent_info:
            header += f"Parent: {parent_info}\n"
        if docstring:
            header += f"Docstring: {textwrap.shorten(docstring, width=100)}\n"

        chunk = f"{header}\n{textwrap.dedent(source)}"

        # If chunk is too large, split it further
        if len(chunk) > self.max_chunk_size:
            return self._split_large_chunk(chunk)
        return chunk

    def _split_large_chunk(self, chunk: str) -> List[str]:
        lines = chunk.split('\n')
        header = '\n'.join(lines[:lines.index('')])  # Extract the header
        code = '\n'.join(lines[lines.index('') + 1:])  # Extract the code

        sub_chunks = []
        current_chunk = [header, ""]  # Start with header and a blank line
        current_size = len(header) + 1

        for line in code.split('\n'):
            if current_size + len(line) + 1 > self.max_chunk_size:
                sub_chunks.append('\n'.join(current_chunk))
                current_chunk = [header, "", line]  # Reset with header, blank line, and current line
                current_size = len(header) + len(line) + 2
            else:
                current_chunk.append(line)
                current_size += len(line) + 1

        if current_chunk:
            sub_chunks.append('\n'.join(current_chunk))

        return sub_chunks

    def _chunk_by_lines(self, file_name: str, content: str) -> List[str]:
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            if current_size + len(line) + 1 > self.max_chunk_size and current_chunk:
                chunks.append(f"File: {file_name}\nType: code\n\n" + '\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += len(line) + 1

        if current_chunk:
            chunks.append(f"File: {file_name}\nType: code\n\n" + '\n'.join(current_chunk))

        return chunks

def split_python_content(file_name: str, content: str, max_chunk_size: int = 1000) -> List[str]:
    extractor = PythonChunkExtractor(max_chunk_size)
    return extractor.extract_chunks(file_name, content)
