from lsprotocol import types as lsp
from pygls.workspace import Document, Workspace
from unittest.mock import Mock

try:
    import sys
    from pathlib import Path
    print(str(Path(__file__).parent.parent))
    sys.path.append(str(Path(__file__).parent.parent))
    import flatline_lsp
except Exception:
    pass


class Server:
    def __init__(self):
        super().__init__()
        self.workspace = Workspace('', None)


def test_completion_without_backend_server():
    ls = Server()
    doc = Document("file:///test/test.py", source="import numpy as np\nnp.arr")
    ls.workspace.get_text_document = Mock(return_value=doc)
    completion_list: lsp.CompletionList = flatline_lsp.completions(
        ls=ls,
        params=lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(
                uri="file:///test/test.py"
            ),
            position=lsp.Position(line=1, character=7)
        ),
    )
    print(completion_list.items)
    assert len(completion_list.items) > 0
