from knowledge_forge import web


def test_structure_and_chunk_pagination_reset_their_scroll_containers():
    script = web.SCRIPT

    assert "function resetPagedListScroll(container)" in script
    assert "container.scrollTop=0;container.scrollLeft=0" in script
    assert "resetPagedListScroll(structureList)" in script
    assert "resetPagedListScroll(chunkList)" in script
