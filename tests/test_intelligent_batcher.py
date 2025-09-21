import pytest
from src.utils.intelligent_batcher import IntelligentBatcher

# Test the creation of batches by IntelligentBatcher
def test_batch_creation():
    batcher = IntelligentBatcher(batch_size=2)
    items = ["item1", "item2", "item3", "item4"]
    batches = batcher.create_batches(items)

    assert len(batches) == 2
    assert batches[0] == ["item1", "item2"]
    assert batches[1] == ["item3", "item4"]

# Test behavior when no items are provided
def test_empty_batch():
    batcher = IntelligentBatcher(batch_size=2)
    items = []
    batches = batcher.create_batches(items)

    assert len(batches) == 0