import pytest
import torch
import sys
import os

# Add the parent directory to the path so we can import refrag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from refrag import SimpleREFRAG, REFRAGTrainingSimulator

class TestSimpleREFRAG:
    """Test cases for SimpleREFRAG implementation"""
    
    @pytest.fixture
    def refrag_model(self):
        """Create a REFRAG model for testing"""
        return SimpleREFRAG(
            decoder_model_name="microsoft/DialoGPT-small",
            encoder_model_name="distilbert-base-uncased",
            chunk_size=8,
            compression_rate=0.7
        )
    
    @pytest.fixture
    def sample_passages(self):
        """Sample passages for testing"""
        return [
            "Large Language Models have shown remarkable capabilities in natural language processing.",
            "Retrieval-augmented generation combines the power of retrievers with language models.",
            "REFRAG introduces efficient compression techniques for RAG systems.",
            "The method achieves significant speedup without sacrificing accuracy.",
        ]
    
    def test_initialization(self, refrag_model):
        """Test model initialization"""
        assert refrag_model.chunk_size == 8
        assert refrag_model.compression_rate == 0.7
        assert refrag_model.tokenizer is not None
        assert refrag_model.encoder is not None
        assert refrag_model.chunk_projection is not None
        assert refrag_model.policy_network is not None
    
    def test_chunk_text(self, refrag_model):
        """Test text chunking functionality"""
        text = "This is a test sentence with multiple words for chunking."
        chunks = refrag_model.chunk_text(text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Check that chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_encode_chunks(self, refrag_model):
        """Test chunk encoding"""
        chunks = ["Test chunk one", "Test chunk two", "Test chunk three"]
        embeddings = refrag_model.encode_chunks(chunks)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == refrag_model.encoder_config.hidden_size
    
    def test_selective_compression_policy(self, refrag_model):
        """Test selective compression policy"""
        # Create dummy embeddings
        num_chunks = 10
        embeddings = torch.randn(num_chunks, refrag_model.encoder_config.hidden_size)
        
        selection_mask = refrag_model.selective_compression_policy(embeddings)
        
        assert isinstance(selection_mask, torch.Tensor)
        assert selection_mask.dtype == torch.bool
        assert len(selection_mask) == num_chunks
        
        # Check that compression rate is approximately respected
        expected_expanded = int(num_chunks * (1 - refrag_model.compression_rate))
        actual_expanded = selection_mask.sum().item()
        
        # Allow for some flexibility due to rounding
        assert abs(actual_expanded - expected_expanded) <= 1
    
    def test_compress_context_empty(self, refrag_model):
        """Test compression with empty context"""
        result = refrag_model.compress_context([])
        
        assert result['compressed_embeddings'].numel() == 0
        assert result['expanded_chunks'] == []
        assert result['compression_stats']['total_chunks'] == 0
    
    def test_compress_context_normal(self, refrag_model, sample_passages):
        """Test normal compression functionality"""
        result = refrag_model.compress_context(sample_passages)
        
        # Check structure
        assert 'compressed_embeddings' in result
        assert 'expanded_chunks' in result
        assert 'compression_stats' in result
        assert 'expansion_mask' in result
        assert 'original_chunks' in result
        
        # Check compression stats
        stats = result['compression_stats']
        assert stats['total_chunks'] > 0
        assert stats['compressed'] + stats['expanded'] == stats['total_chunks']
        assert 0 <= stats['compression_ratio'] <= 1
        
        # Check embeddings shape
        if result['compressed_embeddings'].numel() > 0:
            assert result['compressed_embeddings'].shape[1] == refrag_model.decoder_hidden_size
    
    def test_estimate_latency_savings(self, refrag_model, sample_passages):
        """Test latency savings estimation"""
        original_text = " ".join(sample_passages)
        original_length = len(refrag_model.tokenizer.tokenize(original_text))
        
        compressed_result = refrag_model.compress_context(sample_passages)
        latency_savings = refrag_model.estimate_latency_savings(original_length, compressed_result)
        
        assert 'ttft_improvement' in latency_savings
        assert 'memory_savings' in latency_savings
        assert 'token_reduction' in latency_savings
        assert 'compression_factor' in latency_savings
        
        # Check reasonable values
        assert latency_savings['ttft_improvement'] >= 1.0
        assert 0 <= latency_savings['memory_savings'] <= 1.0
        assert latency_savings['token_reduction'] >= 0
        assert 0 <= latency_savings['compression_factor'] <= 1.0
    
    def test_compression_consistency(self, refrag_model, sample_passages):
        """Test that compression results are consistent"""
        result1 = refrag_model.compress_context(sample_passages)
        result2 = refrag_model.compress_context(sample_passages)
        
        # Results should be deterministic (assuming no randomness in policy)
        assert result1['compression_stats']['total_chunks'] == result2['compression_stats']['total_chunks']
        # Note: exact equality might not hold due to potential randomness in policy network


class TestREFRAGTrainingSimulator:
    """Test cases for REFRAG training simulator"""
    
    @pytest.fixture
    def simulator(self):
        """Create a training simulator for testing"""
        refrag_model = SimpleREFRAG(chunk_size=4, compression_rate=0.5)
        return REFRAGTrainingSimulator(refrag_model)
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.model is not None
        assert isinstance(simulator.model, SimpleREFRAG)
    
    def test_simulate_curriculum_learning(self, simulator):
        """Test curriculum learning simulation (should run without errors)"""
        try:
            simulator.simulate_curriculum_learning()
            assert True  # If we get here, no exceptions were raised
        except Exception as e:
            pytest.fail(f"Curriculum learning simulation failed: {e}")
    
    def test_estimate_training_requirements(self, simulator):
        """Test training requirements estimation (should run without errors)"""
        try:
            simulator.estimate_training_requirements()
            assert True  # If we get here, no exceptions were raised
        except Exception as e:
            pytest.fail(f"Training requirements estimation failed: {e}")


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_demo(self):
        """Test that the main demonstration runs without errors"""
        try:
            from refrag import demonstrate_refrag
            refrag_model, results = demonstrate_refrag()
            
            assert refrag_model is not None
            assert isinstance(results, dict)
            assert 'compression_stats' in results
            
        except Exception as e:
            pytest.fail(f"End-to-end demo failed: {e}")
    
    def test_different_model_configurations(self):
        """Test different model configurations"""
        configs = [
            {"chunk_size": 4, "compression_rate": 0.5},
            {"chunk_size": 16, "compression_rate": 0.8},
            {"chunk_size": 32, "compression_rate": 0.9},
        ]
        
        sample_text = ["This is a test passage for different configurations."]
        
        for config in configs:
            model = SimpleREFRAG(**config)
            result = model.compress_context(sample_text)
            
            assert result is not None
            assert 'compression_stats' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])