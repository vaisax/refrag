# REFRAG: Rethinking RAG based Decoding - Simple Implementation

A simplified implementation of REFRAG (REpresentation For RAG), a novel approach for efficient Retrieval-Augmented Generation (RAG) that significantly reduces inference latency while maintaining performance.

## Overview

REFRAG addresses the fundamental challenge of high latency in RAG systems by exploiting the sparse attention patterns inherent in retrieved passages. Instead of processing all tokens individually, REFRAG compresses context chunks using a lightweight encoder and selectively expands only the most important chunks.

### Key Features

- ** 30.85× TTFT Acceleration**: Dramatic reduction in Time-To-First-Token
- ** Memory Efficient**: Significant reduction in KV cache requirements
- ** Selective Compression**: Smart policy determines which chunks need full representation
- ** Flexible Architecture**: Works with existing decoder models

## Architecture

```
Context Passages → Chunking → Lightweight Encoder → Chunk Embeddings
                                     ↓
Query Tokens → Decoder ← Projection Layer ← Selective Policy
                ↓
           Generated Response
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/refrag-simple
cd refrag-simple
pip install -r requirements.txt
```

### Basic Usage

```python
from refrag import SimpleREFRAG

# Initialize REFRAG
refrag = SimpleREFRAG(
    chunk_size=16,
    compression_rate=0.8  # Compress 80% of chunks
)

# Sample retrieved passages
passages = [
    "Large Language Models have shown remarkable capabilities...",
    "RAG systems require specialized consideration...",
    # ... more passages
]

# Compress context
compressed_result = refrag.compress_context(passages)

print(f"Compression ratio: {compressed_result['compression_stats']['compression_ratio']:.2%}")
print(f"Estimated TTFT improvement: {compressed_result['ttft_improvement']:.2f}x")
```

### Run Demo

```bash
python refrag.py
```

This will run a complete demonstration showing:
- Context compression process
- Performance improvements
- Training simulation

## Performance

Based on the original paper results:

| Metric | Improvement |
|--------|-------------|
| TTFT Acceleration | 30.85× |
| Memory Usage | ~80% reduction |
| Context Extension | 16× longer contexts |
| Accuracy | No degradation |

## Configuration

### Model Parameters

```python
refrag = SimpleREFRAG(
    decoder_model_name="microsoft/DialoGPT-small",  # Base decoder
    encoder_model_name="distilbert-base-uncased",   # Lightweight encoder
    chunk_size=16,                                  # Tokens per chunk
    compression_rate=0.8,                           # Fraction to compress
)
```

### Training Configuration

The implementation includes simulation of the three-stage training process:

1. **Reconstruction Task**: Learn chunk compression with minimal information loss
2. **Continual Pre-training (CPT)**: Align encoder-decoder representations
3. **RL Policy Training**: Learn optimal selective compression

## Project Structure

```
refrag-simple/
├── refrag.py              # Main implementation
├── requirements.txt       # Dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore           # Git ignore rules
├── setup.py             # Package setup
├── examples/            # Usage examples
│   └── demo.py         # Demonstration script
├── tests/               # Unit tests
│   └── test_refrag.py  # Test cases
└── docs/                # Documentation
    └── architecture.md  # Detailed architecture
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=refrag
```

## Benchmarking

To benchmark against standard RAG approaches:

```python
from refrag import SimpleREFRAG
import time

# Standard approach (simulated)
start_time = time.time()
# ... process full context ...
standard_time = time.time() - start_time

# REFRAG approach
start_time = time.time()
compressed_result = refrag.compress_context(passages)
refrag_time = time.time() - start_time

speedup = standard_time / refrag_time
print(f"Speedup: {speedup:.2f}x")
```

## Research Paper

This implementation is based on the paper:
**"REFRAG: Rethinking RAG based Decoding"** by Xiaoqiang Lin et al.

Key contributions from the paper:
- Analysis of block-diagonal attention patterns in RAG
- Curriculum learning approach for encoder-decoder alignment
- RL-based selective compression policy
- Comprehensive evaluation across multiple benchmarks

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/refrag-simple
cd refrag-simple

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Limitations

This is a **simplified implementation** for educational and research purposes. For production use, consider:

- Full curriculum learning training pipeline
- Proper RL optimization with PPO/GRPO
- Integration with larger language models (LLaMA, etc.)
- Extensive evaluation on RAG benchmarks
- Optimization for specific hardware

## Roadmap

- [ ] Full training pipeline implementation
- [ ] Integration with Hugging Face Transformers
- [ ] Support for more encoder/decoder combinations
- [ ] Benchmarking suite
- [ ] Production optimizations
- [ ] Multi-GPU training support

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{lin2025refrag,
  title={REFRAG: Rethinking RAG based Decoding},
  author={Lin, Xiaoqiang and Ghosh, Aritra and Low, Bryan Kian Hsiang and Shrivastava, Anshumali and Mohan, Vijai},
  journal={arXiv preprint arXiv:2509.01092},
  year={2025}
}
```

## Acknowledgments

- Original REFRAG paper authors at Meta AI and collaborating institutions
- Hugging Face for the transformers library
- PyTorch team for the deep learning framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

** Star this repo if you find it useful!**