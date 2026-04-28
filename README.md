# IFSA x IBM workshop: watsonx.ai student tutorials

This repository contains hands-on tutorials for learning AI/ML development with IBM watsonx.ai platform. Each tutorial guides you through building practical AI systems from data preparation to deployment.

## 📚 Tutorials

### 1. AutoAI: Train and Deploy ML Models
**Location:** [`AutoAI/`](AutoAI/)

Learn to build and deploy machine learning models using IBM watsonx.ai's automated machine learning capabilities.

**What You'll Learn:**
- Train models using AutoAI (automated machine learning)
- Deploy models to production via deployment spaces
- Call deployed models via API from Jupyter notebooks
- Evaluate and compare model performance

**Key Technologies:**
- IBM watsonx.ai AutoAI
- watsonx Machine Learning
- Jupyter Notebooks

**Use Case:** Housing price prediction using regression models

---

### 2. MultimodalRAG: AI-Powered Document Q&A
**Location:** [`MultimodalRAG/`](MultimodalRAG/)

Build a Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents containing text, tables, and images.

**What You'll Learn:**
- Parse PDFs and extract multimodal content using Docling
- Use IBM Granite vision models for image understanding
- Build vector databases for semantic search
- Create end-to-end RAG pipelines with LangChain

**Key Technologies:**
- IBM Granite AI models (language + vision)
- Docling document processing
- LangChain framework
- ChromaDB vector database

**Use Case:** Intelligent document question-answering system

---

## 🚀 Getting Started

### Prerequisites

1. **IBM Cloud Account** with access to watsonx.ai
2. **IBM Cloud API Key** (for AutoAI tutorial)
3. **Replicate Account** with API token (for MultimodalRAG tutorial)

### Quick Start

1. Clone this repository
2. Navigate to the tutorial folder of your choice
3. Follow the detailed README.md in each tutorial directory
4. Open the provided Jupyter notebooks in watsonx.ai Studio

## 📁 Repository Structure

```
IFSA-students/
├── README.md                    # This file
├── AutoAI/                      # AutoAI tutorial
│   ├── README.md               # Detailed AutoAI guide
│   ├── data/                   # Training datasets
│   ├── img/                    # Tutorial screenshots
│   └── src/                    # Jupyter notebooks
└── MultimodalRAG/              # Multimodal RAG tutorial
    ├── README.md               # Detailed RAG guide
    ├── data/                   # Sample PDF documents
    └── src/                    # Jupyter notebooks
```

## 🎯 Learning Path

**Recommended Order:**

1. **Start with AutoAI** - Learn the fundamentals of model training and deployment
2. **Progress to MultimodalRAG** - Build advanced AI systems with multiple modalities

Both tutorials are self-contained and can be completed independently.

## 📖 Additional Resources

- [IBM watsonx.ai Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)
- [IBM Granite Models](https://www.ibm.com/granite/docs/models/granite/)
- [LangChain Documentation](https://python.langchain.com/)
- [Docling Documentation](https://docling-project.github.io/docling/)

## 🆘 Support

For questions or issues:
1. Check the tutorial-specific README.md files
2. Review the troubleshooting sections in each tutorial
3. Consult the IBM watsonx.ai documentation
4. Contact your instructor or teaching assistant

---

**Happy Learning!** 🎓


---

**Author**: Jan Garček