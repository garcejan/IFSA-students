# Build an AI-Powered Multimodal RAG System with IBM Granite and Docling

## 📚 Assignment Overview

In this assignment, you will build a **Multimodal Retrieval-Augmented Generation (RAG)** system that can process and answer questions from PDF documents containing text, tables, and images. You'll use IBM Granite AI models, Docling for document processing, and LangChain for workflow orchestration.

### What You'll Learn

- **Document Processing**: Parse PDFs and extract text, tables, and images using Docling
- **Multimodal AI**: Use IBM Granite vision models to understand images and generate descriptions
- **Vector Databases**: Store and retrieve document embeddings for semantic search
- **RAG Pipeline**: Build a complete retrieval-augmented generation system
- **LangChain Integration**: Orchestrate AI workflows with LangChain framework

### Technologies Used

1. **[Docling](https://docling-project.github.io/docling/)** - Open-source document parsing toolkit
2. **[IBM Granite](https://www.ibm.com/granite/docs/models/granite/)** - State-of-the-art language and vision models
3. **[LangChain](https://github.com/langchain-ai/langchain)** - Framework for building LLM applications

## 📋 Prerequisites

### Knowledge Requirements
- Familiarity with Python programming
- Basic understanding of:
  - Large Language Models (LLMs)
  - Natural Language Processing (NLP) concepts
  - Computer vision basics

### Technical Setup

#### 1. Replicate Account Setup

To use IBM Granite models, you need a Replicate account:

1. **Create Account**: Sign up at [replicate.com](https://replicate.com)

2. **Add Credit (Optional)**: Use [this invitation link](https://replicate.com/invites/a8717bfe-2f3d-4a52-88ed-1356231cdf03) to receive free credits for trying Granite models

3. **Get API Token**: 
   - Visit [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
   - Copy your `REPLICATE_API_TOKEN`
   - Keep it secure - you'll need it when running the notebook

## 🚀 Getting Started

### Step 1: Create Jupyter Notebook in watsonx.ai Studio

1. Open watsonx.ai Studio
2. Create a new Jupyter notebook from URL:
   ```
   https://github.com/garcejan/IFSA-students/blob/main/MultimodalRAG/src/MultimodalRAG.ipynb
   ```
3. **Important**: Select **Runtime 25.1 on Python 3.12 S** (4 vCPU, 16 GB RAM)
   - This runtime is required for processing documents and running AI models

### Step 2: Run the Notebook

The notebook is organized into clear steps:

#### **Step 1: Environment Setup**
- Install required Python packages (Docling, LangChain, etc.)
- Set up logging configuration

#### **Step 2: Model Selection**
- Configure IBM Granite embeddings model for text vectorization
- Set up Granite vision model for image understanding
- Connect to Granite language model via Replicate
- **Action Required**: Enter your `REPLICATE_API_TOKEN` when prompted

#### **Step 3: Document Preparation**
- Use Docling to parse PDF documents from the `data/` folder
- Extract text, tables, and images
- Chunk text into appropriate sizes
- Generate image descriptions using Granite vision model
- Create LangChain documents for all content types

#### **Step 4: Vector Database**
- Set up ChromaDB vector store
- Generate embeddings for all document chunks
- Store embeddings for semantic search

#### **Step 5: RAG Pipeline**
- Build the complete RAG system
- Test with sample queries
- Retrieve relevant context and generate answers

### Step 3: Test Your System

Once the notebook is running, you can:
- Ask questions about the PDF documents in the `data/` folder
- Test multimodal understanding (questions about images, tables, and text)
- Experiment with different queries to see how the RAG system responds

## 📁 Project Structure

```
MultimodalRAG/
├── README.md                          # This file
├── data/                              # PDF documents for processing
│   ├── 1706.03762.pdf                # Sample PDF (Attention paper)
│   └── AR_2020_WEB2.pdf              # Sample PDF (Annual report)
└── src/                               # Source code
    └── MultimodalRAG.ipynb           # Main assignment notebook
```

## 💡 Tips for Success

1. **API Token**: Make sure you have your Replicate API token ready before starting
2. **Runtime**: Use the recommended runtime (Python 3.12 S with 4 vCPU, 16 GB RAM)
3. **Processing Time**: Image processing can take several minutes - be patient
4. **Experimentation**: Try modifying prompts and queries to improve results
5. **Documentation**: Read the markdown cells carefully - they explain each step

## 🔧 Customization Options

You can customize the system by:
- Using different PDF documents (add them to the `sources` sources in the notebook)
- Changing the embeddings model
- Modifying image description prompts
- Adjusting text chunk sizes
- Experimenting with different vector databases

## 📝 Assignment Deliverables

1. Successfully run all notebook cells
2. Test the RAG system with at least 3 different queries
3. Document your results and observations
4. (Optional) Experiment with your own PDF documents

## 🆘 Troubleshooting

**Issue**: API token not working
- **Solution**: Verify you copied the complete token from Replicate
- Check that you have credits in your Replicate account

**Issue**: Out of memory errors
- **Solution**: Ensure you're using the recommended runtime (4 vCPU, 16 GB RAM)
- Try processing fewer documents at once

**Issue**: Slow image processing
- **Solution**: This is normal - vision model processing takes time
- Consider reducing the number of images or using a more powerful runtime

## 📚 Additional Resources

- [IBM Granite Documentation](https://www.ibm.com/granite/docs/models/granite/)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Concepts](https://www.ibm.com/think/topics/retrieval-augmented-generation)

## 🎯 Learning Objectives

By completing this assignment, you will:
- ✅ Understand how multimodal RAG systems work
- ✅ Gain hands-on experience with document processing
- ✅ Learn to integrate multiple AI models in a pipeline
- ✅ Build practical skills with LangChain and vector databases
- ✅ Create a system that can answer questions from complex documents

---

**Need Help?** Review the notebook's markdown cells for detailed explanations of each step, consult the additional resources listed above, or ask your advisor.