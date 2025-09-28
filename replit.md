# Overview

This is a Streamlit-based GGUF AI Model Runner application that provides a web interface for uploading, loading, and interacting with AI models in GGUF format. The application supports text generation, conversation management, batch processing, and performance monitoring. It's designed to run AI models locally using the llama-cpp-python library, with graceful degradation when dependencies are not available.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Framework**: Provides the interactive web interface with real-time state management
- **Session State Management**: Maintains conversation history, model state, and user preferences across interactions
- **Component-Based UI**: Modular interface components for chat, model management, and batch processing

## Backend Architecture
- **Model Management Layer**: Handles GGUF model loading, unloading, and configuration through the ModelManager class
- **Chat Interface**: Manages conversation flow, message formatting, and history tracking
- **Batch Processing Engine**: Supports bulk text processing with progress tracking and error handling
- **Performance Monitoring**: Real-time system resource monitoring during model inference

## Core Components
- **ModelManager**: Centralized model lifecycle management with support for different model types (text, code, multimodal)
- **ChatInterface**: Conversation state management and message formatting for model input
- **BatchProcessor**: Asynchronous batch processing with status tracking and error recovery
- **PerformanceMonitor**: System resource monitoring including CPU, memory, and generation timing metrics
- **Utility Functions**: File validation, size formatting, and conversation export functionality

## Design Patterns
- **Graceful Degradation**: Application functions as a UI demo when llama-cpp-python is unavailable
- **Session State Pattern**: Streamlit session state for maintaining application state across interactions
- **Factory Pattern**: Model type configuration with different system prompts and behaviors
- **Observer Pattern**: Performance monitoring with real-time metric collection

## Error Handling
- **Dependency Checking**: Runtime detection of llama-cpp-python availability with user-friendly messaging
- **File Validation**: GGUF file format validation before model loading attempts
- **Resource Management**: Automatic model cleanup and memory management

# External Dependencies

## Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **llama-cpp-python**: GGUF model loading and inference engine (optional with graceful fallback)

## System Dependencies
- **psutil**: System resource monitoring for performance metrics
- **threading**: Background processing for performance monitoring and batch operations

## File Format Support
- **GGUF Models**: Primary model format with magic byte validation
- **JSON/CSV**: Batch processing input formats
- **Text Files**: Simple text-based batch input support

## Python Standard Library
- **json**: Configuration and data serialization
- **os**: File system operations and environment detection
- **time/datetime**: Performance timing and logging
- **typing**: Type hints for better code maintainability