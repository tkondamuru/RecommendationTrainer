# RecommendationTrainer

An ML.NET and Azure OpenAI-powered engine that analyzes automotive sales patterns to recommend relevant parts and accessories for specific vehicle fits.

## Overview

This project implements a high-performance recommendation system for automotive parts. It leverages **ML.NET Matrix Factorization** for identifying product relationships from sales history and **Azure OpenAI** to generate professional, brand-aware narratives for the recommendations.

## Project Structure

- **Program.cs**: Main entry point and web host configuration.
- **RecommendationEngine.cs**: Core logic for generating recommendations using the trained ML model and LLM enrichment.
- **RecommendationModelTrainer.cs**: Logic for training the Matrix Factorization model.
- **DataExtractorSamples/**: Sample datasets for training and validation.
- **docs/**: Detailed documentation on data and algorithms.

## Documentation

For more detailed information, please refer to the following documents:

- [Data Preprocessing](docs/data_preprocessing.md): Overview of the datasets and cleaning processes.
- [Matrix Factorization](docs/matrix_factorization.md): Explanation of the ML model, parameters, and loss functions.
- [Recommendation Concepts](docs/recommendation_concepts.md): Deep dive into the recommendation logic and architecture.

## Getting Started

### Prerequisites

- .NET 9.0 SDK
- Azure OpenAI Service API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tkondamuru/RecommendationTrainer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RecommendationTrainer
   ```
3. Run the application:
   ```bash
   dotnet run
   ```

## License

This project is proprietary and for internal use only.
