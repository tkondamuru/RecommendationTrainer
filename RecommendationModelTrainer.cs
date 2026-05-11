using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace RecommendationTrainer
{
    public class OrderData
    {
        [LoadColumn(0)] public string GlassId;
        [LoadColumn(1)] public string SundryId;
        [LoadColumn(2)] public float Label;
    }

    public class RecommendationModelTrainer
    {
        private readonly MLContext _mlContext;

        public RecommendationModelTrainer()
        {
            _mlContext = new MLContext();
        }

        public void TrainAndSaveModel(string dataPath, string modelPath)
        {
            if (!File.Exists(dataPath))
            {
                throw new FileNotFoundException($"Training data not found at {dataPath}");
            }

            Console.WriteLine($"Training model using data from: {dataPath}...");

            IDataView data = _mlContext.Data.LoadFromTextFile<OrderData>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("G", nameof(OrderData.GlassId))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("S", nameof(OrderData.SundryId)))
                .Append(_mlContext.Recommendation().Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options
                {
                    MatrixColumnIndexColumnName = "G",
                    MatrixRowIndexColumnName = "S",
                    LabelColumnName = "Label",
                    NumberOfIterations = 50,
                    ApproximationRank = 64,
                    LearningRate = 0.01,
                    LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                    Alpha = 1,
                    C = 0.00001
                }));

            Console.WriteLine("Fitting the model (Matrix Factorization)...");
            ITransformer trainedModel = pipeline.Fit(data);

            Console.WriteLine($"Saving model to: {modelPath}");
            _mlContext.Model.Save(trainedModel, data.Schema, modelPath);
            Console.WriteLine("Model training complete.");
        }
    }
}
