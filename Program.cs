using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;

namespace RecommendationTrainer
{
    class Program
    {
        static string AzureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? "https://eastus.api.cognitive.microsoft.com";
        static string AzureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY") ?? "";
        static string DeploymentName = "gpt-4o-mini";

        static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);
            
            // Register Engine as Singleton
            var engine = new RecommendationEngine(AzureEndpoint, AzureKey, DeploymentName);
            builder.Services.AddSingleton(engine);

            var app = builder.Build();

            // 1. Training Logic
            TrainModels();

            // 2. Initialize Engine Data
            engine.LoadCatalog("DataExtractor/all_sundries.csv", "DataExtractor/part_vehicles.tsv", "DataExtractor/autoparts.csv");
            engine.LoadRegionalSundryIds("US", "recommendation_training_US.csv");
            engine.LoadRegionalSundryIds("CA", "recommendation_training_CA.csv");
            engine.LoadModel("US", "recommendation_model_US.zip");
            engine.LoadModel("CA", "recommendation_model_CA.zip");

            // 3. API Endpoints
            app.MapGet("/api/search", (string q, RecommendationEngine eng) => {
                if (string.IsNullOrEmpty(q)) return Results.Ok(new object[0]);
                var results = eng.AutoSuggestParts
                    .Where(p => p.Desc.Contains(q, StringComparison.OrdinalIgnoreCase))
                    .Take(10)
                    .ToList();
                return Results.Ok(results);
            });

            app.MapPost("/api/recommend", async (RecommendationRequest req, RecommendationEngine eng) => {
                if (string.IsNullOrEmpty(req.GlassId)) return Results.BadRequest("Missing GlassId");
                var region = req.Region ?? "US";
                var recResult = await eng.GetRecommendations(req.GlassId, region);
                
                // Add part info to results for UI
                var enriched = recResult.Items.Select(r => {
                    var id = r.Id.ToString();
                    var info = eng.Catalog.GetValueOrDefault(id, new PartInfo { Id = id, Desc = "Unknown Accessory" });
                    return new {
                        item = r,
                        info = info
                    };
                }).ToList();

                return Results.Ok(new {
                    results = enriched,
                    trace = new {
                        request = recResult.RawRequest,
                        response = recResult.RawResponse
                    }
                });
            });

            app.MapGet("/api/prompt", (RecommendationEngine eng) => Results.Ok(new { prompt = eng.GetSystemPrompt() }));

            // 4. Static Files (UI)
            app.UseDefaultFiles();
            app.UseStaticFiles();

            Console.WriteLine("\n>>> RECOMMENDATION ENGINE DEMO STARTING <<<");
            Console.WriteLine("URL: http://localhost:5000");
            app.Run("http://localhost:5000");
        }

        static void TrainModels()
        {
            string[] regions = { "US", "CA" };
            var trainer = new RecommendationModelTrainer();

            foreach (var region in regions)
            {
                string dataPath = $"recommendation_training_{region}.csv";
                string modelPath = $"recommendation_model_{region}.zip";

                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"[TRAIN] Building {region} model...");
                    try {
                        trainer.TrainAndSaveModel(dataPath, modelPath);
                    } catch (Exception ex) {
                        Console.WriteLine($"[ERROR] Training {region} failed: {ex.Message}");
                    }
                }
                else
                {
                    Console.WriteLine($"[INFO] {region} model already exists.");
                }
            }
        }
    }

    public class RecommendationRequest
    {
        public string? GlassId { get; set; }
        public string? Region { get; set; }
    }
}

