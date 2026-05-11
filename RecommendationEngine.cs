using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Azure;
using Azure.AI.OpenAI;

namespace RecommendationTrainer
{
    public class OrderEntry { public string GlassId { get; set; } = ""; public string SundryId { get; set; } = ""; }
    public class Prediction { public float Score { get; set; } }
    public class PartInfo
    {
        public string Id { get; set; } = "";
        public string Desc { get; set; } = ""; // Part Number
        public string DetailedDesc { get; set; } = ""; // Human readable description
        public string VehicleInfo { get; set; } = "";
    }

    public class AIRecResponse
    {
        [JsonPropertyName("recommendations")]
        public List<AIRecItem> Items { get; set; } = new();
    }

    public class AIRecItem
    {
        [JsonPropertyName("id")] public JsonElement Id { get; set; }
        [JsonPropertyName("confidence")] public JsonElement Confidence { get; set; }
        [JsonPropertyName("suggestion")] public string Suggestion { get; set; } = "";
        [JsonPropertyName("reason")] public string Reason { get; set; } = "";
        [JsonPropertyName("show")] public bool Show { get; set; }
    }

    public class AIRecResult
    {
        public List<AIRecItem> Items { get; set; } = new();
        public string RawRequest { get; set; } = "";
        public string RawResponse { get; set; } = "";
    }

    public class RecommendationEngine
    {
        private readonly MLContext _mlContext;
        private readonly Dictionary<string, ITransformer> _models = new();
        private readonly string _azureEndpoint;
        private readonly string _azureKey;
        private readonly string _deploymentName;

        public Dictionary<string, PartInfo> Catalog { get; private set; } = new();
        public Dictionary<string, List<string>> RegionalSundryIds { get; private set; } = new();
        public List<PartInfo> AutoSuggestParts { get; private set; } = new();

        public RecommendationEngine(string azureEndpoint, string azureKey, string deploymentName)
        {
            _mlContext = new MLContext();
            _azureEndpoint = azureEndpoint;
            _azureKey = azureKey;
            _deploymentName = deploymentName;
        }

        public void LoadModel(string region, string modelPath)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found at {modelPath}");
            _models[region.ToUpper()] = _mlContext.Model.Load(modelPath, out _);
        }

        public void LoadCatalog(string sundryPath, string vehiclePath, string autoSuggestPath)
        {
            Console.WriteLine("Loading Catalogs...");
            if (File.Exists(sundryPath))
            {
                foreach (var line in File.ReadAllLines(sundryPath).Skip(1))
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 2)
                    {
                        string id = parts[0].Trim('"', ' ');
                        Catalog[id] = new PartInfo { Id = id, Desc = parts[1].Trim('"', ' '), DetailedDesc = parts.Length >= 3 ? parts[2].Trim('"', ' ') : "" };
                    }
                }
            }
            if (File.Exists(vehiclePath))
            {
                foreach (var line in File.ReadAllLines(vehiclePath).Skip(1))
                {
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        var id = parts[0].Trim();
                        var info = parts[1].Trim('"', ' ');
                        if (!Catalog.ContainsKey(id)) Catalog[id] = new PartInfo { Id = id, Desc = info, VehicleInfo = info };
                        else Catalog[id].VehicleInfo = info;
                    }
                }
            }
            if (File.Exists(autoSuggestPath))
            {
                foreach (var line in File.ReadAllLines(autoSuggestPath).Skip(1))
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 2)
                    {
                        AutoSuggestParts.Add(new PartInfo { Id = parts[0].Trim('"', ' '), Desc = parts[1].Trim('"', ' ') });
                    }
                }
            }
        }

        public void LoadRegionalSundryIds(string region, string dataPath)
        {
            if (!File.Exists(dataPath)) return;
            var lines = File.ReadAllLines(dataPath).Skip(1);
            var sundries = new HashSet<string>();
            foreach (var line in lines)
            {
                var parts = line.Split(',');
                if (parts.Length >= 2) sundries.Add(parts[1]);
            }
            RegionalSundryIds[region.ToUpper()] = sundries.ToList();
        }

        public string GetSystemPrompt()
        {
            return "You are a Senior Automotive Parts Manager. " +
                   "TASK: Return a JSON object with 'recommendations' (EXACTLY 10 items). " +
                   "1. MATH: Score 1.2+ = 95-99% confidence. Score 1.0 = ~85%. Score 0.0 = <60% (General Expert Suggestion). " +
                   "2. DATA TRUTH: If RawPatternWeight is 0.0000, you MUST NOT use high confidence (>70%). These are 'General Expert Suggestions'. " +
                   "3. BRAND GUARDRAIL: If desc has a car brand (Subaru, Ford), it MUST match target. Skip if mismatched. " +
                   "4. UNIVERSAL: Consumables (Urethane, Primers, Wipes) are brand-agnostic. " +
                   "5. VOICE: Write a natural, punchy sales pitch for the 'suggestion' field. " +
                   "JSON: id, confidence (0-100), suggestion, reason, show (bool).";
        }

        public async Task<AIRecResult> GetRecommendations(string glassId, string region)
        {
            region = region.ToUpper();
            var result = new AIRecResult();
            var glass = Catalog.GetValueOrDefault(glassId);
            
            if (glass == null) 
            {
                Console.WriteLine($"[Engine] Glass ID '{glassId}' not found in Catalog.");
                return result;
            }
            
            if (!_models.ContainsKey(region)) 
            {
                Console.WriteLine($"[Engine] Model for region '{region}' not loaded.");
                return result;
            }

            var model = _models[region];
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<OrderEntry, Prediction>(model);
            var sundryIds = RegionalSundryIds.GetValueOrDefault(region, new List<string>());

            if (sundryIds.Count == 0)
            {
                Console.WriteLine($"[Engine] No candidate sundries found for region '{region}'.");
            }

            var rawResults = sundryIds.Select(sId => {
                var p = predictionEngine.Predict(new OrderEntry { GlassId = glassId, SundryId = sId });
                return new { Id = sId, Score = float.IsNaN(p.Score) ? 0 : p.Score };
            })
            .OrderByDescending(x => x.Score)
            .Take(10)
            .ToList();

            if (rawResults.Count == 0) 
            {
                Console.WriteLine($"[Engine] ML Model returned 0 candidates for Glass '{glassId}'.");
                return result;
            }

            Console.WriteLine($"[Engine] Found {rawResults.Count} candidates via ML. Top Score: {rawResults[0].Score:F4}");

            string prompt = $"Target Glass: {glass.VehicleInfo}\n\nCandidate Accessories:\n";
            foreach (var res in rawResults)
            {
                var info = Catalog.GetValueOrDefault(res.Id, new PartInfo { Id = res.Id, Desc = "Unknown" });
                prompt += $"- ID: {res.Id} | PartNo: {info.Desc} | Description: {info.DetailedDesc} | RawPatternWeight: {res.Score:F4}\n";
            }

            result.RawRequest = prompt;
            var (items, rawResp) = await GetAINarratives(prompt);
            result.Items = items;
            result.RawResponse = rawResp;
            return result;
        }

        private async Task<(List<AIRecItem>, string)> GetAINarratives(string prompt)
        {
            string rawResponse = "";
            try
            {
                OpenAIClient client = new OpenAIClient(new Uri(_azureEndpoint), new AzureKeyCredential(_azureKey));
                var options = new ChatCompletionsOptions()
                {
                    DeploymentName = _deploymentName,
                    MaxTokens = 1500,
                    Messages = {
                        new ChatRequestSystemMessage(GetSystemPrompt()),
                        new ChatRequestUserMessage(prompt)
                    }
                };

                Console.WriteLine("\n--- AI REQUEST START ---");
                Console.WriteLine("System Prompt:\n" + GetSystemPrompt());
                Console.WriteLine("User Prompt:\n" + prompt);
                Console.WriteLine("--- AI REQUEST END ---\n");

                Response<ChatCompletions> response = await client.GetChatCompletionsAsync(options);
                rawResponse = response.Value.Choices[0].Message.Content;

                Console.WriteLine("\n--- AI RESPONSE START ---");
                Console.WriteLine(rawResponse);
                Console.WriteLine("--- AI RESPONSE END ---\n");

                int startBrace = rawResponse.IndexOf('{');
                int startBracket = rawResponse.IndexOf('[');
                int start = (startBrace != -1 && startBracket != -1) ? Math.Min(startBrace, startBracket) : Math.Max(startBrace, startBracket);
                int endBrace = rawResponse.LastIndexOf('}');
                int endBracket = rawResponse.LastIndexOf(']');
                int end = Math.Max(endBrace, endBracket);
                
                if (start == -1 || end == -1) throw new Exception("No JSON found.");
                string json = rawResponse.Substring(start, (end - start) + 1);

                var optionsJson = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                AIRecResponse result;
                if (json.Trim().StartsWith("[")) result = new AIRecResponse { Items = JsonSerializer.Deserialize<List<AIRecItem>>(json, optionsJson) };
                else result = JsonSerializer.Deserialize<AIRecResponse>(json, optionsJson);

                return (result?.Items ?? new List<AIRecItem>(), rawResponse);
            }
            catch (Exception ex)
            {
                Console.WriteLine("AI Error: " + ex.Message);
                return (new List<AIRecItem>(), rawResponse);
            }
        }
    }
}
