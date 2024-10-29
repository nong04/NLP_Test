//Source: https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.Analysis;
using Microsoft.ML.TorchSharp;
using System;
using System.IO;
using System.Net;

// Initialize MLContext
var mlContext = new MLContext();
string dataPath = @"..\..\yelp_labelled.csv";
var columnNames = new[] { "Text", "Sentiment" };
var df = DataFrame.LoadCsv(dataPath, header: true, columnNames: columnNames);

//foreach (var row in df.Rows)
//{
//    Console.WriteLine(string.Join("\t", row));
//}

// Split the data into train and test sets.
var trainTestSplit = mlContext.Data.TrainTestSplit(df, testFraction: 0.2);
//Console.WriteLine("Splitted the data");

//Define your training pipeline
var pipeline =
        mlContext.Transforms.Conversion.MapValueToKey("Label", "Sentiment")
            .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "Text"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
//Console.WriteLine("Defined the pipeline");

// Train the model
var model = pipeline.Fit(trainTestSplit.TrainSet);
//Console.WriteLine("Trained the model");

// Use the model to make predictions
var predictionIDV = model.Transform(trainTestSplit.TestSet);
//Console.WriteLine("Used the model");

// The result of calling Transform is an IDataView with your predicted values. To make it easier to view your predictions, convert the IDataView to an IDataFrame.
// In this case, the only columns that I'm interested in are the Text, Sentiment (actual value), and PredictedLabel (predicted value).
var columnsToSelect = new[] { "Text", "Sentiment", "PredictedLabel" };

var predictions = predictionIDV.ToDataFrame(columnsToSelect);
//Console.WriteLine("Predicted");

//Console.WriteLine("Text\tSentiment\tPredictedLabel");
//foreach (var row in predictions.Rows)
//{
//    Console.WriteLine(string.Join("\t", row));
//}

// Evaluate the model
var evaluationMetrics =
    mlContext
        .MulticlassClassification
        .Evaluate(predictionIDV);

Console.WriteLine($"MacroAccuracy: {evaluationMetrics.MacroAccuracy:F2}");
Console.WriteLine($"MicroAccuracy: {evaluationMetrics.MicroAccuracy:F2}");
Console.WriteLine($"LogLoss: {evaluationMetrics.LogLoss:F2}");
Console.WriteLine($"LogLossReduction: {evaluationMetrics.LogLossReduction:F2}");

for (int i = 0; i < evaluationMetrics.PerClassLogLoss.Count; i++)
{
    Console.WriteLine($"LogLoss cho nhãn {i}: {evaluationMetrics.PerClassLogLoss[i]:F2}");
}