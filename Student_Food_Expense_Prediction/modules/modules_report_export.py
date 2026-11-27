import csv

def export_prediction_report(output_path, predictions):
    """
    Save the list of prediction results into a CSV report file.
    """
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sample Number", "Predicted Daily Food Expense (PHP)"])
        for idx, value in enumerate(predictions, start=1):
            writer.writerow([idx, round(value, 2)])
            