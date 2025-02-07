import json
import tensorflow as tf
import numpy as np

def load_classes(json_file):
    """
    Opens the JSON file and extracts unique class labels from the "org_text" field.
    Returns a sorted list of unique class names.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract unique org_text values; sorting ensures consistent order.
    classes = sorted({record["org_text"] for record in data})
    return classes

def main():
    # Automatically load classes from the JSON file.
    json_file = "MS-ASL/MSASL_val.json"  # Adjust path as needed.
    classes = load_classes(json_file)
    
    # Load the trained model (which includes text preprocessing).
    model = tf.keras.models.load_model("json_training_model.h5")
    
    print("Text Classification App")
    print("Classes loaded from JSON:")
    print(classes)
    print("\nEnter an input text (to be classified based on org_text). Type 'q' to quit.\n")
    
    while True:
        user_input = input("Input text: ")
        if user_input.lower() == 'q':
            break
        
        # The model is expected to handle raw string input.
        prediction = model.predict([user_input], verbose=0)
        pred_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][pred_index]
        
        print(f"Predicted class: {classes[pred_index]} (confidence: {confidence:.2f})\n")

if __name__ == "__main__":
    main()
